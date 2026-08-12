#include "HandTracker.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/gpu/gpu_buffer.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <array>

namespace cv_keyboard {

static constexpr char kInputStream[] = "input_video";
static constexpr char kMultiHandLandmarksStream[] = "multi_hand_landmarks";
static constexpr char kMultiHandednessStream[]   = "multi_handedness";

// ---------------------------------------------------------------------------
// KALMAN FILTER LOGIC
// ---------------------------------------------------------------------------
struct HandSmoother {
    std::array<cv::KalmanFilter, 21> kfs;
    bool initialized = false;
    int frames_unseen = 0;

    void reset() {
        initialized = false;
        frames_unseen = 0;
    }

    void smooth(std::array<Landmark, 21>& landmarks) {
        if (!initialized) {
            for (int i = 0; i < 21; ++i) {
                kfs[i].init(4, 2, 0); // State: [x, y, dx, dy], Measurement: [x, y]
                
                // Constant velocity model
                kfs[i].transitionMatrix = (cv::Mat_<float>(4, 4) <<
                    1, 0, 1, 0,
                    0, 1, 0, 1,
                    0, 0, 1, 0,
                    0, 0, 0, 1);
                
                kfs[i].measurementMatrix = (cv::Mat_<float>(2, 4) <<
                    1, 0, 0, 0,
                    0, 1, 0, 0);

                // ==========================================
                // TUNABLE KALMAN FILTER PARAMETERS
                // ==========================================
                
                // Process Noise (How much we expect the physics to drift)
                // Smaller value = Slower reaction, but extremely smooth.
                // Larger value  = Fast reaction to sudden movements, but more jitter.
                cv::setIdentity(kfs[i].processNoiseCov, cv::Scalar::all(1e-2));
                
                // Measurement Noise (How much we trust MediaPipe)
                // Larger value  = We distrust MediaPipe and rely heavily on our smoothed prediction.
                // Smaller value = We trust MediaPipe and follow its jitter more closely.
                cv::setIdentity(kfs[i].measurementNoiseCov, cv::Scalar::all(1e-2));
                
                cv::setIdentity(kfs[i].errorCovPost, cv::Scalar::all(1));

                // Snap to the very first measurement to prevent sweeping in from [0,0]
                kfs[i].statePost.at<float>(0) = landmarks[i].x;
                kfs[i].statePost.at<float>(1) = landmarks[i].y;
                kfs[i].statePost.at<float>(2) = 0.0f;
                kfs[i].statePost.at<float>(3) = 0.0f;
            }
            initialized = true;
        }

        cv::Mat measurement(2, 1, CV_32F);
        for (int i = 0; i < 21; ++i) {
            kfs[i].predict();

            measurement.at<float>(0) = landmarks[i].x;
            measurement.at<float>(1) = landmarks[i].y;

            cv::Mat estimated = kfs[i].correct(measurement);
            
            // Overwrite the raw MediaPipe coordinate with the smoothed coordinate
            landmarks[i].x = estimated.at<float>(0);
            landmarks[i].y = estimated.at<float>(1);
        }
        frames_unseen = 0;
    }
};


// ---------------------------------------------------------------------------
// PIMPL: hides MediaPipe types from the header
// ---------------------------------------------------------------------------
struct HandTracker::Impl {
    mediapipe::CalculatorGraph graph_;
    bool initialised_ = false;

    std::mutex mutex_;
    std::shared_ptr<std::vector<HandData>> cached_hands_ = std::make_shared<std::vector<HandData>>();
    int64_t latest_timestamp_us_ = 0;

    HandSmoother left_smoother_;
    HandSmoother right_smoother_;

    ~Impl() {
        if (initialised_) {
            auto status = graph_.CloseInputStream(kInputStream);
            if (!status.ok()) {
                std::cerr << "[HandTracker] CloseInputStream error: "
                          << status.message() << "\n";
            }
            graph_.WaitUntilDone().IgnoreError();
        }
    }
};

HandTracker::HandTracker()
    : impl_(std::make_unique<Impl>()) {}

HandTracker::~HandTracker() = default;

bool HandTracker::init() {
    std::string graph_path = "graphs/hand_landmark_tracker.pbtxt";
    const char* env_path = std::getenv("CVKB_GRAPH_PATH");
    if (env_path && env_path[0] != '\0') {
        graph_path = env_path;
    }

    std::string graph_config_contents;
    auto status = mediapipe::file::GetContents(graph_path, &graph_config_contents);
    if (!status.ok()) {
        std::cerr << "[HandTracker] Failed to read graph config from '"
                  << graph_path << "': " << status.message() << "\n";
        return false;
    }

    mediapipe::CalculatorGraphConfig config;
    if (!mediapipe::ParseTextProto<mediapipe::CalculatorGraphConfig>(
            graph_config_contents, &config)) {
        std::cerr << "[HandTracker] Failed to parse graph config proto.\n";
        return false;
    }

    status = impl_->graph_.Initialize(config);
    if (!status.ok()) {
        std::cerr << "[HandTracker] Graph initialisation failed: "
                  << status.message() << "\n";
        return false;
    }

    // --- Asynchronously observe output streams (Non-blocking) ---
    status = impl_->graph_.ObserveOutputStream(
    kMultiHandLandmarksStream,
    [this](const mediapipe::Packet& packet) -> absl::Status {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        
        if (packet.IsEmpty()) {
            impl_->cached_hands_->clear();
            return absl::OkStatus();
        }

        int64_t packet_ts = packet.Timestamp().Value();
        impl_->latest_timestamp_us_ = packet_ts;
        const auto& multi_hand_landmarks =
            packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            
        if (multi_hand_landmarks.empty()) {
            impl_->cached_hands_->clear();
            return absl::OkStatus();
        }

        impl_->cached_hands_->resize(multi_hand_landmarks.size());
        for (size_t h = 0; h < multi_hand_landmarks.size(); ++h) {
            const auto& landmark_list = multi_hand_landmarks[h];
            int count = std::min(landmark_list.landmark_size(), 21);
            for (int i = 0; i < count; ++i) {
                const auto& src = landmark_list.landmark(i);
                (*impl_->cached_hands_)[h].landmarks[i].x = static_cast<float>(src.x());
                (*impl_->cached_hands_)[h].landmarks[i].y = static_cast<float>(src.y());
                (*impl_->cached_hands_)[h].landmarks[i].z = static_cast<float>(src.z());
                (*impl_->cached_hands_)[h].landmarks[i].confidence = static_cast<float>(src.visibility());
            }
            (*impl_->cached_hands_)[h].timestamp_us = packet_ts;
        }
        return absl::OkStatus();
    });

    if (!status.ok()) {
        std::cerr << "[HandTracker] Failed to observe landmarks stream: "
                  << status.message() << "\n";
        return false;
    }

    status = impl_->graph_.ObserveOutputStream(
        kMultiHandednessStream,
        [this](const mediapipe::Packet& packet) -> absl::Status {
            if (packet.IsEmpty()) return absl::OkStatus();

            const auto& multi_handedness =
                packet.Get<std::vector<mediapipe::ClassificationList>>();

            std::lock_guard<std::mutex> lock(impl_->mutex_);
            for (size_t h = 0; h < multi_handedness.size(); ++h) {
                if (h < impl_->cached_hands_->size() && multi_handedness[h].classification_size() > 0) {
                    const auto& cls = multi_handedness[h].classification(0);
                    (*impl_->cached_hands_)[h].hand_confidence = cls.score();
                    
                    if (cls.label() == "Left") {
                        (*impl_->cached_hands_)[h].handedness = 1;
                    } else if (cls.label() == "Right") {
                        (*impl_->cached_hands_)[h].handedness = 2;
                    }
                }
            }
            return absl::OkStatus();
        });

    if (!status.ok()) {
        std::cerr << "[HandTracker] Failed to observe handedness stream: "
                  << status.message() << "\n";
        return false;
    }

    status = impl_->graph_.StartRun({});
    if (!status.ok()) {
        std::cerr << "[HandTracker] Graph StartRun failed: "
                  << status.message() << "\n";
        return false;
    }

    impl_->initialised_ = true;
    std::cout << "[HandTracker] Initialised successfully.\n";
    return true;
}

bool HandTracker::isInitialised() const {
    return impl_->initialised_;
}

int64_t HandTracker::latestTimestamp() const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    return impl_->latest_timestamp_us_;
}

std::shared_ptr<const std::vector<HandData>> HandTracker::detect(const cv::Mat& frame, int64_t timestamp_us) {
    if (!impl_->initialised_) {
        return {};
    }

    auto input_frame = std::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGBA, frame.cols, frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    cv::cvtColor(frame, input_frame_mat, cv::COLOR_BGR2RGBA);

    auto status = impl_->graph_.AddPacketToInputStream(
        kInputStream,
        mediapipe::Adopt(input_frame.release())
            .At(mediapipe::Timestamp(timestamp_us)));

    if (!status.ok()) {
        std::cerr << "[HandTracker] AddPacketToInputStream error: "
                  << status.message() << "\n";
    }

    impl_->graph_.WaitUntilIdle().IgnoreError();

    // Now return the freshly processed hands synchronously
    std::lock_guard<std::mutex> lock(impl_->mutex_);

    bool left_seen = false;
    bool right_seen = false;

    // Apply the correct Kalman filter based on Handedness
    for (auto& hand : *impl_->cached_hands_) {
        if (hand.handedness == 1) { // Left Hand
            impl_->left_smoother_.smooth(hand.landmarks);
            left_seen = true;
        } else if (hand.handedness == 2) { // Right Hand
            impl_->right_smoother_.smooth(hand.landmarks);
            right_seen = true;
        }
    }

    // Reset smoothers if hands leave the frame for too long
    if (!left_seen) {
        if (++impl_->left_smoother_.frames_unseen > 5) impl_->left_smoother_.reset();
    }
    if (!right_seen) {
        if (++impl_->right_smoother_.frames_unseen > 5) impl_->right_smoother_.reset();
    }

    return impl_->cached_hands_;
}

} // namespace cv_keyboard