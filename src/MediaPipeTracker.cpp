#include "MediaPipeTracker.h"

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
// KALMAN FILTER + OPTICAL FLOW FUSION LOGIC (Option A)
// ---------------------------------------------------------------------------
struct HandSmoother {
    std::array<cv::KalmanFilter, 21> kfs;
    std::vector<cv::Point2f> prev_pts; // Stores physical pixel locations for LK Flow
    bool initialized = false;
    int frames_unseen = 0;

    void reset() {
        initialized = false;
        frames_unseen = 0;
        prev_pts.clear();
    }

    void smooth(std::array<Landmark, 21>& landmarks, const cv::Mat& prev_gray, const cv::Mat& curr_gray) {
        float width = static_cast<float>(curr_gray.cols);
        float height = static_cast<float>(curr_gray.rows);

        if (!initialized) {
            prev_pts.resize(21);
            for (int i = 0; i < 21; ++i) {
                // State: [x, y, dx, dy], Measurement: [x_mp, y_mp, dx_of, dy_of]
                kfs[i].init(4, 4, 0); 
                
                // Kinematic Model: P_new = P_old + V_old
                kfs[i].transitionMatrix = (cv::Mat_<float>(4, 4) <<
                    1, 0, 1, 0,
                    0, 1, 0, 1,
                    0, 0, 1, 0,
                    0, 0, 0, 1);
                
                // H Matrix: We are directly measuring all 4 state variables now
                cv::setIdentity(kfs[i].measurementMatrix);

                // Process Noise: How much we expect the actual physics to deviate from pure constant velocity
                cv::setIdentity(kfs[i].processNoiseCov, cv::Scalar::all(1e-5));
                
                // ==========================================
                // FUSION TUNING: R Matrix
                // ==========================================
                // We trust MediaPipe for general position, but it is noisy (1e-3).
                // We trust Optical Flow heavily for short-term velocity (1e-4).
                kfs[i].measurementNoiseCov = (cv::Mat_<float>(4, 4) <<
                    1e-5, 0, 0, 0,     // MediaPipe X Noise
                    0, 1e-5, 0, 0,     // MediaPipe Y Noise
                    0, 0, 1e-4, 0,     // Optical Flow dX Noise
                    0, 0, 0, 1e-4);    // Optical Flow dY Noise
                
                cv::setIdentity(kfs[i].errorCovPost, cv::Scalar::all(1));

                // Snap to initial MediaPipe coordinates
                kfs[i].statePost.at<float>(0) = landmarks[i].x;
                kfs[i].statePost.at<float>(1) = landmarks[i].y;
                kfs[i].statePost.at<float>(2) = 0.0f;
                kfs[i].statePost.at<float>(3) = 0.0f;

                // Save pixel coordinate for LK flow next frame
                prev_pts[i] = cv::Point2f(landmarks[i].x * width, landmarks[i].y * height);
            }
            initialized = true;
            return; // Exit early, no optical flow possible on frame 1
        }

        // 1. Calculate Sparse Optical Flow (Lucas-Kanade)
        std::vector<cv::Point2f> curr_pts;
        std::vector<uchar> status;
        std::vector<float> err;
        
        // 21x21 window, 3 pyramid levels to handle fast typing blur
        cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err, cv::Size(21, 21), 3);

        cv::Mat measurement(4, 1, CV_32F);
        
        for (int i = 0; i < 21; ++i) {
            kfs[i].predict();

            float vx_of = 0.0f;
            float vy_of = 0.0f;

            // 2. Extract Velocity from Optical Flow
            if (status[i]) {
                // Convert pixel displacement back to normalized [0,1] velocity
                vx_of = (curr_pts[i].x - prev_pts[i].x) / width;
                vy_of = (curr_pts[i].y - prev_pts[i].y) / height;
            } else {
                // Fallback: If OF lost the feature due to extreme blur, use the KF's internal prediction
                vx_of = kfs[i].statePre.at<float>(2);
                vy_of = kfs[i].statePre.at<float>(3);
            }

            // 3. Assemble the Hybrid Measurement Vector [MP_x, MP_y, OF_vx, OF_vy]
            measurement.at<float>(0) = landmarks[i].x;
            measurement.at<float>(1) = landmarks[i].y;
            measurement.at<float>(2) = vx_of;
            measurement.at<float>(3) = vy_of;

            // 4. Correct the state
            cv::Mat estimated = kfs[i].correct(measurement);
            
            // 5. Update MediaPipe outputs with fused data
            landmarks[i].x = estimated.at<float>(0);
            landmarks[i].y = estimated.at<float>(1);

            // 6. Anchor the LK Flow's starting point to the newly smoothed KF state
            // This prevents the Optical Flow tracking window from drifting away over time
            prev_pts[i] = cv::Point2f(landmarks[i].x * width, landmarks[i].y * height);
        }
        frames_unseen = 0;
    }
};

// ---------------------------------------------------------------------------
// PIMPL: hides MediaPipe types from the header
// ---------------------------------------------------------------------------
struct MediaPipeTracker::Impl {
    mediapipe::CalculatorGraph graph_;
    bool initialised_ = false;

    std::mutex mutex_;
    std::shared_ptr<std::vector<HandData>> cached_hands_ = std::make_shared<std::vector<HandData>>();
    int64_t latest_timestamp_us_ = 0;

    HandSmoother left_smoother_;
    HandSmoother right_smoother_;
    
    // NEW: Grayscale image cache for Optical Flow
    cv::Mat prev_gray_;

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

MediaPipeTracker::MediaPipeTracker()
    : impl_(std::make_unique<Impl>()) {}

MediaPipeTracker::~MediaPipeTracker() = default;

bool MediaPipeTracker::init() {
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

bool MediaPipeTracker::isInitialised() const {
    return impl_->initialised_;
}

int64_t MediaPipeTracker::latestTimestamp() const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    return impl_->latest_timestamp_us_;
}

std::shared_ptr<const std::vector<HandData>> MediaPipeTracker::detect(const cv::Mat& frame, int64_t timestamp_us) {
    if (!impl_->initialised_) {
        return {};
    }

    // 1. Prepare Current Grayscale frame for Optical Flow
    cv::Mat curr_gray;
    cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);

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

    std::lock_guard<std::mutex> lock(impl_->mutex_);

    // 2. Initialise prev_gray_ on the very first frame
    if (impl_->prev_gray_.empty()) {
        impl_->prev_gray_ = curr_gray.clone();
    }

    bool left_seen = false;
    bool right_seen = false;

    // 3. Apply the Fusion Filter
    for (auto& hand : *impl_->cached_hands_) {
        if (hand.handedness == 1) { // Left Hand
            impl_->left_smoother_.smooth(hand.landmarks, impl_->prev_gray_, curr_gray);
            left_seen = true;
        } else if (hand.handedness == 2) { // Right Hand
            impl_->right_smoother_.smooth(hand.landmarks, impl_->prev_gray_, curr_gray);
            right_seen = true;
        }
    }

    if (!left_seen) {
        if (++impl_->left_smoother_.frames_unseen > 5) impl_->left_smoother_.reset();
    }
    if (!right_seen) {
        if (++impl_->right_smoother_.frames_unseen > 5) impl_->right_smoother_.reset();
    }

    // 4. Cache current grayscale frame for the next iteration
    impl_->prev_gray_ = curr_gray.clone();

    return impl_->cached_hands_;
}

} // namespace cv_keyboard