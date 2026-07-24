#include "HandTracker.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/status.h"

#include <opencv2/imgproc.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace cv_keyboard {

static constexpr char kInputStream[] = "input_video";
static constexpr char kMultiHandLandmarksStream[] = "multi_hand_landmarks";
static constexpr char kMultiHandednessStream[]   = "multi_handedness";

// ---------------------------------------------------------------------------
// PIMPL: hides MediaPipe types from the header
// ---------------------------------------------------------------------------
struct HandTracker::Impl {
    mediapipe::CalculatorGraph graph_;
    bool initialised_ = false;

    std::mutex mutex_;
    std::vector<HandData> cached_hands_;

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
            if (packet.IsEmpty()) return absl::OkStatus();

            const auto& multi_hand_landmarks =
                packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

            std::lock_guard<std::mutex> lock(impl_->mutex_);
            
            if (multi_hand_landmarks.empty()) {
                impl_->cached_hands_.clear();
                return absl::OkStatus();
            }

            impl_->cached_hands_.resize(multi_hand_landmarks.size());
            for (size_t h = 0; h < multi_hand_landmarks.size(); ++h) {
                const auto& landmark_list = multi_hand_landmarks[h];
                int count = std::min(landmark_list.landmark_size(), 21);
                for (int i = 0; i < count; ++i) {
                    const auto& src = landmark_list.landmark(i);
                    impl_->cached_hands_[h].landmarks[i].x = static_cast<float>(src.x());
                    impl_->cached_hands_[h].landmarks[i].y = static_cast<float>(src.y());
                    impl_->cached_hands_[h].landmarks[i].z = static_cast<float>(src.z());
                    impl_->cached_hands_[h].landmarks[i].confidence = static_cast<float>(src.visibility());
                }
                impl_->cached_hands_[h].hand_confidence = 1.0f;
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
                if (h < impl_->cached_hands_.size() &&
                    multi_handedness[h].classification_size() > 0) {
                    const auto& cls = multi_handedness[h].classification(0);
                    if (cls.label() == "Left") {
                        impl_->cached_hands_[h].handedness = 1;
                    } else if (cls.label() == "Right") {
                        impl_->cached_hands_[h].handedness = 2;
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

    // Start the graph.
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

std::vector<HandData> HandTracker::detect(const cv::Mat& frame) {
    if (!impl_->initialised_) {
        return {};
    }

    // Convert BGR → RGB
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

    // Wrap into a MediaPipe ImageFrame
    auto input_frame = std::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, rgb_frame.cols, rgb_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    rgb_frame.copyTo(input_frame_mat);

    // Send frame into the graph asynchronously
    int64_t timestamp_us =
        static_cast<int64_t>(cv::getTickCount() * 1e6 / cv::getTickFrequency());
    auto status = impl_->graph_.AddPacketToInputStream(
        kInputStream,
        mediapipe::Adopt(input_frame.release())
            .At(mediapipe::Timestamp(timestamp_us)));

    if (!status.ok()) {
        std::cerr << "[HandTracker] AddPacketToInputStream error: "
                  << status.message() << "\n";
    }

    // Non-blocking return of the latest cached hand data
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    return impl_->cached_hands_;
}

} // namespace cv_keyboard