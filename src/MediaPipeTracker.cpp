#include "MediaPipeTracker.h"

namespace cv_keyboard {

static constexpr char kInputStream[] = "input_video";
static constexpr char kMultiHandLandmarksStream[] = "multi_hand_landmarks";
static constexpr char kMultiHandednessStream[]   = "multi_handedness";

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
    if (!impl_->initialised_) return {};
    
    int64_t t_start = cv::getTickCount(); // START ML TRACKER TIMER

    // Prepare Current Grayscale frame for Optical Flow
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

    int64_t t_mid = cv::getTickCount(); // END ML TRACKER / START FUSION TIMER
    tracker_time_ms_ = (t_mid - t_start) * 1000.0 / cv::getTickFrequency();

    std::lock_guard<std::mutex> lock(impl_->mutex_);

    // Initialise prev_gray_ on the very first frame
    if (impl_->prev_gray_.empty()) {
        impl_->prev_gray_ = curr_gray.clone();
    }

    bool left_seen = false;
    bool right_seen = false;

    // Apply the Fusion Filter
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

    // Cache current grayscale frame for the next iteration
    impl_->prev_gray_ = curr_gray.clone();

    int64_t t_end = cv::getTickCount(); // END FUSION TIMER
    fusion_time_ms_ = (t_end - t_mid) * 1000.0 / cv::getTickFrequency();

    return impl_->cached_hands_;
}

} // namespace cv_keyboard