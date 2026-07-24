#include "Mediator.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>

namespace cv_keyboard {

// Colours (BGR for OpenCV)
static const cv::Scalar kColorFingerTip(0, 255, 0);       // Green
static const cv::Scalar kColorLandmark(255, 255, 0);       // Cyan
static const cv::Scalar kColorConnection(200, 200, 200);   // Light grey
static const cv::Scalar kColorDebugText(0, 255, 255);      // Yellow
static const cv::Scalar kColorHandLabel(0, 165, 255);      // Orange

static constexpr int kFingerTipRadius = 8;
static constexpr int kLandmarkRadius = 4;
static constexpr int kConnectionThickness = 2;
static constexpr double kFontScale = 0.45;
static constexpr int kFontThickness = 1;

Mediator::Mediator()
    : hand_tracker_(std::make_unique<HandTracker>()) {}

Mediator::~Mediator() = default;

bool Mediator::init() {
    if (!hand_tracker_->init()) {
        std::cerr << "[Mediator] HandTracker initialisation failed.\n";
        return false;
    }
    std::cout << "[Mediator] Pipeline initialised.\n";
    return true;
}

bool Mediator::isInitialised() const {
    return hand_tracker_->isInitialised();
}

void Mediator::processFrame(const cv::Mat& frame) {
    auto hands = hand_tracker_->detect(frame);
    {
        std::lock_guard<std::mutex> lock(hands_mutex_);
        latest_hands_ = std::move(hands);
    }
}

std::vector<HandData> Mediator::latestHands() const {
    std::lock_guard<std::mutex> lock(hands_mutex_);
    return latest_hands_;
}

// ---------------------------------------------------------------------------
// Overlay rendering
// ---------------------------------------------------------------------------
void Mediator::renderOverlay(cv::Mat& frame) {
    auto hands = latestHands();
    if (hands.empty()) {
        return;
    }

    int frame_w = frame.cols;
    int frame_h = frame.rows;

    for (size_t h = 0; h < hands.size(); ++h) {
        const auto& hand = hands[h];

        // Draw hand label (left/right/unknown)
        std::string hand_label = "Hand " + std::to_string(h + 1);
        cv::putText(frame, hand_label, cv::Point(10, 30 + static_cast<int>(h) * 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, kColorHandLabel, 2);

        if (show_full_skeleton_) {
            // --- Full skeleton mode: draw all 21 landmarks + connections ---
            for (int i = 0; i < 21; ++i) {
                if (hand.landmarks[i].confidence < 0.5f) continue;

                int px = static_cast<int>(hand.landmarks[i].x * frame_w);
                int py = static_cast<int>(hand.landmarks[i].y * frame_h);

                cv::circle(frame, cv::Point(px, py), kLandmarkRadius,
                           kColorLandmark, -1);

                // Draw landmark index label
                cv::putText(frame, std::to_string(i),
                            cv::Point(px + 5, py - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.35, kColorLandmark, 1);
            }

            // Draw connection lines
            for (int c = 0; c < NUM_HAND_CONNECTIONS; ++c) {
                int i0 = HAND_CONNECTIONS[c][0];
                int i1 = HAND_CONNECTIONS[c][1];

                if (hand.landmarks[i0].confidence < 0.5f ||
                    hand.landmarks[i1].confidence < 0.5f) {
                    continue;
                }

                cv::Point p0(
                    static_cast<int>(hand.landmarks[i0].x * frame_w),
                    static_cast<int>(hand.landmarks[i0].y * frame_h));
                cv::Point p1(
                    static_cast<int>(hand.landmarks[i1].x * frame_w),
                    static_cast<int>(hand.landmarks[i1].y * frame_h));

                cv::line(frame, p0, p1, kColorConnection, kConnectionThickness);
            }
        } else {
            // --- Finger tips only mode ---
            for (int idx : FINGER_TIP_INDICES) {
                if (hand.landmarks[idx].confidence < 0.5f) continue;

                int px = static_cast<int>(hand.landmarks[idx].x * frame_w);
                int py = static_cast<int>(hand.landmarks[idx].y * frame_h);

                cv::circle(frame, cv::Point(px, py), kFingerTipRadius,
                           kColorFingerTip, -1);
            }
        }

        // --- Debug overlay: show (x, y) pixel coordinates for finger tips ---
        if (show_debug_overlay_) {
            int text_x = 10;
            int text_y = 80 + static_cast<int>(h) * 140;

            cv::putText(frame, hand_label + " finger tips:",
                        cv::Point(text_x, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, kFontScale, kColorDebugText,
                        kFontThickness);

            const char* tip_names[] = {"Thumb", "Index", "Mid", "Ring", "Pinky"};
            for (int i = 0; i < 5; ++i) {
                int idx = FINGER_TIP_INDICES[i];
                int px = static_cast<int>(hand.landmarks[idx].x * frame_w);
                int py = static_cast<int>(hand.landmarks[idx].y * frame_h);

                std::ostringstream oss;
                oss << tip_names[i] << " (" << idx << "): ("
                    << px << ", " << py << ") z="
                    << std::fixed << std::setprecision(3)
                    << hand.landmarks[idx].z;

                cv::putText(frame, oss.str(),
                            cv::Point(text_x, text_y + 20 + i * 20),
                            cv::FONT_HERSHEY_SIMPLEX, kFontScale,
                            kColorDebugText, kFontThickness);
            }
        }
    }
}

} // namespace cv_keyboard