#include "Mediator.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>

namespace cv_keyboard {

// Colours (BGR for OpenCV)
// Colors mapped to: {THUMB, INDEX, MIDDLE, RING, PINKY} in BGR format
static const cv::Scalar kFingerTipColors[] = {
    cv::Scalar(0, 0, 255),    // Thumb: Red
    cv::Scalar(0, 165, 255),  // Index: Orange
    cv::Scalar(0, 255, 255),  // Middle: Yellow
    cv::Scalar(0, 255, 0),    // Ring: Green
    cv::Scalar(255, 0, 0)     // Pinky: Blue
};
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
    int64_t ts_us = static_cast<int64_t>(cv::getTickCount() * 1e6 / cv::getTickFrequency());
    
    // Push the copy into thread-safe ring buffer
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        frame_buffer_.emplace_back(ts_us, frame.clone());
        while (frame_buffer_.size() > kMaxBufferSize) {
            frame_buffer_.pop_front(); // Prevent memory overflow if UI stalls
        }
    }

    auto hands = hand_tracker_->detect(frame, ts_us);
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
    int64_t target_ts = hand_tracker_->latestTimestamp();

    if (target_ts > 0) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        auto it = frame_buffer_.begin();
        while (it != frame_buffer_.end()) {
            if (it->first == target_ts) {
                // Exact timestamp match found. Overwrite live camera frame with synchronized frame.
                it->second.copyTo(frame);
                // Clean up: Erase this frame and all older frames
                frame_buffer_.erase(frame_buffer_.begin(), std::next(it));
                break;
            } else if (it->first < target_ts) {
                // Older than current inference; discard to prevent memory leaks
                it = frame_buffer_.erase(it);
            } else {
                // We reached frames newer than target_ts without finding an exact match
                break;
            }
        }
    }

    if (show_grid_) {
        drawGrid(frame, 100, 0.35); // 100px step, 35% opacity
    }

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
                // if (hand.landmarks[i].confidence < 0.5f) continue;

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

                // if (hand.landmarks[i0].confidence < 0.5f ||
                //     hand.landmarks[i1].confidence < 0.5f) {
                //     continue;
                // }

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
            for (int i = 0; i < 5; ++i) {
                int idx = FINGER_TIP_INDICES[i];

                int px = static_cast<int>(hand.landmarks[idx].x * frame_w);
                int py = static_cast<int>(hand.landmarks[idx].y * frame_h);

                // Draw using the finger-specific color from our array
                cv::circle(frame, cv::Point(px, py), kFingerTipRadius,
                           kFingerTipColors[i], -1);
            }
        }

        for (int i = 0; i < 5; ++i) {
            int idx = FINGER_TIP_INDICES[i];

            int px = static_cast<int>(hand.landmarks[idx].x * frame_w);
            int py = static_cast<int>(hand.landmarks[idx].y * frame_h);

            // Format coordinate text: e.g., "(412, 230)"
            std::string coord_text = "(" + std::to_string(px) + "," + std::to_string(py) + ")";
            
            // Draw slightly offset above-right of the fingertip to avoid clipping the circle
            cv::putText(frame, coord_text, cv::Point(px + 10, py - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, kFingerTipColors[i], 1);
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

void Mediator::drawGrid(cv::Mat& frame, int step, double alpha) const {
    int width = frame.cols;
    int height = frame.rows;

    // Create a temporary copy to draw the semitransparent elements
    cv::Mat grid_overlay = frame.clone();
    cv::Scalar grid_color(255, 255, 255);  // White grid lines
    cv::Scalar label_color(200, 200, 200); // Light grey labels

    // Draw vertical lines and X-axis coordinate labels
    for (int x = step; x < width; x += step) {
        cv::line(grid_overlay, cv::Point(x, 0), cv::Point(x, height), grid_color, 1);
        cv::putText(grid_overlay, std::to_string(x), cv::Point(x + 4, 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1);
    }

    // Draw horizontal lines and Y-axis coordinate labels
    for (int y = step; y < height; y += step) {
        cv::line(grid_overlay, cv::Point(0, y), cv::Point(width, y), grid_color, 1);
        cv::putText(grid_overlay, std::to_string(y), cv::Point(4, y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1);
    }

    // Draw coordinate labels at each grid intersection for immediate GT reading
    for (int x = step; x < width; x += step) {
        for (int y = step; y < height; y += step) {
            std::string coord_label = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
            cv::putText(grid_overlay, coord_label, cv::Point(x + 4, y + 14),
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(180, 180, 180), 1);
        }
    }

    // Blend the grid overlay back into the original frame with transparency
    cv::addWeighted(grid_overlay, alpha, frame, 1.0 - alpha, 0, frame);
}

} // namespace cv_keyboard