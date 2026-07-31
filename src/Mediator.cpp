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
    : hand_tracker_(std::make_unique<HandTracker>()),
      frame_pool_(kMaxBufferSize) {}

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
    
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        frame.copyTo(frame_pool_[write_index_].frame);
        frame_pool_[write_index_].timestamp = ts_us;
        write_index_ = (write_index_ + 1) % kMaxBufferSize;
    }

    // O(1) pointer assignment instead of copying vector data
    auto hands = hand_tracker_->detect(frame, ts_us);
    {
        std::lock_guard<std::mutex> lock(hands_mutex_);
        latest_hands_ = std::move(hands);
    }
}

std::shared_ptr<const std::vector<HandData>> Mediator::latestHands() const {
    std::lock_guard<std::mutex> lock(hands_mutex_);
    return latest_hands_;
}

// ---------------------------------------------------------------------------
// Overlay rendering
// ---------------------------------------------------------------------------

void Mediator::drawGrid(cv::Mat& frame, int step) const {
    // 1. Only draw the grid and generate the mask from scratch if resolution changed
    if (cached_grid_overlay_.empty() || last_frame_size_ != frame.size()) {
        last_frame_size_ = frame.size();
        cached_grid_overlay_ = cv::Mat::zeros(frame.size(), frame.type());
        
        int width = frame.cols;
        int height = frame.rows;
        
        // Slightly softer colors since we no longer have alpha blending
        cv::Scalar grid_color(200, 200, 200); 
        cv::Scalar label_color(180, 180, 180);

        // Draw lines and labels
        for (int x = step; x < width; x += step) {
            cv::line(cached_grid_overlay_, cv::Point(x, 0), cv::Point(x, height), grid_color, 1);
            cv::putText(cached_grid_overlay_, std::to_string(x), cv::Point(x + 4, 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1);
        }
        for (int y = step; y < height; y += step) {
            cv::line(cached_grid_overlay_, cv::Point(0, y), cv::Point(width, y), grid_color, 1);
            cv::putText(cached_grid_overlay_, std::to_string(y), cv::Point(4, y - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1);
        }
        
        // Draw intersection coordinates ONCE
        for (int x = step; x < width; x += step) {
            for (int y = step; y < height; y += step) {
                std::string coord_label = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
                cv::putText(cached_grid_overlay_, coord_label, cv::Point(x + 4, y + 14),
                            cv::FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1);
            }
        }

        // --- NEW: Generate the mask ---
        // Any pixel in the overlay that isn't pure black (0,0,0) becomes 255 (white) in the mask.
        cv::cvtColor(cached_grid_overlay_, cached_grid_mask_, cv::COLOR_BGR2GRAY);
        cv::threshold(cached_grid_mask_, cached_grid_mask_, 1, 255, cv::THRESH_BINARY);
    }

    // 2. Every frame, do a lightning-fast masked copy.
    // OpenCV iterates over the mask. If mask[x,y] == 255, it copies overlay[x,y] to frame[x,y].
    // It entirely skips pixels where mask[x,y] == 0.
    cached_grid_overlay_.copyTo(frame, cached_grid_mask_);
}

void Mediator::renderOverlay(const cv::Mat& raw_frame, cv::Mat& display_frame) {
    int64_t target_ts = hand_tracker_->latestTimestamp();
    bool frame_retrieved = false;

    if (target_ts > 0) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        for (auto& slot : frame_pool_) {
            if (slot.timestamp == target_ts) {
                // O(1) pointer swap instead of deep copy!
                // display_frame takes ownership of the matched historical frame.
                // slot.frame receives display_frame's old buffer, recycling the allocated
                // memory back into the ring buffer for processFrame() to overwrite later!
                std::swap(slot.frame, display_frame);
                slot.timestamp = 0; // Mark slot as consumed
                frame_retrieved = true;
                break;
            } else if (slot.timestamp > 0 && slot.timestamp < target_ts) {
                // Older than current inference; mark as consumed so we don't hold stale timestamps
                slot.timestamp = 0;
            }
        }
    }

    if (!frame_retrieved) {
        // Fallback if no timestamp match exists: copy raw_frame into display_frame.
        // Because display_frame persists across loops, copyTo() reuses its memory automatically.
        raw_frame.copyTo(display_frame);
    }

    if (show_grid_) {
        drawGrid(display_frame, 100);
    }

    auto hands = latestHands();
    if (!hands || hands->empty()) {
        return;
    }

    int frame_w = display_frame.cols;
    int frame_h = display_frame.rows;

    for (size_t h = 0; h < hands->size(); ++h) {
        const auto& hand = (*hands)[h];

        // Draw hand label (left/right/unknown)
        std::string hand_label = "Hand " + std::to_string(h + 1);
        cv::putText(display_frame, hand_label, cv::Point(10, 30 + static_cast<int>(h) * 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, kColorHandLabel, 2);

        if (show_full_skeleton_) {
            // --- Full skeleton mode: draw all 21 landmarks + connections ---
            for (int i = 0; i < 21; ++i) {
                int px = static_cast<int>(hand.landmarks[i].x * frame_w);
                int py = static_cast<int>(hand.landmarks[i].y * frame_h);

                cv::circle(display_frame, cv::Point(px, py), kLandmarkRadius,
                           kColorLandmark, -1);

                cv::putText(display_frame, std::to_string(i),
                            cv::Point(px + 5, py - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.35, kColorLandmark, 1);
            }

            for (int c = 0; c < NUM_HAND_CONNECTIONS; ++c) {
                int i0 = HAND_CONNECTIONS[c][0];
                int i1 = HAND_CONNECTIONS[c][1];

                cv::Point p0(
                    static_cast<int>(hand.landmarks[i0].x * frame_w),
                    static_cast<int>(hand.landmarks[i0].y * frame_h));
                cv::Point p1(
                    static_cast<int>(hand.landmarks[i1].x * frame_w),
                    static_cast<int>(hand.landmarks[i1].y * frame_h));

                cv::line(display_frame, p0, p1, kColorConnection, kConnectionThickness);
            }
        } else {
            // --- Finger tips only mode ---
            for (int i = 0; i < 5; ++i) {
                int idx = FINGER_TIP_INDICES[i];

                int px = static_cast<int>(hand.landmarks[idx].x * frame_w);
                int py = static_cast<int>(hand.landmarks[idx].y * frame_h);

                cv::circle(display_frame, cv::Point(px, py), kFingerTipRadius,
                           kFingerTipColors[i], -1);
            }
        }

        for (int i = 0; i < 5; ++i) {
            int idx = FINGER_TIP_INDICES[i];

            int px = static_cast<int>(hand.landmarks[idx].x * frame_w);
            int py = static_cast<int>(hand.landmarks[idx].y * frame_h);

            std::string coord_text = "(" + std::to_string(px) + "," + std::to_string(py) + ")";
            
            cv::putText(display_frame, coord_text, cv::Point(px + 10, py - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, kFingerTipColors[i], 1);
        }

        // --- Debug overlay ---
        if (show_debug_overlay_) {
            int text_x = 10;
            int text_y = 80 + static_cast<int>(h) * 140;

            cv::putText(display_frame, hand_label + " finger tips:",
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

                cv::putText(display_frame, oss.str(),
                            cv::Point(text_x, text_y + 20 + i * 20),
                            cv::FONT_HERSHEY_SIMPLEX, kFontScale,
                            kColorDebugText, kFontThickness);
            }
        }
    }
}

} // namespace cv_keyboard