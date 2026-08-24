#include "Mediator.h"

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

static constexpr int kFingerTipRadius = 4;
static constexpr int kLandmarkRadius = 3;
static constexpr int kConnectionThickness = 2;
static constexpr double kFontScale = 0.45;
static constexpr int kFontThickness = 1;

Mediator::Mediator(const std::string& tracker_type, const std::string& processor_type) {
    // Instantiate Tracker
    if (tracker_type == "mediapipe") {
        hand_tracker_ = std::make_unique<MediaPipeTracker>();
    } else {
        // Default / RTMPose fallback
        hand_tracker_ = std::make_unique<RTMPoseTracker>();
    }

    // Instantiate Processor
    if (processor_type == "interpolation") {
        click_processor_ = std::make_unique<InterpolationProcessor>();
    } else {
        // Default / ZeroCrossing fallback
        click_processor_ = std::make_unique<ZeroCrossingProcessor>();
    }
}

Mediator::~Mediator() = default;

bool Mediator::init() {
    if (!hand_tracker_->init()) {
        std::cerr << "[Mediator] HandTracker initialisation failed.\n";
        return false;
    }
    std::cout << "[Mediator] Pipeline initialised.\n";
    virtual_keyboard_.loadUKLayout();
    return true;
}

bool Mediator::isInitialised() const {
    return hand_tracker_->isInitialised();
}

void Mediator::processFrame(const cv::Mat& frame) {
    int64_t t0 = cv::getTickCount();
    virtual_keyboard_.updateTransform(frame);

    int64_t t1 = cv::getTickCount();
    metrics_.homography_ms = (t1 - t0) * 1000.0 / cv::getTickFrequency();

    int64_t ts_us = static_cast<int64_t>(cv::getTickCount() * 1e6 / cv::getTickFrequency());
    auto hands = hand_tracker_->detect(frame, ts_us);

    metrics_.mp_tracker_ms = hand_tracker_->getTrackerTimeMs();
    metrics_.sensor_fusion_ms = hand_tracker_->getFusionTimeMs();
    
    {
        std::lock_guard<std::mutex> lock(hands_mutex_);
        latest_hands_ = std::move(hands);
    }

    int64_t t2 = cv::getTickCount();
    if (latest_hands_) {
        click_processor_->process(*latest_hands_, virtual_keyboard_, frame.cols, frame.rows);
    }

    int64_t t3 = cv::getTickCount();
    metrics_.click_process_ms = (t3 - t2) * 1000.0 / cv::getTickFrequency();
    metrics_.total_ms = (t3 - t0) * 1000.0 / cv::getTickFrequency();
}

std::shared_ptr<const std::vector<HandData>> Mediator::latestHands() const {
    std::lock_guard<std::mutex> lock(hands_mutex_);
    return latest_hands_;
}

// Test function to inject cached hand data and re-run the ClickProcessor logic
void Mediator::injectCachedHands(std::shared_ptr<const std::vector<HandData>> cached_hands, const cv::Mat& frame) {
    // 1. We MUST update the ArUco board transform for this frame (ArUco is not stateful, so this is safe)
    virtual_keyboard_.updateTransform(frame);

    // 2. Inject the cached MediaPipe data bypassing the ML model
    {
        std::lock_guard<std::mutex> lock(hands_mutex_);
        latest_hands_ = cached_hands;
    }

    // 3. Re-run the ClickProcessor logic so we can experiment with it!
    if (latest_hands_) {
        click_processor_->process(*latest_hands_, virtual_keyboard_, frame.cols, frame.rows);
    }
}

void Mediator::warmUpClickProcessor(std::shared_ptr<const std::vector<HandData>> past_hands, int frame_width, int frame_height) {
    if (past_hands && click_processor_) {
        // We evaluate the historical hands against the CURRENT ArUco board transform.
        // Since T-2 was only 60ms ago, it is mathematically safe to assume the desk hasn't moved.
        click_processor_->process(*past_hands, virtual_keyboard_, frame_width, frame_height);
    }
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

void Mediator::drawVirtualKeyboard(cv::Mat& frame) {
    if (cached_kb_overlay_.empty() || last_frame_size_ != frame.size()) {
        last_frame_size_ = frame.size();
        cached_kb_overlay_ = cv::Mat::zeros(frame.size(), frame.type());
        
        // We want the keyboard to take up exactly half the screen width
        float target_px_width = frame.cols / 2.0f;
        float target_px_height = frame.rows / 2.0f; // Optional: limit height to half the screen
        
        // Our UK layout is exactly 15u wide
        float kb_width_u = 15.0f; 
        
        // Calculate pixels per 'u'
        float px_per_u = target_px_width / kb_width_u;
        
        // Offset to push it to the top right corner
        float offset_x = frame.cols - target_px_width - 20.0f; // Slight 20px padding from the right edge
        float offset_y = frame.rows - target_px_height - 20.0f; // Slight 20px padding from the top edge

        cv::Scalar border_color(0, 255, 0); // Green
        cv::Scalar text_color(255, 255, 255); // White

        for (const auto& key : virtual_keyboard_.getKeys()) {
            // Convert 'u' coordinates to pixel coordinates
            int px = static_cast<int>(offset_x + (key.x_u * px_per_u));
            int py = static_cast<int>(offset_y + (key.y_u * px_per_u));
            int pw = static_cast<int>(key.width_u * px_per_u);
            int ph = static_cast<int>(key.height_u * px_per_u);

            // Draw the key border
            cv::rectangle(cached_kb_overlay_, cv::Rect(px, py, pw, ph), border_color, 2);

            // Center the text inside the key
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(key.id, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
            int text_x = px + (pw - text_size.width) / 2;
            int text_y = py + (ph + text_size.height) / 2;

            cv::putText(cached_kb_overlay_, key.id, cv::Point(text_x, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);
        }

        // Generate the lightning-fast copy mask
        cv::cvtColor(cached_kb_overlay_, cached_kb_mask_, cv::COLOR_BGR2GRAY);
        cv::threshold(cached_kb_mask_, cached_kb_mask_, 1, 255, cv::THRESH_BINARY);
    }

    // Drop it onto the frame
    cached_kb_overlay_.copyTo(frame, cached_kb_mask_);
}

void Mediator::drawPhysicalKeyboard(cv::Mat& frame) {
    if (!virtual_keyboard_.hasValidTransform()) {
        return; 
    }

    cv::Scalar border_color(0, 255, 0);         // Green border
    cv::Scalar hover_fill_color(0, 165, 255);   // Orange fill (BGR)
    cv::Scalar click_fill_color(255, 0, 0);     // Blue fill (BGR)
    cv::Scalar text_color(255, 255, 255);       // White text

    for (const auto& key : virtual_keyboard_.getKeys()) {
        float x = key.x_cm();
        float y = key.y_cm();
        float w = key.width_cm();
        float h = key.height_cm();

        // 1. Map physical corners to camera pixels
        cv::Point2f p1 = virtual_keyboard_.physicalToPixel(x, y);
        cv::Point2f p2 = virtual_keyboard_.physicalToPixel(x + w, y);
        cv::Point2f p3 = virtual_keyboard_.physicalToPixel(x + w, y + h);
        cv::Point2f p4 = virtual_keyboard_.physicalToPixel(x, y + h);

        std::vector<cv::Point> pts = {
            cv::Point(p1.x, p1.y),
            cv::Point(p2.x, p2.y),
            cv::Point(p3.x, p3.y),
            cv::Point(p4.x, p4.y)
        };
        std::vector<std::vector<cv::Point>> contours = { pts };

        // 2. Check if hovered and draw a semi-transparent filled polygon
        if (click_processor_->isClicked(key.id)) {
            cv::Mat overlay;
            frame.copyTo(overlay);
            
            // Fill the polygon on the duplicate frame
            cv::fillPoly(overlay, contours, click_fill_color);
            
            // Blend the overlay back into the original frame (50% opacity)
            cv::addWeighted(overlay, 0.5, frame, 0.5, 0, frame);
        } else if (click_processor_->isHovered(key.id)) {
            cv::Mat overlay;
            frame.copyTo(overlay);
            
            // Fill the polygon on the duplicate frame
            cv::fillPoly(overlay, contours, hover_fill_color);
            
            // Blend the overlay back into the original frame (50% opacity)
            cv::addWeighted(overlay, 0.5, frame, 0.5, 0, frame);
        }

        // 3. Draw the persistent green border
        cv::polylines(frame, contours, true, border_color, 2);

        // 4. Center text label inside the key polygon
        int center_x = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4;
        int center_y = (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4;

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(key.id, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
        
        cv::putText(frame, key.id, 
                    cv::Point(center_x - text_size.width / 2, center_y + text_size.height / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);
    }
}

void Mediator::drawHands(cv::Mat& frame) {
    if (!show_hands_) {
        return;
    }
    auto hands = latestHands();
    if (!hands || hands->empty()) {
        return;
    }

    int frame_w = frame.cols;
    int frame_h = frame.rows;

    // Drawing hands and landmarks
    for (size_t h = 0; h < hands->size(); ++h) {
        const auto& hand = (*hands)[h];

        // Draw hand label (left/right/unknown)
        std::string hand_label = "Hand " + std::to_string(h + 1);
        cv::putText(frame, hand_label, cv::Point(10, 30 + static_cast<int>(h) * 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, kColorHandLabel, 2);

        if (show_full_skeleton_) {
            // --- Full skeleton mode: draw all 21 landmarks + connections ---
            for (int i = 0; i < 21; ++i) {
                int px = static_cast<int>(hand.landmarks[i].x * frame_w);
                int py = static_cast<int>(hand.landmarks[i].y * frame_h);

                cv::circle(frame, cv::Point(px, py), kLandmarkRadius,
                           kColorLandmark, -1);

                cv::putText(frame, std::to_string(i),
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

                cv::line(frame, p0, p1, kColorConnection, kConnectionThickness);
            }
        } else {
            // --- Finger tips only mode ---
            for (int i = 0; i < 5; ++i) {
                int idx = FINGER_TIP_INDICES[i];

                int px = static_cast<int>(hand.landmarks[idx].x * frame_w);
                int py = static_cast<int>(hand.landmarks[idx].y * frame_h);

                cv::circle(frame, cv::Point(px, py), kFingerTipRadius,
                           kFingerTipColors[i], -1);
            }
        }

        // --- Debug overlay ---
        if (debug_mode_ == DebugMode::POSE) {
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
                int pvx = static_cast<int>(hand.landmarks[idx].vx * frame_w);
                int pvy = static_cast<int>(hand.landmarks[idx].vy * frame_h);
                

                float conf = hand.hand_confidence;

                std::ostringstream oss;
                oss << tip_names[i] << " (" << idx << "): ("
                    << px << ", " << py << ") z="
                    << std::fixed << std::setprecision(3)
                    << hand.landmarks[idx].z
                    << " conf=" << std::fixed << std::setprecision(2)
                    << conf;

                cv::putText(frame, oss.str(),
                            cv::Point(text_x, text_y + 20 + i * 20),
                            cv::FONT_HERSHEY_SIMPLEX, kFontScale,
                            kColorDebugText, kFontThickness);

                std::string coord_text = "(" + std::to_string(px) + "," + std::to_string(py) + "), (" 
                    + std::to_string(pvx) + "," + std::to_string(pvy) + ")";
            
                cv::putText(frame, coord_text, cv::Point(px + 10, py - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, kFingerTipColors[i], 1);
            }
        }
    }
}

void Mediator::drawPerfMetrics(cv::Mat& frame) const {
    int y = 30;
    auto drawText = [&](const std::string& text) {
        cv::putText(frame, text, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        y += 30;
    };

    drawText("--- ABLATION METRICS ---");
    drawText("Total Time : " + std::to_string(metrics_.total_ms).substr(0, 5) + " ms");
    drawText("MP Tracker : " + std::to_string(metrics_.mp_tracker_ms).substr(0, 5) + " ms");
    drawText("Fusion     : " + std::to_string(metrics_.sensor_fusion_ms).substr(0, 5) + " ms");
    drawText("Homography : " + std::to_string(metrics_.homography_ms).substr(0, 5) + " ms");
    drawText("Click Math : " + std::to_string(metrics_.click_process_ms).substr(0, 5) + " ms");
}

void Mediator::renderOverlay(const cv::Mat& raw_frame, cv::Mat& display_frame) {
    raw_frame.copyTo(display_frame);
    if (show_grid_) drawGrid(display_frame, 100);
    if (show_keyboard_) drawPhysicalKeyboard(display_frame);
    drawHands(display_frame);

    if (debug_mode_ == DebugMode::PERF) drawPerfMetrics(display_frame);

}

} // namespace cv_keyboard