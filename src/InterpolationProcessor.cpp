#include "InterpolationProcessor.h"

namespace cv_keyboard {

void InterpolationProcessor::detectClicks(const std::vector<HandData>& hands, 
                                          const KeyboardMap& keyboard_map, 
                                          int frame_width, 
                                          int frame_height) {
    
    const float NOISE_FLOOR = 0.000001f;

    for (const auto& hand : hands) {
        for (int tip_idx : FINGER_TIP_INDICES) {
            int finger_id = (hand.handedness << 8) | tip_idx;
            const auto& tip = hand.landmarks[tip_idx];

            auto& history = state_history_[finger_id];
            history.push_back({tip.x, tip.y, tip.vy});
            if (history.size() > 3) {
                history.pop_front();
            }

            if (history.size() == 3) {
                const auto& prev = history[1];
                const auto& curr = history[2];

                if (prev.vy > NOISE_FLOOR && curr.vy <= NOISE_FLOOR) {
                    float t = prev.vy / (prev.vy - curr.vy); 
                    t = std::max(0.0f, std::min(1.0f, t));

                    float interp_x = prev.x + t * (curr.x - prev.x);
                    float interp_y = prev.y + t * (curr.y - prev.y);

                    float px = interp_x * static_cast<float>(frame_width);
                    float py = interp_y * static_cast<float>(frame_height);
                    
                    cv::Point2f pt_cm = keyboard_map.pixelToPhysical(px, py);
                    std::string key_id = keyboard_map.getKeyAt(pt_cm.x, pt_cm.y);
                    
                    if (!key_id.empty()) {
                        active_clicks_[finger_id] = key_id;
                    }
                }

                // Detect Lift Phase
                if (curr.vy < -NOISE_FLOOR) {
                    active_clicks_.erase(finger_id);
                }
            }
        }
    }
}
}