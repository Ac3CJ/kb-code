#include "ZeroCrossingProcessor.h"
#include <iostream>

namespace cv_keyboard {

void ZeroCrossingProcessor::detectClicks(const std::vector<HandData>& hands, 
                                         const KeyboardMap& keyboard_map, 
                                         int frame_width, 
                                         int frame_height) {
                            
    // TUNABLE PARAMETERS (Normalized pixel velocity per frame)
    // Adjust these to filter out slow hovering movements
    const float STRIKE_THRESH = 0.000015f; 
    const float RELEASE_THRESH = 0.000015f;

    for (const auto& hand : hands) {
        for (int tip_idx : FINGER_TIP_INDICES) {
            // Unique ID: 1000 for Left Thumb, 2000 for Right Thumb, etc.
            int finger_id = (hand.handedness << 8) | tip_idx;
            const auto& tip = hand.landmarks[tip_idx];

            // Track the rolling velocity history for this specific finger
            auto& history = vy_history_[finger_id];
            history.push_back(tip.vy);
            if (history.size() > 3) {
                history.pop_front();
            }

            if (history.size() == 3) {
                float v_prev = history[1];
                float v_curr = history[2];

                // std::cout << "[ZeroCrossingProcessor] Finger ID: " << finger_id 
                //           << " | v_prev: " << v_prev 
                //           << " | v_curr: " << v_curr << std::endl;

                // Zero-Crossing Logic: Was it moving down fast, and is now moving up fast?
                if (v_prev > STRIKE_THRESH && v_curr < -RELEASE_THRESH) {
                    float px = tip.x * static_cast<float>(frame_width);
                    float py = tip.y * static_cast<float>(frame_height);
                    
                    cv::Point2f pt_cm = keyboard_map.pixelToPhysical(px, py);
                    std::string key_id = keyboard_map.getKeyAt(pt_cm.x, pt_cm.y);
                    
                    if (!key_id.empty()) {
                        clicked_key_ids_.insert(key_id);
                    }
                }
            }
        }
    }
}

} // namespace cv_keyboard