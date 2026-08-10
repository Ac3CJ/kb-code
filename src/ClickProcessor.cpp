#include "ClickProcessor.h"

namespace cv_keyboard {

ClickProcessor::ClickProcessor() = default;
ClickProcessor::~ClickProcessor() = default;

void ClickProcessor::process(const std::vector<HandData>& hands, 
                            const KeyboardMap& keyboard_map, 
                            int frame_width, 
                            int frame_height) {
    hovered_key_ids_.clear();

    if (!keyboard_map.hasValidTransform() || hands.empty()) {
        return;
    }

    // Process each detected hand
    for (const auto& hand : hands) {
        // Check each fingertip (Thumb, Index, Middle, Ring, Pinky)
        for (int tip_idx : FINGER_TIP_INDICES) {
            const auto& tip = hand.landmarks[tip_idx];

            // Convert normalized coordinates [0.0, 1.0] to camera pixel space
            float px = tip.x * static_cast<float>(frame_width);
            float py = tip.y * static_cast<float>(frame_height);

            // Project camera pixel coordinate down to physical plane (cm)
            cv::Point2f pt_cm = keyboard_map.pixelToPhysical(px, py);

            // Lookup key ID at physical location
            std::string key_id = keyboard_map.getKeyAt(pt_cm.x, pt_cm.y);

            if (!key_id.empty()) {
                hovered_key_ids_.insert(key_id);
            }
        }
    }
}

bool ClickProcessor::isHovered(const std::string& key_id) const {
    return hovered_key_ids_.find(key_id) != hovered_key_ids_.end();
}

} // namespace cv_keyboard