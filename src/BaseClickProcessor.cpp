#include "BaseClickProcessor.h"

namespace cv_keyboard {

BaseClickProcessor::BaseClickProcessor() = default;

void BaseClickProcessor::process(const std::vector<HandData>& hands, 
                                 const KeyboardMap& keyboard_map, 
                                 int frame_width, 
                                 int frame_height) {
    // 1. Clear previous frame states[cite: 19]
    hovered_key_ids_.clear();
    clicked_key_ids_.clear();

    if (!keyboard_map.hasValidTransform() || hands.empty()) {
        return;
    }

    // 2. Perform shared Hover Logic[cite: 19]
    for (const auto& hand : hands) {
        for (int tip_idx : FINGER_TIP_INDICES) {
            const auto& tip = hand.landmarks[tip_idx];

            float px = tip.x * static_cast<float>(frame_width);
            float py = tip.y * static_cast<float>(frame_height);

            cv::Point2f pt_cm = keyboard_map.pixelToPhysical(px, py);
            std::string key_id = keyboard_map.getKeyAt(pt_cm.x, pt_cm.y);

            if (!key_id.empty()) {
                hovered_key_ids_.insert(key_id);
            }
        }
    }

    // 3. Delegate specific Touch Logic to the Child Class
    detectClicks(hands, keyboard_map, frame_width, frame_height);
}

void BaseClickProcessor::reset() {
    hovered_key_ids_.clear();
    clicked_key_ids_.clear();
    resetHistory();
}

bool BaseClickProcessor::isHovered(const std::string& key_id) const {
    return hovered_key_ids_.find(key_id) != hovered_key_ids_.end();
}

const std::unordered_set<std::string>& BaseClickProcessor::getHoveredKeys() const {
    return hovered_key_ids_;
}

bool BaseClickProcessor::isClicked(const std::string& key_id) const {
    return clicked_key_ids_.find(key_id) != clicked_key_ids_.end();
}

const std::unordered_set<std::string>& BaseClickProcessor::getClickedKeys() const {
    return clicked_key_ids_;
}

} // namespace cv_keyboard