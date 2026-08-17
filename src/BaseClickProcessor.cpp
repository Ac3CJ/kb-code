#include "BaseClickProcessor.h"

namespace cv_keyboard {

BaseClickProcessor::BaseClickProcessor() = default;

void BaseClickProcessor::process(const std::vector<HandData>& hands, 
                                 const KeyboardMap& keyboard_map, 
                                 int frame_width, 
                                 int frame_height) {
    hovered_key_ids_.clear();
    clicked_key_ids_.clear();
    finger_hovers_.clear();

    if (!keyboard_map.hasValidTransform() || hands.empty()) {
        active_clicks_.clear();
        return;
    }

    for (const auto& hand : hands) {
        for (int tip_idx : FINGER_TIP_INDICES) {
            int finger_id = (hand.handedness << 8) | tip_idx;
            const auto& tip = hand.landmarks[tip_idx];

            float px = tip.x * static_cast<float>(frame_width);
            float py = tip.y * static_cast<float>(frame_height);

            cv::Point2f pt_cm = keyboard_map.pixelToPhysical(px, py);
            std::string key_id = keyboard_map.getKeyAt(pt_cm.x, pt_cm.y);

            if (!key_id.empty()) {
                hovered_key_ids_.insert(key_id);
                finger_hovers_[finger_id] = key_id;
            }
        }
    }

    // 3. Delegate specific Touch Logic to the Child Class
    detectClicks(hands, keyboard_map, frame_width, frame_height);

    // Debounce
    for (auto it = active_clicks_.begin(); it != active_clicks_.end(); ) {
        int finger_id = it->first;
        std::string pressed_key = it->second;

        // Check if the finger is still hovering over the key it pressed
        auto hover_it = finger_hovers_.find(finger_id);
        
        if (hover_it == finger_hovers_.end() || hover_it->second != pressed_key) {
            // The finger slid off. Auto-release the click.
            it = active_clicks_.erase(it);
        } else {
            // The finger is still dwelling inside the key. 
            clicked_key_ids_.insert(pressed_key);
            ++it;
        }
    }
}

void BaseClickProcessor::reset() {
    hovered_key_ids_.clear();
    clicked_key_ids_.clear();
    finger_hovers_.clear();
    active_clicks_.clear();
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