#ifndef CV_KEYBOARD_CLICK_PROCESSOR_H
#define CV_KEYBOARD_CLICK_PROCESSOR_H

#include <string>
#include <vector>
#include <unordered_set>
#include "HandTracker.h"
#include "KeyboardMap.h"

namespace cv_keyboard {

class ClickProcessor {
public:
    ClickProcessor();
    ~ClickProcessor();

    /// Reads fingertip data and keyboard mapping to compute hover states.
    /// Does not mutate HandTracker or KeyboardMap.
    void process(const std::vector<HandData>& hands, 
                 const KeyboardMap& keyboard_map, 
                 int frame_width, 
                 int frame_height);

    /// Query whether a specific key ID is currently hovered
    bool isHovered(const std::string& key_id) const;

    /// Returns the set of all currently hovered key IDs
    const std::unordered_set<std::string>& getHoveredKeys() const { return hovered_key_ids_; }

private:
    std::unordered_set<std::string> hovered_key_ids_;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_CLICK_PROCESSOR_H