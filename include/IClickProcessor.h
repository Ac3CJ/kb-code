#ifndef CV_KEYBOARD_ICLICK_PROCESSOR_H
#define CV_KEYBOARD_ICLICK_PROCESSOR_H

#include <string>
#include <vector>
#include <unordered_set>
#include "IHandTracker.h"
#include "KeyboardMap.h"

namespace cv_keyboard {

class IClickProcessor {
public:
    virtual ~IClickProcessor() = default;

    /// Reads fingertip data and keyboard mapping to compute states.
    virtual void process(const std::vector<HandData>& hands, 
                         const KeyboardMap& keyboard_map, 
                         int frame_width, 
                         int frame_height) = 0;

    /// Query hover states
    virtual bool isHovered(const std::string& key_id) const = 0;
    virtual const std::unordered_set<std::string>& getHoveredKeys() const = 0;

    /// Query physical click states (New for touch detection)
    virtual bool isClicked(const std::string& key_id) const = 0;
    virtual const std::unordered_set<std::string>& getClickedKeys() const = 0;

    // Reset internal state to prevent glitches during backwards playback
    virtual void reset() = 0;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_ICLICK_PROCESSOR_H