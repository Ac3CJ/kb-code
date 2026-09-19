#ifndef CV_KEYBOARD_IPHYSICAL_CLICK_PROCESSOR_H
#define CV_KEYBOARD_IPHYSICAL_CLICK_PROCESSOR_H

#include "PhysicalHand.h"
#include "KeyboardMap.h"
#include <vector>
#include <string>
#include <unordered_set>

namespace cv_keyboard {

class IPhysicalClickProcessor {
public:
    virtual ~IPhysicalClickProcessor() = default;

    // Replaces the 2D HandData and drops frame dimensions
    virtual void process(const std::vector<PhysicalHand>& physical_hands, 
                         const KeyboardMap& keyboard_map) = 0;

    virtual bool isHovered(const std::string& key_id) const = 0;
    virtual const std::unordered_set<std::string>& getHoveredKeys() const = 0;

    virtual bool isClicked(const std::string& key_id) const = 0;
    virtual const std::unordered_set<std::string>& getClickedKeys() const = 0;

    virtual void reset() = 0;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_IPHYSICAL_CLICK_PROCESSOR_H