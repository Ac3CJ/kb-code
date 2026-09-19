#ifndef CV_KEYBOARD_BASE_PHYSICAL_CLICK_PROCESSOR_H
#define CV_KEYBOARD_BASE_PHYSICAL_CLICK_PROCESSOR_H

#include "IPhysicalClickProcessor.h"
#include <map>

namespace cv_keyboard {

class BasePhysicalClickProcessor : public IPhysicalClickProcessor {
public:
    BasePhysicalClickProcessor() = default;
    ~BasePhysicalClickProcessor() override = default;

    void process(const std::vector<PhysicalHand>& physical_hands, 
                 const KeyboardMap& keyboard_map) override;

    bool isHovered(const std::string& key_id) const override;
    const std::unordered_set<std::string>& getHoveredKeys() const override;

    bool isClicked(const std::string& key_id) const override;
    const std::unordered_set<std::string>& getClickedKeys() const override;

    void reset() override;

protected:
    // Child classes (e.g., KinematicZProcessor) implement their 3D math here
    virtual void detectClicks(const std::vector<PhysicalHand>& physical_hands, 
                              const KeyboardMap& keyboard_map) = 0;

    virtual void resetHistory() = 0;

    std::unordered_set<std::string> hovered_key_ids_;
    std::unordered_set<std::string> clicked_key_ids_;

    std::map<int, std::string> finger_hovers_;
    std::map<int, std::string> active_clicks_;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_BASE_PHYSICAL_CLICK_PROCESSOR_H