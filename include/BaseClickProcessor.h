#ifndef CV_KEYBOARD_BASE_CLICK_PROCESSOR_H
#define CV_KEYBOARD_BASE_CLICK_PROCESSOR_H

#include "IClickProcessor.h"

namespace cv_keyboard {

class BaseClickProcessor : public IClickProcessor {
public:
    BaseClickProcessor();
    ~BaseClickProcessor() override = default;

    // The main entry point. It calculates hovers, then delegates clicks to the child class.
    void process(const std::vector<HandData>& hands, 
                 const KeyboardMap& keyboard_map, 
                 int frame_width, 
                 int frame_height) override;

    bool isHovered(const std::string& key_id) const override;
    const std::unordered_set<std::string>& getHoveredKeys() const override;

    bool isClicked(const std::string& key_id) const override;
    const std::unordered_set<std::string>& getClickedKeys() const override;

    void reset() override;

protected:
    // Pure virtual function: Child classes MUST implement their specific touch algorithms here!
    virtual void detectClicks(const std::vector<HandData>& hands, 
                              const KeyboardMap& keyboard_map, 
                              int frame_width, 
                              int frame_height) = 0;

    virtual void resetHistory() = 0;

    std::unordered_set<std::string> hovered_key_ids_;
    std::unordered_set<std::string> clicked_key_ids_;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_BASE_CLICK_PROCESSOR_H