#ifndef CV_KEYBOARD_INTERPOLATION_PROCESSOR_H
#define CV_KEYBOARD_INTERPOLATION_PROCESSOR_H

#include "BaseClickProcessor.h"
#include <map>
#include <deque>
#include <algorithm>

namespace cv_keyboard {

struct FrameState {
    float x;
    float y;
    float vy;
};

class InterpolationProcessor : public BaseClickProcessor {
public:
    InterpolationProcessor() = default;
    ~InterpolationProcessor() override = default;

protected:
    void detectClicks(const std::vector<HandData>& hands, 
                      const KeyboardMap& keyboard_map, 
                      int frame_width, 
                      int frame_height) override;

    void resetHistory() override {state_history_.clear();}
private:
    std::map<int, std::deque<FrameState>> state_history_;
};

} // namespace cv_keyboard
#endif // CV_KEYBOARD_INTERPOLATION_PROCESSOR_H