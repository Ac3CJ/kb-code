#ifndef CV_KEYBOARD_ZERO_CROSSING_PROCESSOR_H
#define CV_KEYBOARD_ZERO_CROSSING_PROCESSOR_H

#include "BaseClickProcessor.h"
#include <map>
#include <deque>

namespace cv_keyboard {

class ZeroCrossingProcessor : public BaseClickProcessor {
public:
    ZeroCrossingProcessor() = default;
    ~ZeroCrossingProcessor() override = default;

protected:
    // TUNABLE PARAMETERS (Normalized pixel velocity per frame)
    // Adjust these to filter out slow hovering movements
    const float STRIKE_THRESH = 0.0015f; 
    const float RELEASE_THRESH = 0.0000015f;

    void detectClicks(const std::vector<HandData>& hands, 
                      const KeyboardMap& keyboard_map, 
                      int frame_width, 
                      int frame_height) override;

    void resetHistory() override {vy_history_.clear();}

private:
    std::map<int, std::deque<float>> vy_history_;
};

} // namespace cv_keyboard
#endif // CV_KEYBOARD_ZERO_CROSSING_PROCESSOR_H