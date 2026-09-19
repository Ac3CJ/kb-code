#ifndef CV_KEYBOARD_PHYSICAL_HAND_H
#define CV_KEYBOARD_PHYSICAL_HAND_H

#include <array>
#include <cstdint>

namespace cv_keyboard {

struct PhysicalLandmark {
    float x_cm = 0.0f;
    float y_cm = 0.0f;
    float z_cm = 0.0f;  // Height above the virtual keyboard plane
    
    float vx_cm = 0.0f;
    float vy_cm = 0.0f;
    float vz_cm = 0.0f;
};

struct PhysicalHand {
    std::array<PhysicalLandmark, 21> landmarks;
    int handedness = 0;
    int64_t timestamp_us = 0;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_PHYSICAL_HAND_H