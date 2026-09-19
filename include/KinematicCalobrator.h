#ifndef CV_KEYBOARD_KINEMATIC_CALIBRATOR_H
#define CV_KEYBOARD_KINEMATIC_CALIBRATOR_H

#include "IHandTracker.h"
#include "PhysicalHand.h"
#include "KeyboardMap.h"

#include <vector>
#include <string>

namespace cv_keyboard {

class KinematicCalibrator {
public:
    KinematicCalibrator() = default;
    ~KinematicCalibrator() = default;

    // Triggered when the user places hands flat on the desk
    void calibrate(const std::vector<HandData>& hands, const KeyboardMap& kb_map);
    
    // Core pipeline step: converts 2D normalized pixels into 3D centimeters
    std::vector<PhysicalHand> transform(const std::vector<HandData>& hands, const KeyboardMap& kb_map);

    // Profile I/O for offline testing
    bool loadProfile(const std::string& filepath);
    bool saveProfile(const std::string& filepath) const;
    
    bool isCalibrated() const { return is_calibrated_; }

private:
    bool is_calibrated_ = false;

    // TODO: We will add the internal state variables for storing 
    // baseline bone lengths (e.g., distance from MCP to PIP) in the next step.
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_KINEMATIC_CALIBRATOR_H