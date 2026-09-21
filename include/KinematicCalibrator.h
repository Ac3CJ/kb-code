#ifndef CV_KEYBOARD_KINEMATIC_CALIBRATOR_H
#define CV_KEYBOARD_KINEMATIC_CALIBRATOR_H

#include "IHandTracker.h"
#include "PhysicalHand.h"
#include "KeyboardMap.h"

#include <vector>
#include <string>
#include <map>
#include <array>

namespace cv_keyboard {

struct HandProfile {
    bool is_valid = false;
    std::map<int, float> bone_lengths_cm; 
    std::map<std::string, float> angles_rad; // NEW: Stores triangle/knuckle angles
};

class KinematicCalibrator {
public:
    KinematicCalibrator() = default;
    ~KinematicCalibrator() = default;

    void startCalibration();
    bool updateCalibration(const std::vector<HandData>& hands, const KeyboardMap& kb_map, int frame_width, int frame_height);

    bool saveProfile(const std::string& filepath) const;
    bool loadProfile(const std::string& filepath);

    std::vector<PhysicalHand> transform(const std::vector<HandData>& hands, const KeyboardMap& kb_map, int frame_width, int frame_height);

    bool isCalibrated() const { return is_calibrated_; }
    bool isCalibrating() const { return is_calibrating_; }

private:
    bool is_calibrated_ = false;
    bool is_calibrating_ = false;
    
    int frames_collected_ = 0;
    const int MAX_CALIBRATION_FRAMES = 30;

    std::vector<std::array<Landmark, 21>> hand0_buffer_;
    std::vector<std::array<Landmark, 21>> hand1_buffer_;

    HandProfile hand0_profile_;
    HandProfile hand1_profile_;

    void finalizeCalibration(const KeyboardMap& kb_map, int frame_width, int frame_height);
    void calculateBoneLengths(const std::vector<std::array<Landmark, 21>>& buffer, HandProfile& profile, const KeyboardMap& kb_map, int frame_width, int frame_height);
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_KINEMATIC_CALIBRATOR_H