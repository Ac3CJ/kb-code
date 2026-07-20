#ifndef CV_KEYBOARD_SETTINGS_H
#define CV_KEYBOARD_SETTINGS_H

#include <string>

namespace cv_keyboard {

enum class CameraSource {
    Laptop,
    Phone
};

struct Settings {
    /// Which camera source is currently selected
    CameraSource active_source = CameraSource::Phone;

    /// Device path for the laptop's built-in webcam
    std::string laptop_device_path = "/dev/video0";

    /// Device path for the phone camera (via V4L2 / scrcpy)
    std::string phone_device_path = "/dev/video2";

    /// Window name for the debug / overlay display
    std::string window_name = "CV Keyboard";

    /// Returns the device path for the currently active source
    std::string activeDevicePath() const {
        return active_source == CameraSource::Laptop
            ? laptop_device_path
            : phone_device_path;
    }
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_SETTINGS_H