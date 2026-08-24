#ifndef CV_KEYBOARD_APPLICATION_H
#define CV_KEYBOARD_APPLICATION_H

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "Settings.h"
#include "Mediator.h"

namespace cv_keyboard {

class Application {
public:
    explicit Application(const Settings& settings = Settings());
    ~Application();

    /// Main loop: capture frames until ESC or window close
    void run();

    /// Thread-safe accessor for the latest frame (for Overlay thread)
    cv::Mat currentFrame() const;

    /// Runtime switch between laptop and phone camera
    /// Re-initialises the camera if the source changes
    void setCameraSource(CameraSource source);

    /// Access the settings (mutable for config changes)
    Settings& settings();
    const Settings& settings() const;

    /// Check if the application is still running
    bool isRunning() const;

private:
    /// Initialise the camera using the active source in settings
    /// Returns true if successful, false otherwise
    bool initCamera();

    /// Try to open a specific device path
    /// Returns true if the camera opened successfully
    bool tryOpenCamera(const std::string& device_path);

    /// Handle keyboard input (ESC to quit, 'c' to toggle camera, etc.)
    void handleKeyInput(int key);

    /// Initialise the MediaPipe pipeline
    bool initPipeline();

    Settings settings_;
    cv::VideoCapture capture_;
    cv::Mat frame_;
    mutable std::mutex frame_mutex_;
    std::atomic<bool> running_{false};

    std::unique_ptr<Mediator> mediator_;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_APPLICATION_H