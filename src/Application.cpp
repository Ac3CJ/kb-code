#include "Application.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace cv_keyboard {

Application::Application() {
    cv::namedWindow(settings_.window_name, cv::WINDOW_AUTOSIZE);

    if (!initCamera()) {
        std::cerr << "[Application] No camera available. Exiting.\n";
        running_ = false;
        return;
    }

    running_ = true;
}

Application::~Application() {
    running_ = false;
    if (capture_.isOpened()) {
        capture_.release();
    }
    cv::destroyWindow(settings_.window_name);
}

void Application::run() {
    if (!running_) {
        return;
    }

    while (running_) {
        cv::Mat raw_frame;
        if (!capture_.read(raw_frame)) {
            std::cerr << "[Application] Failed to grab frame. Retrying...\n";
            // Small delay to avoid busy-looping on camera failure
            cv::waitKey(30);
            continue;
        }

        // Store the frame under the mutex so the overlay thread can read it
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            raw_frame.copyTo(frame_);
        }

        // Display the frame (placeholder for future overlay rendering)
        cv::imshow(settings_.window_name, raw_frame);

        // Process keyboard input
        int key = cv::waitKey(1);
        handleKeyInput(key);

        // Check if the user closed the window via the title-bar X button
        if (cv::getWindowProperty(settings_.window_name, cv::WND_PROP_VISIBLE) < 1) {
            std::cout << "[Application] Window closed. Shutting down.\n";
            running_ = false;
        }
    }
}

cv::Mat Application::currentFrame() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return frame_.clone();
}

void Application::setCameraSource(CameraSource source) {
    if (settings_.active_source == source && capture_.isOpened()) {
        return;  // Already using this source
    }

    settings_.active_source = source;

    // Release the old camera before trying the new one
    if (capture_.isOpened()) {
        capture_.release();
    }

    if (!initCamera()) {
        std::cerr << "[Application] Failed to switch camera source.\n";
    }
}

Settings& Application::settings() {
    return settings_;
}

const Settings& Application::settings() const {
    return settings_;
}

bool Application::isRunning() const {
    return running_;
}

bool Application::initCamera() {
    // First, try the configured device path for the current source
    std::string primary_path = settings_.activeDevicePath();
    std::cout << "[Application] Trying camera: " << primary_path << "\n";

    if (tryOpenCamera(primary_path)) {
        std::cout << "[Application] Camera opened: " << primary_path << "\n";
        return true;
    }

    // If phone cam was requested but failed, fall back to laptop
    if (settings_.active_source == CameraSource::Phone) {
        std::cout << "[Application] Phone camera unavailable. Falling back to laptop.\n";
        settings_.active_source = CameraSource::Laptop;
        std::string fallback_path = settings_.activeDevicePath();
        std::cout << "[Application] Trying camera: " << fallback_path << "\n";

        if (tryOpenCamera(fallback_path)) {
            std::cout << "[Application] Camera opened: " << fallback_path << "\n";
            return true;
        }
    }

    return false;
}

bool Application::tryOpenCamera(const std::string& device_path) {
    capture_.open(device_path);
    if (!capture_.isOpened()) {
        return false;
    }

    // Give the camera a moment to initialise and grab a test frame
    cv::Mat test_frame;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (capture_.read(test_frame) && !test_frame.empty()) {
            return true;
        }
        cv::waitKey(50);
    }

    // Camera opened but didn't produce valid frames — release it
    capture_.release();
    return false;
}

void Application::handleKeyInput(int key) {
    if (key == 27) {  // ESC key
        std::cout << "[Application] ESC pressed. Shutting down.\n";
        running_ = false;
    } else if (key == 'c' || key == 'C') {
        // Toggle between phone and laptop camera
        CameraSource new_source = (settings_.active_source == CameraSource::Laptop)
            ? CameraSource::Phone
            : CameraSource::Laptop;
        std::cout << "[Application] Toggling camera to: "
                  << (new_source == CameraSource::Laptop ? "Laptop" : "Phone") << "\n";
        setCameraSource(new_source);
    }
}

} // namespace cv_keyboard