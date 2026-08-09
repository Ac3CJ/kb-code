#include "Application.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace cv_keyboard {

Application::Application() {
    cv::namedWindow(settings_.window_name, cv::WINDOW_NORMAL);

    if (!initCamera()) {
        std::cerr << "[Application] No camera available. Exiting.\n";
        running_ = false;
        return;
    }

    if (!initPipeline()) {
        std::cerr << "[Application] Pipeline initialisation failed. "
                  << "Running in camera-only mode (no hand tracking).\n";
        // Continue running without hand tracking
        mediator_ = nullptr;
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

bool Application::initPipeline() {
    mediator_ = std::make_unique<Mediator>();
    if (!mediator_->init()) {
        mediator_.reset();
        return false;
    }
    return true;
}

void Application::run() {
    if (!running_) {
        return;
    }

    // Declare display_frame OUTSIDE the loop.
    // Its internal memory buffer will be allocated once and recycled indefinitely.
    cv::Mat display_frame;

    while (running_) {
        cv::Mat raw_frame;
        if (!capture_.read(raw_frame)) {
            std::cerr << "[Application] Failed to grab frame. Retrying...\n";
            cv::waitKey(30);
            continue;
        }

        // O(1) header assignment. Bumps OpenCV refcount instead of copying 2MB of pixel data.
        // If VideoCapture needs to modify raw_frame next frame, OpenCV will automatically detach.
        // {
        //     std::lock_guard<std::mutex> lock(frame_mutex_);
        //     frame_ = raw_frame;
        // }

        // Run the pipeline (HandTracker inference) if available
        if (mediator_) {
            mediator_->processFrame(raw_frame);
        }

        // Render overlay without redundant cloning
        if (mediator_) {
            mediator_->renderOverlay(raw_frame, display_frame);
        } else {
            display_frame = raw_frame; // Zero-copy fallback
        }

        // Display the frame with overlay
        cv::imshow(settings_.window_name, display_frame);

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

// cv::Mat Application::currentFrame() const {
//     std::lock_guard<std::mutex> lock(frame_mutex_);
//     return frame_.clone();
// }

void Application::setCameraSource(CameraSource source) {
    if (settings_.active_source == source && capture_.isOpened()) {
        return;
    }

    settings_.active_source = source;

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
    std::string primary_path = settings_.activeDevicePath();
    std::cout << "[Application] Trying camera: " << primary_path << "\n";

    if (tryOpenCamera(primary_path)) {
        std::cout << "[Application] Camera opened: " << primary_path << "\n";
        return true;
    }

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

    cv::Mat test_frame;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (capture_.read(test_frame) && !test_frame.empty()) {
            return true;
        }
        cv::waitKey(50);
    }

    capture_.release();
    return false;
}

void Application::handleKeyInput(int key) {
    if (key == 27) {  // ESC key
        std::cout << "[Application] ESC pressed. Shutting down.\n";
        running_ = false;
    } else if (key == 'c' || key == 'C') {
        CameraSource new_source = (settings_.active_source == CameraSource::Laptop)
            ? CameraSource::Phone
            : CameraSource::Laptop;
        std::cout << "[Application] Toggling camera to: "
                  << (new_source == CameraSource::Laptop ? "Laptop" : "Phone") << "\n";
        setCameraSource(new_source);
    } else if (key == 's' || key == 'S') {
        if (mediator_) {
            mediator_->toggleFullSkeleton();
            std::cout << "[Application] Skeleton mode: "
                      << (mediator_->showFullSkeleton() ? "Full" : "Finger tips only")
                      << "\n";
        }
    } else if (key == 'd' || key == 'D') {
        if (mediator_) {
            mediator_->toggleDebugOverlay();
            std::cout << "[Application] Debug overlay: "
                      << (mediator_->showDebugOverlay() ? "ON" : "OFF")
                      << "\n";
        }
    } else if (key == 'k' || key == 'K') {
        if (mediator_) {
            mediator_->toggleKeyboard();
            std::cout << "[Application] Virtual Keyboard: "
                      << (mediator_->showKeyboard() ? "ON" : "OFF")
                      << "\n";
        }
    }
}

} // namespace cv_keyboard