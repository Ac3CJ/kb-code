#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <algorithm>

#include "Mediator.h"

namespace cv_keyboard {

class OfflineTester {
public:
    OfflineTester(const std::string& video_path) : video_path_(video_path) {
        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        cv::setMouseCallback(window_name_, onMouse, this);
    }

    ~OfflineTester() {
        cv::destroyWindow(window_name_);
    }

    void run() {
        cv::VideoCapture cap(video_path_);
        if (!cap.isOpened()) {
            std::cerr << "[Tester] Failed to open video file: " << video_path_ << "\n";
            return;
        }

        if (!mediator_.init()) {
            std::cerr << "[Tester] Mediator failed to initialise.\n";
            return;
        }

        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        int current_frame_idx = 0;
        bool is_paused = true;

        cv::Mat raw_frame;
        cv::Mat display_frame;

        std::cout << "\n=== CV Keyboard Offline Tester ===\n";
        std::cout << "Controls:\n";
        std::cout << "  SPACE : Play / Pause\n";
        std::cout << "  D     : Step Forward (1 Frame)\n";
        std::cout << "  A     : Step Backward (1 Frame)\n";
        std::cout << "  R     : Reset Zoom\n";
        std::cout << "  ESC   : Quit\n";
        std::cout << "Mouse:\n";
        std::cout << "  Left Click + Drag : Draw box to zoom in\n";
        std::cout << "  Right Click       : Reset zoom\n==================================\n";

        // Read the very first frame to initialize the view
        cap.read(raw_frame);
        full_view_ = cv::Rect(0, 0, raw_frame.cols, raw_frame.rows);
        current_view_ = full_view_;

        bool frame_needs_processing = true;
        while (true) {
            if (!is_paused) {
                if (!cap.read(raw_frame)) {
                    std::cout << "[Tester] End of video reached. Pausing.\n";
                    is_paused = true;
                    // Step back to last valid frame to prevent crashing
                    current_frame_idx--;
                    cap.set(cv::CAP_PROP_POS_FRAMES, current_frame_idx);
                    cap.read(raw_frame);
                } else {
                    current_frame_idx++;
                }
                frame_needs_processing = true;
            }

            // 1. Process the math on the FULL, uncropped frame so coordinates match
            if (frame_needs_processing && !raw_frame.empty()) {
                mediator_.processFrame(raw_frame);
                mediator_.renderOverlay(raw_frame, display_frame);
                frame_needs_processing = false;
            }

            // 2. Handle Zooming / Cropping for the UI
            cv::Mat render_frame = display_frame.clone();
            
            // If the user is currently dragging to zoom, draw a blue selection box
            if (is_selecting_ && selection_box_.width > 0 && selection_box_.height > 0) {
                cv::rectangle(render_frame, selection_box_, cv::Scalar(255, 0, 0), 2);
            }

            // Crop the frame based on the current zoom level
            cv::Mat zoomed_frame;
            if (current_view_.width > 0 && current_view_.height > 0) {
                zoomed_frame = render_frame(current_view_);
            } else {
                zoomed_frame = render_frame; // Fallback
            }

            // Draw frame counter
            std::string frame_text = "Frame: " + std::to_string(current_frame_idx) + " / " + std::to_string(total_frames) + (is_paused ? " (PAUSED)" : "");
            cv::putText(zoomed_frame, frame_text, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

            cv::imshow(window_name_, zoomed_frame);

            // 3. Handle Keyboard Input
            int key = cv::waitKey(is_paused ? 30 : 16); 
            
            if (key == 27) { // ESC
                break;
            } else if (key == ' ') {
                is_paused = !is_paused;
            } else if ((key == 'd' || key == 'D') && is_paused) {
                if (current_frame_idx < total_frames - 1) {
                    current_frame_idx++;
                    cap.set(cv::CAP_PROP_POS_FRAMES, current_frame_idx);
                    cap.read(raw_frame);
                    frame_needs_processing = true;
                }
            } else if ((key == 'a' || key == 'A') && is_paused) {
                if (current_frame_idx > 0) {
                    current_frame_idx--;
                    cap.set(cv::CAP_PROP_POS_FRAMES, current_frame_idx);
                    cap.read(raw_frame);
                    frame_needs_processing = true;
                }
            } else if (key == 'r' || key == 'R') {
                resetZoom();
            } 
            // --- NEW: Toggle Overlays ---
            else if (key == '1') {
                mediator_.toggleGrid();
                std::cout << "[Tester] Grid: " << (mediator_.showGrid() ? "ON" : "OFF") << "\n";
            } else if (key == '2') {
                mediator_.toggleFullSkeleton();
                std::cout << "[Tester] Skeleton Mode: " << (mediator_.showFullSkeleton() ? "Full" : "Fingertips Only") << "\n";
            } else if (key == '3') {
                mediator_.toggleHands();
                std::cout << "[Tester] MediaPipe Hands: " << (mediator_.showHands() ? "ON" : "OFF") << "\n";
            } else if (key == '4') {
                mediator_.toggleKeyboard();
                std::cout << "[Tester] Virtual Keyboard: " << (mediator_.showKeyboard() ? "ON" : "OFF") << "\n";
            }
        }
    }

private:
    void resetZoom() {
        current_view_ = full_view_;
        is_selecting_ = false;
    }

    // Static wrapper to route OpenCV C-style callback back into the C++ class instance
    static void onMouse(int event, int x, int y, int flags, void* userdata) {
        OfflineTester* tester = reinterpret_cast<OfflineTester*>(userdata);
        tester->handleMouse(event, x, y, flags);
    }

    void handleMouse(int event, int x, int y, int flags) {
        // Map the mouse click (which is relative to the zoomed window) back to the full frame coordinates
        int real_x = current_view_.x + static_cast<int>(x * (static_cast<float>(current_view_.width) / current_view_.width));
        int real_y = current_view_.y + static_cast<int>(y * (static_cast<float>(current_view_.height) / current_view_.height));

        if (event == cv::EVENT_LBUTTONDOWN) {
            is_selecting_ = true;
            selection_start_ = cv::Point(real_x, real_y);
            selection_box_ = cv::Rect(real_x, real_y, 0, 0);
        } 
        else if (event == cv::EVENT_MOUSEMOVE && is_selecting_) {
            selection_box_.x = std::min(real_x, selection_start_.x);
            selection_box_.y = std::min(real_y, selection_start_.y);
            selection_box_.width = std::abs(real_x - selection_start_.x);
            selection_box_.height = std::abs(real_y - selection_start_.y);
        } 
        else if (event == cv::EVENT_LBUTTONUP) {
            is_selecting_ = false;
            if (selection_box_.width > 10 && selection_box_.height > 10) {
                // Ensure the box stays within the bounds of the original image
                selection_box_ &= full_view_;
                current_view_ = selection_box_;
            }
        }
        else if (event == cv::EVENT_RBUTTONDOWN) {
            resetZoom();
        }
    }

    std::string video_path_;
    std::string window_name_ = "Offline Tester";
    Mediator mediator_;

    // Zoom and Pan states
    cv::Rect full_view_;
    cv::Rect current_view_;
    cv::Rect selection_box_;
    cv::Point selection_start_;
    bool is_selecting_ = false;
};

} // namespace cv_keyboard

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./offline_tester <path_to_video.mp4>\n";
        return 1;
    }

    std::string video_path = argv[1];
    cv_keyboard::OfflineTester tester(video_path);
    tester.run();

    return 0;
}