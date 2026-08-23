#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <algorithm>
#include <map>
#include <memory>
#include <fstream>

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
        int highest_frame_processed = -1; // Tracks the "Frontier" of our cache
        bool is_paused = true;

        cv::Mat raw_frame;
        cv::Mat display_frame;

        std::cout << "\n=== CV Keyboard Offline Tester ===\n";
        std::cout << "Controls:\n";
        std::cout << "  SPACE : Play / Pause\n";
        std::cout << "  D     : Step Forward (1 Frame)\n";
        std::cout << "  A     : Step Backward (1 Frame)\n";
        std::cout << "  R     : Reset Zoom\n";
        std::cout << "  1-4   : Toggle Visuals (Grid, Skeleton, Hands, Keyboard)\n";
        std::cout << "  6     : Cycle Debug Mode (Off / Pose / Perf)\n";
        std::cout << "  ESC   : Quit\n";
        std::cout << "Mouse:\n";
        std::cout << "  Left Click + Drag : Draw box to zoom in\n";
        std::cout << "  Right Click       : Reset zoom\n==================================\n";

        cap.read(raw_frame);
        full_view_ = cv::Rect(0, 0, raw_frame.cols, raw_frame.rows);
        current_view_ = full_view_;

        bool force_process = true;
        bool summary_printed = false;

        std::vector<PerformanceMetrics> ablation_history;

        while (true) {
            if (!is_paused) {
                if (!cap.read(raw_frame)) {
                    if (!summary_printed && !ablation_history.empty()) {
                        double sum_homo = 0, sum_mp = 0, sum_fusion = 0, sum_click = 0, sum_total = 0;
                        double max_homo = 0, max_mp = 0, max_fusion = 0, max_click = 0, max_total = 0;

                        for (const auto& m : ablation_history) {
                            sum_homo += m.homography_ms;
                            max_homo = std::max(max_homo, m.homography_ms);
                            
                            sum_mp += m.mp_tracker_ms;
                            max_mp = std::max(max_mp, m.mp_tracker_ms);
                            
                            sum_fusion += m.sensor_fusion_ms;
                            max_fusion = std::max(max_fusion, m.sensor_fusion_ms);
                            
                            sum_click += m.click_process_ms;
                            max_click = std::max(max_click, m.click_process_ms);
                            
                            sum_total += m.total_ms;
                            max_total = std::max(max_total, m.total_ms);
                        }

                        size_t count = ablation_history.size();
                        std::cout << "\n\n=== ABLATION METRICS SUMMARY (" << count << " frames) ===\n"
                                  << std::fixed << std::setprecision(2)
                                  << "--- AVERAGE TIMES ---\n"
                                  << "  Homography   : " << (sum_homo / count) << " ms\n"
                                  << "  MP Tracker   : " << (sum_mp / count) << " ms\n"
                                  << "  Fusion       : " << (sum_fusion / count) << " ms\n"
                                  << "  Click Math   : " << (sum_click / count) << " ms\n"
                                  << "  Total        : " << (sum_total / count) << " ms\n"
                                  << "--- MAXIMUM TIMES ---\n"
                                  << "  Homography   : " << max_homo << " ms\n"
                                  << "  MP Tracker   : " << max_mp << " ms\n"
                                  << "  Fusion       : " << max_fusion << " ms\n"
                                  << "  Click Math   : " << max_click << " ms\n"
                                  << "  Total        : " << max_total << " ms\n"
                                  << "===============================================\n";
                        
                        summary_printed = true;
                    }
                    
                    std::cout << "[Tester] End of video reached. Pausing.\n";
                    is_paused = true;
                    current_frame_idx--;
                    cap.set(cv::CAP_PROP_POS_FRAMES, current_frame_idx);
                    cap.read(raw_frame);
                } else {
                    current_frame_idx++;
                }
                force_process = true;
            }

            if (force_process && !raw_frame.empty()) {
                // Determine if we are moving into the future or scrubbing the past
                if (current_frame_idx > highest_frame_processed) {
                    // NEW FRAME: Run the heavy MediaPipe graph
                    mediator_.processFrame(raw_frame);

                    ablation_history.push_back(mediator_.getMetrics());
                    
                    auto latest = mediator_.latestHands();
                    if (latest) {
                        // Create a brand new shared_ptr containing a physical copy of the vector
                        hand_cache_[current_frame_idx] = std::make_shared<std::vector<HandData>>(*latest);
                    } else {
                        hand_cache_[current_frame_idx] = nullptr;
                    }

                    highest_frame_processed = current_frame_idx;
                } else {
                    // PAST FRAME: Bypass MediaPipe and inject the cached data directly
                    mediator_.injectCachedHands(hand_cache_[current_frame_idx], raw_frame);
                }

                if (current_frame_idx <= highest_frame_processed) {
                    const auto& m = mediator_.getMetrics();
                    std::cout << "\r[Metrics F" << current_frame_idx << "] "
                              << "MP: " << std::fixed << std::setprecision(2) << m.mp_tracker_ms << "ms | "
                              << "Fus: " << m.sensor_fusion_ms << "ms | "
                              << "Homo: " << m.homography_ms << "ms | "
                              << "Clk: " << m.click_process_ms << "ms        " 
                              << std::flush;
                }

                mediator_.renderOverlay(raw_frame, display_frame);
                force_process = false;
            }

            // --- UI Rendering ---
            cv::Mat render_frame = display_frame.clone();
            
            if (is_selecting_ && selection_box_.width > 0 && selection_box_.height > 0) {
                cv::rectangle(render_frame, selection_box_, cv::Scalar(255, 0, 0), 2);
            }

            cv::Mat zoomed_frame;
            if (current_view_.width > 0 && current_view_.height > 0) {
                zoomed_frame = render_frame(current_view_);
            } else {
                zoomed_frame = render_frame;
            }

            std::string frame_text = "Frame: " + std::to_string(current_frame_idx) + " / " + std::to_string(total_frames) + (is_paused ? " (PAUSED)" : "");
            cv::putText(zoomed_frame, frame_text, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

            cv::imshow(window_name_, zoomed_frame);

            // --- Keyboard Inputs ---
            int key = cv::waitKey(is_paused ? 30 : 16); 
            
            if (key == 27) { // ESC
                std::cout << "\n\n[Tester] Video halted. Save Ablation Metrics to CSV? (y/n): ";
                char ans;
                std::cin >> ans;
                if (ans == 'y' || ans == 'Y') {
                    std::ofstream f("ablation_metrics.csv");
                    f << "Frame,Homography_ms,MediaPipe_ms,Fusion_ms,ClickMath_ms,Total_ms\n";
                    for (size_t i = 0; i < ablation_history.size(); ++i) {
                        const auto& m = ablation_history[i];
                        f << i << "," << m.homography_ms << "," << m.mp_tracker_ms << "," 
                          << m.sensor_fusion_ms << "," << m.click_process_ms << "," << m.total_ms << "\n";
                    }
                    std::cout << "[Tester] Saved " << ablation_history.size() << " frames to ablation_metrics.csv\n";
                }
                break;
            } else if (key == ' ') {
                is_paused = !is_paused;
            } else if ((key == 'd' || key == 'D') && is_paused) {
                if (current_frame_idx < total_frames - 1) {
                    current_frame_idx++;
                    cap.set(cv::CAP_PROP_POS_FRAMES, current_frame_idx);
                    cap.read(raw_frame);
                    force_process = true;
                }
            } else if ((key == 'a' || key == 'A') && is_paused) {
                if (current_frame_idx > 0) {
                    current_frame_idx--;
                    cap.set(cv::CAP_PROP_POS_FRAMES, current_frame_idx);
                    cap.read(raw_frame);
                    
                    // Flush the corrupted forward-moving history
                    mediator_.resetClickState(); 
                    
                    // Pre-Roll T-2
                    if (hand_cache_.count(current_frame_idx - 2)) {
                        mediator_.warmUpClickProcessor(hand_cache_[current_frame_idx - 2], raw_frame.cols, raw_frame.rows);
                    }
                    
                    // Pre-Roll T-1
                    if (hand_cache_.count(current_frame_idx - 1)) {
                        mediator_.warmUpClickProcessor(hand_cache_[current_frame_idx - 1], raw_frame.cols, raw_frame.rows);
                    }
                    
                    force_process = true;
                }
            } else if (key == 'r' || key == 'R') {
                resetZoom();
            } else if (key == '1') {
                mediator_.toggleGrid();
                force_process = true; // Force redraw of overlay
            } else if (key == '2') {
                mediator_.toggleFullSkeleton();
                force_process = true;
            } else if (key == '3') {
                mediator_.toggleHands();
                force_process = true;
            } else if (key == '4') {
                mediator_.toggleKeyboard();
                force_process = true;
            } else if (key == '6') {
                mediator_.cycleDebugMode();
                std::cout << "[Tester] Debug mode: ";
                if (mediator_.debugMode() == DebugMode::OFF) std::cout << "OFF\n";
                else if (mediator_.debugMode() == DebugMode::POSE) std::cout << "POSE\n";
                else std::cout << "PERF\n";
                force_process = true;
            }
        }
    }

private:
    void resetZoom() {
        current_view_ = full_view_;
        is_selecting_ = false;
    }

    static void onMouse(int event, int x, int y, int flags, void* userdata) {
        OfflineTester* tester = reinterpret_cast<OfflineTester*>(userdata);
        tester->handleMouse(event, x, y, flags);
    }

    void handleMouse(int event, int x, int y, int flags) {
        int real_x = current_view_.x + static_cast<int>(x * (static_cast<float>(current_view_.width) / current_view_.width));
        int real_y = current_view_.y + static_cast<int>(y * (static_cast<float>(current_view_.height) / current_view_.height));

        if (event == cv::EVENT_LBUTTONDOWN) {
            is_selecting_ = true;
            selection_start_ = cv::Point(real_x, real_y);
            selection_box_ = cv::Rect(real_x, real_y, 0, 0);
        } else if (event == cv::EVENT_MOUSEMOVE && is_selecting_) {
            selection_box_.x = std::min(real_x, selection_start_.x);
            selection_box_.y = std::min(real_y, selection_start_.y);
            selection_box_.width = std::abs(real_x - selection_start_.x);
            selection_box_.height = std::abs(real_y - selection_start_.y);
        } else if (event == cv::EVENT_LBUTTONUP) {
            is_selecting_ = false;
            if (selection_box_.width > 10 && selection_box_.height > 10) {
                selection_box_ &= full_view_;
                current_view_ = selection_box_;
            }
        } else if (event == cv::EVENT_RBUTTONDOWN) {
            resetZoom();
        }
    }

    std::string video_path_;
    std::string window_name_ = "Offline Tester";
    Mediator mediator_;

    // Data Cache for Deterministic Playback
    std::map<int, std::shared_ptr<const std::vector<HandData>>> hand_cache_;

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