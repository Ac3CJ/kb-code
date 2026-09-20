#ifndef CV_KEYBOARD_MEDIATOR_H
#define CV_KEYBOARD_MEDIATOR_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <memory>
#include <mutex>
#include <vector>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "Settings.h"
#include "KeyboardMap.h"
#include "DebugVisualizer.h"

#include "MediaPipeTracker.h"
#include "RTMPoseTracker.h"

#include "ZeroCrossingProcessor.h"
#include "InterpolationProcessor.h"

#include "TypingEngine.h"
#include "KinematicCalibrator.h"
#include "IPhysicalClickProcessor.h"

namespace cv_keyboard {

struct PerformanceMetrics {
    double homography_ms = 0.0;
    double mp_tracker_ms = 0.0;
    double sensor_fusion_ms = 0.0;
    double click_process_ms = 0.0;
    double total_ms = 0.0;
};

enum class DebugMode { OFF, POSE, PERF };

class Mediator {
public:
    explicit Mediator(const std::string& tracker_type = "rtmpose", 
                      const std::string& processor_type = "zero_crossing");
    ~Mediator();

    bool init(const std::string& profile_path = "");
    void triggerCalibration() { kinematic_calibrator_.startCalibration(); }

    void processFrame(const cv::Mat& frame);
    std::shared_ptr<const std::vector<HandData>> latestHands() const;
    std::shared_ptr<const std::vector<PhysicalHand>> latestPhysicalHands() const; // NEW

    bool isInitialised() const;

    // --- State Toggles ---
    bool showFullSkeleton() const { return show_full_skeleton_; }
    void setShowFullSkeleton(bool show) { show_full_skeleton_ = show; }
    void toggleFullSkeleton() { show_full_skeleton_ = !show_full_skeleton_; }

    bool showHands() const { return show_hands_; }
    void setShowHands(bool show) { show_hands_ = show; }
    void toggleHands() { show_hands_ = !show_hands_; }

    void updateCameraIntrinsics(CameraSource source);

    DebugMode debugMode() const { return debug_mode_; }
    void cycleDebugMode() {
        if (debug_mode_ == DebugMode::OFF) debug_mode_ = DebugMode::POSE;
        else if (debug_mode_ == DebugMode::POSE) debug_mode_ = DebugMode::PERF;
        else debug_mode_ = DebugMode::OFF;
    }

    FilterMode filterMode() const { return filter_mode_; }
    void cycleFilterMode() {
        if (filter_mode_ == FilterMode::NONE) filter_mode_ = FilterMode::SOBEL;
        else if (filter_mode_ == FilterMode::SOBEL) filter_mode_ = FilterMode::LAPLACIAN;
        else if (filter_mode_ == FilterMode::LAPLACIAN) filter_mode_ = FilterMode::CANNY;
        else if (filter_mode_ == FilterMode::CANNY) filter_mode_ = FilterMode::BLACKHAT;
        else if (filter_mode_ == FilterMode::BLACKHAT) filter_mode_ = FilterMode::FRAMEDIFF;
        else if (filter_mode_ == FilterMode::FRAMEDIFF) filter_mode_ = FilterMode::LAB_LIGHTNESS;
        else if (filter_mode_ == FilterMode::LAB_LIGHTNESS) filter_mode_ = FilterMode::HEATMAP;
        else if (filter_mode_ == FilterMode::MOTION_LIGHTNESS) filter_mode_ = FilterMode::MOTION_HEATMAP;
        else if (filter_mode_ == FilterMode::HEATMAP) filter_mode_ = FilterMode::MOTION_LIGHTNESS;
        else filter_mode_ = FilterMode::NONE; 
    }

    const PerformanceMetrics& getMetrics() const { return metrics_; }

    bool showGrid() const { return show_grid_; }
    void setShowGrid(bool show) { show_grid_ = show; }
    void toggleGrid() { show_grid_ = !show_grid_; }

    bool showKeyboard() const { return show_keyboard_; }
    void setShowKeyboard(bool show) { show_keyboard_ = show; }
    void toggleKeyboard() { show_keyboard_ = !show_keyboard_; }

    void renderOverlay(const cv::Mat& raw_frame, cv::Mat& display_frame);

    void injectCachedHands(std::shared_ptr<const std::vector<HandData>> cached_hands, const cv::Mat& frame);

    void resetClickState() {
        if (click_processor_) click_processor_->reset(); 
        if (physical_click_processor_) physical_click_processor_->reset(); // NEW
        typing_engine_.reset();
    }
    
    void warmUpClickProcessor(std::shared_ptr<const std::vector<HandData>> past_hands, int frame_width, int frame_height);

private:
    void drawGrid(cv::Mat& frame, int step = 100) const;
    void drawVirtualKeyboard(cv::Mat& frame);
    void drawPhysicalKeyboard(cv::Mat& frame);
    void drawHands(cv::Mat& frame);
    
    void drawDebug(cv::Mat& frame);

    KeyboardMap virtual_keyboard_;
    
    // Parallel Processors
    std::unique_ptr<IClickProcessor> click_processor_;
    std::unique_ptr<IPhysicalClickProcessor> physical_click_processor_; // NEW
    
    mutable cv::Mat cached_kb_overlay_;
    mutable cv::Mat cached_kb_mask_;

    std::unique_ptr<IHandTracker> hand_tracker_;
    
    std::shared_ptr<const std::vector<HandData>> latest_hands_;
    std::shared_ptr<const std::vector<PhysicalHand>> latest_physical_hands_; // NEW
    mutable std::mutex hands_mutex_;

    TypingEngine typing_engine_;

    mutable cv::Mat cached_grid_overlay_;
    mutable cv::Mat cached_grid_mask_; 
    mutable cv::Size last_frame_size_;

    bool show_grid_ = false;          
    bool show_full_skeleton_ = false; 
    bool show_hands_ = true;          
    bool show_keyboard_ = true;       
    
    PerformanceMetrics metrics_;
    DebugMode debug_mode_ = DebugMode::OFF;
    void drawPerfMetrics(cv::Mat& frame) const;

    KinematicCalibrator kinematic_calibrator_;
    FilterMode filter_mode_ = FilterMode::NONE;
    DebugVisualizer debug_visualizer_;
    bool show_debug_overlay_ = false;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_MEDIATOR_H