#ifndef CV_KEYBOARD_MEDIATOR_H
#define CV_KEYBOARD_MEDIATOR_H

#include <opencv2/core/mat.hpp>
#include <memory>
#include <mutex>
#include <vector>
#include <cstdint>

#include "HandTracker.h"

namespace cv_keyboard {

class Mediator {
public:
    Mediator();
    ~Mediator();

    bool init();
    void processFrame(const cv::Mat& frame);
    std::shared_ptr<const std::vector<HandData>> latestHands() const;
    bool isInitialised() const;

    bool showFullSkeleton() const { return show_full_skeleton_; }
    void setShowFullSkeleton(bool show) { show_full_skeleton_ = show; }
    void toggleFullSkeleton() { show_full_skeleton_ = !show_full_skeleton_; }

    bool showDebugOverlay() const { return show_debug_overlay_; }
    void setShowDebugOverlay(bool show) { show_debug_overlay_ = show; }
    void toggleDebugOverlay() { show_debug_overlay_ = !show_debug_overlay_; }

    bool showGrid() const { return show_grid_; }
    void setShowGrid(bool show) { show_grid_ = show; }
    void toggleGrid() { show_grid_ = !show_grid_; }

    void renderOverlay(const cv::Mat& raw_frame, cv::Mat& display_frame);

private:
    void drawGrid(cv::Mat& frame, int step = 100) const;

    std::unique_ptr<HandTracker> hand_tracker_;
    std::shared_ptr<const std::vector<HandData>> latest_hands_;
    mutable std::mutex hands_mutex_;

    mutable cv::Mat cached_grid_overlay_;
    mutable cv::Mat cached_grid_mask_; 
    mutable cv::Size last_frame_size_;

    bool show_full_skeleton_ = false;
    bool show_debug_overlay_ = false;
    bool show_grid_ = true;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_MEDIATOR_H