#ifndef CV_KEYBOARD_MEDIATOR_H
#define CV_KEYBOARD_MEDIATOR_H

#include <opencv2/core/mat.hpp>
#include <memory>
#include <mutex>
#include <vector>
#include <deque>
#include <utility>
#include <cstdint>

#include "HandTracker.h"

namespace cv_keyboard {

/// Pipeline orchestrator.
///
/// Owns the HandTracker and will later own KeyboardMap, ClickProcessor, etc.
/// Each frame flows: HandTracker::detect() → (future) ClickProcessor.
/// The Mediator exposes the latest results for the overlay thread to read.
class Mediator {
public:
    Mediator();
    ~Mediator();

    /// Initialise all pipeline components (HandTracker, etc.).
    /// Returns true if all components initialised successfully.
    bool init();

    /// Process a single camera frame through the pipeline.
    /// Must be called from the main capture loop.
    void processFrame(const cv::Mat& frame);

    /// Access the latest detected hands (thread-safe via internal snapshot).
    std::shared_ptr<const std::vector<HandData>> latestHands() const;

    /// Check whether the pipeline is initialised and ready.
    bool isInitialised() const;

    /// Overlay display mode toggles
    bool showFullSkeleton() const { return show_full_skeleton_; }
    void setShowFullSkeleton(bool show) { show_full_skeleton_ = show; }
    void toggleFullSkeleton() { show_full_skeleton_ = !show_full_skeleton_; }

    bool showDebugOverlay() const { return show_debug_overlay_; }
    void setShowDebugOverlay(bool show) { show_debug_overlay_ = show; }
    void toggleDebugOverlay() { show_debug_overlay_ = !show_debug_overlay_; }

    bool showGrid() const { return show_grid_; }
    void setShowGrid(bool show) { show_grid_ = show; }
    void toggleGrid() { show_grid_ = !show_grid_; }

    /// Render the overlay (hand skeleton + optional debug text) onto the display frame.
    /// Uses O(1) buffer swapping to extract synchronized frames without deep copying.
    void renderOverlay(const cv::Mat& raw_frame, cv::Mat& display_frame);

private:
    void drawGrid(cv::Mat& frame, int step = 100, double alpha = 0.4) const;

    std::unique_ptr<HandTracker> hand_tracker_;
    std::shared_ptr<const std::vector<HandData>> latest_hands_;
    mutable std::mutex hands_mutex_;

    static constexpr size_t kMaxBufferSize = 15; // Buffers ~250ms at 60 FPS

    struct BufferSlot {
        int64_t timestamp = 0;
        cv::Mat frame;
    };

    std::deque<BufferSlot> frame_pool_;
    size_t write_index_ = 0;
    mutable std::mutex buffer_mutex_;

    mutable cv::Mat cached_grid_overlay_;
    mutable cv::Size last_frame_size_;

    bool show_full_skeleton_ = false;
    bool show_debug_overlay_ = false;
    bool show_grid_ = true;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_MEDIATOR_H