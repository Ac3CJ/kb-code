#ifndef CV_KEYBOARD_MEDIATOR_H
#define CV_KEYBOARD_MEDIATOR_H

#include <opencv2/core/mat.hpp>
#include <memory>
#include <mutex>
#include <vector>

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
    std::vector<HandData> latestHands() const;

    /// Check whether the pipeline is initialised and ready.
    bool isInitialised() const;

    /// Overlay display mode toggles
    bool showFullSkeleton() const { return show_full_skeleton_; }
    void setShowFullSkeleton(bool show) { show_full_skeleton_ = show; }
    void toggleFullSkeleton() { show_full_skeleton_ = !show_full_skeleton_; }

    bool showDebugOverlay() const { return show_debug_overlay_; }
    void setShowDebugOverlay(bool show) { show_debug_overlay_ = show; }
    void toggleDebugOverlay() { show_debug_overlay_ = !show_debug_overlay_; }

    /// Render the overlay (hand skeleton + optional debug text) onto the frame.
    void renderOverlay(cv::Mat& frame);

private:
    std::unique_ptr<HandTracker> hand_tracker_;
    std::vector<HandData> latest_hands_;
    mutable std::mutex hands_mutex_;

    bool show_full_skeleton_ = false;
    bool show_debug_overlay_ = false;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_MEDIATOR_H