#ifndef CV_KEYBOARD_DEBUG_VISUALIZER_H
#define CV_KEYBOARD_DEBUG_VISUALIZER_H

#include <opencv2/core.hpp>

namespace cv_keyboard {

// Moved from Mediator to be shared cleanly
enum class FilterMode { NONE, SOBEL, LAPLACIAN, CANNY, BLACKHAT, FRAMEDIFF, LAB_LIGHTNESS, HEATMAP };

class DebugVisualizer {
public:
    DebugVisualizer() = default;
    ~DebugVisualizer() = default;

    // Evaluates the mode and applies the corresponding filter
    void showFilters(const cv::Mat& current_frame, FilterMode mode);

private:
    cv::Mat gray_, prev_gray_, temp_16s_, display_out_;

    const std::string window_name_ = "Debug Filter";

    void applySobel();
    void applyLaplacian();
    void applyCanny();
    void applyBlackHat();
    void applyFrameDiff();
    void applyLabLightness(const cv::Mat& color_frame);
    void applyHeatmap();
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_DEBUG_VISUALIZER_H