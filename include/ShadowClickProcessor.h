#ifndef CV_KEYBOARD_SHADOW_CLICK_PROCESSOR_H
#define CV_KEYBOARD_SHADOW_CLICK_PROCESSOR_H

#include "BaseClickProcessor.h"
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <map>
#include <string>

namespace cv_keyboard {

class ShadowClickProcessor : public BaseClickProcessor {
public:
    ShadowClickProcessor() = default;
    ~ShadowClickProcessor() override = default;

    void setFrame(const cv::Mat& bgr_frame) { current_bgr_ = bgr_frame; }

protected:
    void detectClicks(const std::vector<HandData>& hands, 
                      const KeyboardMap& keyboard_map, 
                      int frame_width, 
                      int frame_height) override;

    void resetHistory() override;

private:
    cv::Mat current_bgr_;
    cv::Mat prev_gray_;
    
    enum class StrikeState { IDLE, CONVERGING };
    std::map<int, StrikeState> finger_states_;

    // Configurable ROI sizing and positioning
    const int ROI_WIDTH = 150;
    const int ROI_HEIGHT = 75;
    const int ROI_OFFSET_X = 0;  
    const int ROI_OFFSET_Y = 20;  

    // Configurable filter thresholds
    const int SHADOW_L_THRESH = 100; 
    const int MOTION_THRESH = 10;    
    const int BLUR_SIZE = 31; // NEW: Gaussian blur kernel size (must be odd)
    
    // Debug Visualizer variables
    bool show_debug_window_ = true; 
    void drawDebugWindow(const cv::Mat& roi_bgr, const cv::Mat& diff_mask, const cv::Mat& combined_mask, const cv::Mat& flow, const cv::Point2f& true_center);
};

} // namespace cv_keyboard
#endif