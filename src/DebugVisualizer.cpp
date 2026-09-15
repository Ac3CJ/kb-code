#include "DebugVisualizer.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace cv_keyboard {

void DebugVisualizer::showFilters(const cv::Mat& current_frame, FilterMode mode) {
    if (current_frame.empty()) return;

    // Turn off and close window if mode is NONE
    if (mode == FilterMode::NONE) {
        try { cv::destroyWindow(window_name_); } catch (...) {}
        return;
    }

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::cvtColor(current_frame, gray_, cv::COLOR_BGR2GRAY);

    switch (mode) {
        case FilterMode::SOBEL: applySobel(); break;
        case FilterMode::LAPLACIAN: applyLaplacian(); break;
        case FilterMode::CANNY: applyCanny(); break;
        case FilterMode::BLACKHAT: applyBlackHat(); break;
        case FilterMode::FRAMEDIFF: applyFrameDiff(); break;
        case FilterMode::LAB_LIGHTNESS: applyLabLightness(current_frame); break;
        case FilterMode::MOTION_LIGHTNESS: applyMotionLightness(current_frame); break;
        case FilterMode::MOTION_HEATMAP: applyMotionHeatmap(current_frame); break;
        case FilterMode::HEATMAP: applyHeatmap(); break; // NEW
        default: break;
    }

    gray_.copyTo(prev_gray_);
    current_frame.copyTo(prev_color_); // feed the motion filters (FrameDiff, Motion Lightness)
}

void DebugVisualizer::applySobel() {
    cv::Sobel(gray_, temp_16s_, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(temp_16s_, display_out_);

    cv::setWindowTitle(window_name_, "Sobel Filter");
    cv::imshow(window_name_, display_out_);
}

void DebugVisualizer::applyLaplacian() {
    cv::Laplacian(gray_, temp_16s_, CV_16S, 3);
    cv::convertScaleAbs(temp_16s_, display_out_);

    cv::setWindowTitle(window_name_, "Laplacian Filter");
    cv::imshow(window_name_, display_out_);
}

void DebugVisualizer::applyCanny() {
    cv::Canny(gray_, display_out_, 50, 150);

    cv::setWindowTitle(window_name_, "Canny Filter");
    cv::imshow(window_name_, display_out_);
}

void DebugVisualizer::applyBlackHat() {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(gray_, display_out_, cv::MORPH_BLACKHAT, kernel);

    cv::setWindowTitle(window_name_, "BlackHat Filter");
    cv::imshow(window_name_, display_out_);
}

void DebugVisualizer::applyFrameDiff() {
    if (prev_gray_.empty()) return;
    cv::absdiff(gray_, prev_gray_, display_out_);
    cv::threshold(display_out_, display_out_, 10, 255, cv::THRESH_BINARY);

    cv::setWindowTitle(window_name_, "Frame Difference Filter");
    cv::imshow(window_name_, display_out_);
}

void DebugVisualizer::applyLabLightness(const cv::Mat& color_frame) {
    cv::Mat lab;
    std::vector<cv::Mat> channels;
    cv::cvtColor(color_frame, lab, cv::COLOR_BGR2Lab);
    cv::split(lab, channels);

    cv::setWindowTitle(window_name_, "Lab Lightness Channel");
    cv::imshow(window_name_, channels[0]); // L channel
}

void DebugVisualizer::applyHeatmap() {
    cv::Mat inverted, heatmap;
    
    // Invert the grayscale image. 
    // Now, dark shadows = 255 (Max), Bright desk = 0 (Min)
    cv::bitwise_not(gray_, inverted);

    // Apply a thermal colormap. 
    // COLORMAP_JET maps low values to blue, and high values to red.
    // Inverted image, deep shadows will appear as bright red.
    cv::applyColorMap(inverted, heatmap, cv::COLORMAP_JET);

    cv::setWindowTitle(window_name_, "Shadow Intensity Heatmap");
    cv::imshow(window_name_, heatmap);
}

void DebugVisualizer::applyMotionLightness(const cv::Mat& color_frame) {
    if (prev_color_.empty()) return;

    // Stage 1 — Frame difference: isolate motion between consecutive frames.
    // Anything static (desk, keyboard, resting shadows) cancels out.
    cv::Mat diff;
    cv::absdiff(color_frame, prev_color_, diff);

    // Stage 2 — Lab lightness: reduce the motion residue to the L channel,
    // so intensity-only changes (moving finger shadow, cast shadows) remain
    // visible while chroma noise is suppressed.
    cv::Mat lab;
    std::vector<cv::Mat> channels;
    
    cv::cvtColor(diff, lab, cv::COLOR_BGR2Lab);
    cv::split(lab, channels);

    // Noise floor: zero out low-level sensor noise but keep the graduated
    // lightness values above it (THRESH_TOZERO preserves intensity levels,
    // unlike the binary threshold used by the plain FRAMEDIFF filter).
    cv::threshold(channels[0], display_out_, 5, 255, cv::THRESH_TOZERO);

    cv::setWindowTitle(window_name_, "Motion Lightness (Frame Diff -> Lab L)");
    cv::imshow(window_name_, display_out_);
}

void DebugVisualizer::applyMotionHeatmap(const cv::Mat& color_frame) {
    if (prev_color_.empty()) return;

    // Stage 1 — Frame difference: isolate motion between consecutive frames.
    cv::Mat diff, motion;
    cv::absdiff(color_frame, prev_color_, diff);
    cv::cvtColor(diff, motion, cv::COLOR_BGR2GRAY);

    // Noise floor: zero out sensor noise while keeping graduated motion
    // intensity (same floor as the Motion Lightness filter).
    cv::threshold(motion, motion, 10, 255, cv::THRESH_TOZERO);

    // Stage 2 — Heatmap: JET maps bright to red, so moving segments appear
    // hot. No bitwise_not here (unlike the static shadow heatmap): motion is
    // already bright in the diff, inverting would flood the frame with red.
    cv::Mat heatmap;
    cv::applyColorMap(motion, heatmap, cv::COLORMAP_JET);

    cv::setWindowTitle(window_name_, "Motion Heatmap (Frame Diff -> JET)");
    cv::imshow(window_name_, heatmap);
}

} // namespace cv_keyboard