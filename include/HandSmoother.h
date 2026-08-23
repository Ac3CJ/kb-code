#ifndef CV_KEYBOARD_HAND_SMOOTHER_H
#define CV_KEYBOARD_HAND_SMOOTHER_H

#include "IHandTracker.h"
#include <opencv2/video/tracking.hpp>
#include <array>
#include <vector>

namespace cv_keyboard {

class HandSmoother {
public:
    // Pass custom covariance noise coefficients depending on the tracker's jitter
    HandSmoother(float pos_noise = 1e-5f, float vel_noise = 1e-4f);
    ~HandSmoother() = default;

    void reset();
    void smooth(std::array<Landmark, 21>& landmarks, const cv::Mat& prev_gray, const cv::Mat& curr_gray);

    int frames_unseen = 0;

private:
    std::array<cv::KalmanFilter, 21> kfs_;
    std::vector<cv::Point2f> prev_pts_;
    bool initialized_ = false;
    
    float pos_noise_;
    float vel_noise_;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_HAND_SMOOTHER_H