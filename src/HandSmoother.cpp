#include "HandSmoother.h"

namespace cv_keyboard {

HandSmoother::HandSmoother(float pos_noise, float vel_noise)
    : pos_noise_(pos_noise), vel_noise_(vel_noise) {}

void HandSmoother::reset() {
    initialized_ = false;
    frames_unseen = 0;
    prev_pts_.clear();
}

void HandSmoother::smooth(std::array<Landmark, 21>& landmarks, const cv::Mat& prev_gray, const cv::Mat& curr_gray) {
    float width = static_cast<float>(curr_gray.cols);
    float height = static_cast<float>(curr_gray.rows);

    if (!initialized_) {
        prev_pts_.resize(21);
        for (int i = 0; i < 21; ++i) {
            kfs_[i].init(4, 4, 0); 
            kfs_[i].transitionMatrix = (cv::Mat_<float>(4, 4) <<
                1, 0, 1, 0,
                0, 1, 0, 1,
                0, 0, 1, 0,
                0, 0, 0, 1);
            
            cv::setIdentity(kfs_[i].measurementMatrix);
            cv::setIdentity(kfs_[i].processNoiseCov, cv::Scalar::all(1e-5));
            
            kfs_[i].measurementNoiseCov = (cv::Mat_<float>(4, 4) <<
                pos_noise_, 0, 0, 0,
                0, pos_noise_, 0, 0,
                0, 0, vel_noise_, 0,
                0, 0, 0, vel_noise_);
            
            cv::setIdentity(kfs_[i].errorCovPost, cv::Scalar::all(1));

            kfs_[i].statePost.at<float>(0) = landmarks[i].x;
            kfs_[i].statePost.at<float>(1) = landmarks[i].y;
            kfs_[i].statePost.at<float>(2) = 0.0f;
            kfs_[i].statePost.at<float>(3) = 0.0f;

            prev_pts_[i] = cv::Point2f(landmarks[i].x * width, landmarks[i].y * height);
        }
        initialized_ = true;
        return; 
    }

    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;
    
    cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts_, curr_pts, status, err, cv::Size(21, 21), 3);

    cv::Mat measurement(4, 1, CV_32F);
    
    for (int i = 0; i < 21; ++i) {
        kfs_[i].predict();

        float vx_of = 0.0f;
        float vy_of = 0.0f;

        if (status[i]) {
            vx_of = (curr_pts[i].x - prev_pts_[i].x) / width;
            vy_of = (curr_pts[i].y - prev_pts_[i].y) / height;
        } else {
            vx_of = kfs_[i].statePre.at<float>(2);
            vy_of = kfs_[i].statePre.at<float>(3);
        }

        measurement.at<float>(0) = landmarks[i].x;
        measurement.at<float>(1) = landmarks[i].y;
        measurement.at<float>(2) = vx_of;
        measurement.at<float>(3) = vy_of;

        cv::Mat estimated = kfs_[i].correct(measurement);
        
        landmarks[i].x = estimated.at<float>(0);
        landmarks[i].y = estimated.at<float>(1);
        landmarks[i].vx = measurement.at<float>(2);
        landmarks[i].vy = measurement.at<float>(3);

        prev_pts_[i] = cv::Point2f(landmarks[i].x * width, landmarks[i].y * height);
    }
    frames_unseen = 0;
}

} // namespace cv_keyboard