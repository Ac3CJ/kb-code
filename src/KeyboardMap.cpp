#include "KeyboardMap.h"

#include <iostream>

namespace cv_keyboard {

KeyboardMap::KeyboardMap() {
    aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    aruco_params_ = cv::aruco::DetectorParameters::create();

    kf_.init(9, 9, 0);
    cv::setIdentity(kf_.transitionMatrix);
    cv::setIdentity(kf_.measurementMatrix);

    // Tweak these to change how much "smoothing" vs "responsiveness" you want.
    // Smaller processNoiseCov = smoother but more lag.
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
}

KeyboardMap::~KeyboardMap() = default;

void KeyboardMap::applyKalmanFilter(cv::Mat& H) {
    // 1. Normalize the homography matrix (Scale-invariant)
    H /= H.at<double>(2, 2);
    
    // 2. Flatten the 3x3 double matrix into a 9x1 float measurement vector
    cv::Mat measurement = cv::Mat::zeros(9, 1, CV_32F);
    int k = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            measurement.at<float>(k++) = static_cast<float>(H.at<double>(i, j));
        }
    }

    // 3. If it's the very first frame, snap the filter directly to the measurement
    if (!kf_initialized_) {
        kf_.statePost = measurement.clone();
        kf_initialized_ = true;
    }

    // 4. Run the prediction and correction steps
    kf_.predict();
    cv::Mat estimated = kf_.correct(measurement);

    // 5. Rebuild the smoothed 9x1 vector back into our 3x3 double matrix
    k = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            H.at<double>(i, j) = static_cast<double>(estimated.at<float>(k++));
        }
    }
}

bool KeyboardMap::updateTransform(const cv::Mat& frame) {
    // 1. Lazy-Initialize a "Dummy" Camera Matrix based on frame resolution
    if (camera_matrix_.empty()) {
        double focal_length = frame.cols; // Reasonable guess for most webcams
        double center_x = frame.cols / 2.0;
        double center_y = frame.rows / 2.0;
        
        // Dummy
        // camera_matrix_ = (cv::Mat_<double>(3, 3) << 
        //     focal_length, 0, center_x,
        //     0, focal_length, center_y,
        //     0, 0, 1);

        // First Pass
        // camera_matrix_ = (cv::Mat_<double>(3, 3) << 
        //     3401.30496, 0.00000, 2032.81880,
        //     0.00000, 3403.65091, 1114.93330,
        //     0.00000, 0.00000, 1.00000);

        camera_matrix_ = (cv::Mat_<double>(3, 3) << 
            1419.58704, 0.00000, 958.93454,
            0.00000, 1420.44749, 537.41207,
            0.00000, 0.00000, 1.00000);
        
        // dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F); // Assume no lens distortion for now

        // First Pass
        // dist_coeffs_ = (cv::Mat_<double>(5, 1) << 
        //     0.37974, -2.23392, -0.00405, -0.00154, 3.27364);

        dist_coeffs_ = (cv::Mat_<double>(5, 1) << 
            0.38305, -2.50514, 0.00176, 0.00058, 4.69315);
    }

    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners, rejected;

    // 2. Detect all visible markers
    cv::aruco::detectMarkers(frame, aruco_dict_, marker_corners, marker_ids, aruco_params_, rejected);

    if (marker_ids.empty() || !aruco_board_) {
        valid_pose_ = false;
        return false;
    }

    // 3. Estimate 3D Pose using the rigid board
    int markers_used = cv::aruco::estimatePoseBoard(
        marker_corners, marker_ids, aruco_board_, 
        camera_matrix_, dist_coeffs_, rvec_, tvec_
    );

    if (markers_used > 0) {
        valid_pose_ = true;

        // 4. Derive the Inverse Homography Matrix for finger tap detection
        // Convert the rotation vector to a 3x3 rotation matrix
        cv::Mat R;
        cv::Rodrigues(rvec_, R);
        
        // H = CameraMatrix * [R_col0 | R_col1 | tvec]
        cv::Mat r1 = R.col(0);
        cv::Mat r2 = R.col(1);
        cv::Mat t = (cv::Mat_<double>(3, 1) << tvec_[0], tvec_[1], tvec_[2]);
        
        cv::Mat Rt;
        std::vector<cv::Mat> cols = {r1, r2, t};
        cv::hconcat(cols, Rt);
        
        cv::Mat homography = camera_matrix_ * Rt;
        inv_homography_ = homography.inv();

        return true;
    }

    valid_pose_ = false;
    return false;
}

cv::Point2f KeyboardMap::pixelToPhysical(float px, float py) const {
    if (!valid_pose_ || inv_homography_.empty()) {
        return cv::Point2f(-1.0f, -1.0f);
    }

    std::vector<cv::Point2f> src_point = { cv::Point2f(px, py) };
    std::vector<cv::Point2f> dst_point;
    
    // Project the camera pixel straight down onto the physical paper map
    cv::perspectiveTransform(src_point, dst_point, inv_homography_);
    return dst_point[0];
}

cv::Point2f KeyboardMap::physicalToPixel(float x_cm, float y_cm) const {
    if (!valid_pose_) {
        return cv::Point2f(-1.0f, -1.0f);
    }

    // Define the physical point in 3D space (Z is 0 because the paper is flat)
    std::vector<cv::Point3f> object_points = { cv::Point3f(x_cm, y_cm, 0.0f) };
    std::vector<cv::Point2f> image_points;
    
    // Use the camera parameters to project the 3D point onto the 2D camera sensor
    cv::projectPoints(object_points, rvec_, tvec_, camera_matrix_, dist_coeffs_, image_points);
    return image_points[0];
}

void KeyboardMap::addRow(float start_x, float y, const std::vector<std::pair<std::string, float>>& row_data) {
    float current_x = start_x;
    for (const auto& key : row_data) {
        keys_.push_back({
            key.first, 
            current_x, 
            y, 
            key.second, // width
            1.0f        // height is always 1u for these keys
        });
        current_x += key.second; // Move cursor right by the width of the key
    }
}

void KeyboardMap::loadUKLayout() {
    keys_.clear();
    markers_.clear();

    // Define the keys
    
    // Row 1: Numbers (Y = 0.0)
    addRow(0.0f, 0.0f, {
        {"~", 1.0f}, {"1", 1.0f}, {"2", 1.0f}, {"3", 1.0f}, {"4", 1.0f}, 
        {"5", 1.0f}, {"6", 1.0f}, {"7", 1.0f}, {"8", 1.0f}, {"9", 1.0f}, 
        {"0", 1.0f}, {"-", 1.0f}, {"=", 1.0f}, {"Backspace", 2.0f}
    }); // Total width: 15u

    // Row 2: QWERTY (Y = 1.0)
    addRow(0.0f, 1.0f, {
        {"Tab", 1.5f}, {"Q", 1.0f}, {"W", 1.0f}, {"E", 1.0f}, {"R", 1.0f}, 
        {"T", 1.0f}, {"Y", 1.0f}, {"U", 1.0f}, {"I", 1.0f}, {"O", 1.0f}, 
        {"P", 1.0f}, {"[", 1.0f}, {"]", 1.0f}, {"Enter", 1.5f} // Top half of Enter
    }); // Total width: 15u

    // Row 3: ASDF (Y = 2.0)
    addRow(0.0f, 2.0f, {
        {"Caps", 1.75f}, {"A", 1.0f}, {"S", 1.0f}, {"D", 1.0f}, {"F", 1.0f}, 
        {"G", 1.0f}, {"H", 1.0f}, {"J", 1.0f}, {"K", 1.0f}, {"L", 1.0f}, 
        {";", 1.0f}, {"'", 1.0f}, {"#", 1.0f}, {"Enter", 1.25f} // Bottom half of Enter
    }); // Total width: 15u

    // Row 4: ZXCV (Y = 3.0)
    addRow(0.0f, 3.0f, {
        {"L_Shift", 1.25f}, {"\\", 1.0f}, {"Z", 1.0f}, {"X", 1.0f}, {"C", 1.0f}, 
        {"V", 1.0f}, {"B", 1.0f}, {"N", 1.0f}, {"M", 1.0f}, {",", 1.0f}, 
        {".", 1.0f}, {"/", 1.0f}, {"R_Shift", 2.75f}
    }); // Total width: 15u

    // Row 5: Modifiers (Y = 4.0)
    addRow(0.0f, 4.0f, {
        {"L_Ctrl", 1.25f}, {"L_Win", 1.25f}, {"L_Alt", 1.25f}, {"Space", 6.25f}, 
        {"R_AltGr", 1.25f}, {"Fn", 1.25f}, {"Option", 1.25f}, {"R_Ctrl", 1.25f}
    }); // Total width: 15u

    // Define the ArUco Markers
    
    // Map dimensions and spacing into 'u' units
    float padding_u = 0.05f / kBaseUnitCm; // ARUCO_PADDING_CM converted to 'u'
    float marker_size_u = 1.0f;            // ARUCO_SIZE_CM is equal to 1u
    float board_w_u = 15.0f;
    float board_h_u = 5.0f;

    // Outer bounding box for the markers (top-left coordinates of the corners)
    float left_x = -marker_size_u - padding_u;
    float right_x = board_w_u + padding_u;
    float top_y = -marker_size_u - padding_u;
    float bottom_y = board_h_u + padding_u;

    // Calculate step sizes precisely as the Python script did
    float step_x = (right_x - left_x) / 8.0f;
    
    // The Python script calculated step_y based on the distance between the bottom 
    // of the top row and the bottom of the bottom row. 
    float py_top_bottom_edge = -padding_u;
    float py_bot_bottom_edge = board_h_u + padding_u + marker_size_u;
    float step_y = (py_bot_bottom_edge - py_top_bottom_edge) / 3.0f;

    int marker_id = 0;

    // 1st Row
    for (int i = 0; i < 9; ++i) {
        markers_.push_back({marker_id++, left_x + (i * step_x), top_y});
    }

    // 2nd Row
    markers_.push_back({marker_id++, left_x, top_y + step_y});
    markers_.push_back({marker_id++, right_x, top_y + step_y});

    // 3rd Row
    markers_.push_back({marker_id++, left_x, top_y + (2.0f * step_y)});
    markers_.push_back({marker_id++, right_x, top_y + (2.0f * step_y)});

    // 4th Row
    for (int i = 0; i < 9; ++i) {
        markers_.push_back({marker_id++, left_x + (i * step_x), bottom_y});
    }

    // Building ArUco Board
    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<int> ids;

    for (const auto& marker : markers_) {
        float x = marker.x_u * kBaseUnitCm;
        float y = marker.y_u * kBaseUnitCm;
        float size = marker.size_u * kBaseUnitCm;

        // Add 3D coordinates for the 4 corners of this specific marker (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        obj_points.push_back({
            cv::Point3f(x, y, 0.0f),
            cv::Point3f(x + size, y, 0.0f),
            cv::Point3f(x + size, y + size, 0.0f),
            cv::Point3f(x, y + size, 0.0f)
        });
        ids.push_back(marker.id);
    }

    aruco_board_ = cv::aruco::Board::create(obj_points, aruco_dict_, ids);
}

std::string KeyboardMap::getKeyAt(float x_cm, float y_cm) const {
    for (const auto& key : keys_) {
        float kx = key.x_cm();
        float ky = key.y_cm();
        float kw = key.width_cm();
        float kh = key.height_cm();

        if (x_cm >= kx && x_cm <= (kx + kw) &&
            y_cm >= ky && y_cm <= (ky + kh)) {
            return key.id;
        }
    }
    return ""; 
}

} // namespace cv_keyboard