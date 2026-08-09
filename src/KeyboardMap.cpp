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
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-3));
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
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners, rejected_candidates;

    // Detect all visible markers in the frame
    cv::aruco::detectMarkers(frame, aruco_dict_, marker_corners, marker_ids, aruco_params_, rejected_candidates);

    if (marker_ids.empty()) {
        // std::cout << "[KeyboardMap] No ArUco markers detected in the current frame.\n";
        return false; // No markers found to map
    }

    // std::cout << "[KeyboardMap] Detected " << marker_ids.size() << " ArUco markers.\n";

    // TODO (Future Optimization): Compare the centers of 'marker_corners' to the previous frame's 
    // corners here. If the deviation is < threshold, return early to save compute time.

    std::vector<cv::Point2f> src_pixel_points;
    std::vector<cv::Point2f> dst_physical_points;

    for (size_t i = 0; i < marker_ids.size(); ++i) {
        int id = marker_ids[i];
        
        // Find if this detected marker belongs to our virtual keyboard layout
        auto it = std::find_if(markers_.begin(), markers_.end(), [id](const ArucoMarkerDef& m) {
            return m.id == id;
        });

        if (it != markers_.end()) {
            const auto& pixels = marker_corners[i];
            
            // Calculate the 4 physical corners of this marker in cm
            float x_cm = it->x_u * kBaseUnitCm;
            float y_cm = it->y_u * kBaseUnitCm;
            float size_cm = it->size_u * kBaseUnitCm;

            // ArUco returns corners clockwise starting from top-left.
            // Push the Pixels
            src_pixel_points.push_back(pixels[0]); // Top-Left
            src_pixel_points.push_back(pixels[1]); // Top-Right
            src_pixel_points.push_back(pixels[2]); // Bottom-Right
            src_pixel_points.push_back(pixels[3]); // Bottom-Left

            // Push the corresponding Physical Coordinates (cm)
            dst_physical_points.push_back(cv::Point2f(x_cm, y_cm));                             // Top-Left
            dst_physical_points.push_back(cv::Point2f(x_cm + size_cm, y_cm));                   // Top-Right
            dst_physical_points.push_back(cv::Point2f(x_cm + size_cm, y_cm + size_cm));         // Bottom-Right
            dst_physical_points.push_back(cv::Point2f(x_cm, y_cm + size_cm));                   // Bottom-Left
        }
    }

    // OpenCV requires a minimum of 4 points (1 full marker) to calculate homography
    if (src_pixel_points.size() >= 4) {
        // Pixels -> Physical (For touch detection)
        homography_ = cv::findHomography(src_pixel_points, dst_physical_points, cv::RANSAC);
        
        if (!homography_.empty()) {
            applyKalmanFilter(homography_);
            inv_homography_ = homography_.inv();
            return true;
        }
    }

    return false;
}

cv::Point2f KeyboardMap::pixelToPhysical(float px, float py) const {
    if (homography_.empty()) {
        return cv::Point2f(-1.0f, -1.0f);
    }

    std::vector<cv::Point2f> src_point = { cv::Point2f(px, py) };
    std::vector<cv::Point2f> dst_point;
    
    // Apply the 3x3 matrix to shift the pixel into the flat physical 'cm' plane
    cv::perspectiveTransform(src_point, dst_point, homography_);
    
    return dst_point[0];
}

cv::Point2f KeyboardMap::physicalToPixel(float x_cm, float y_cm) const {
    if (inv_homography_.empty()) {
        return cv::Point2f(-1.0f, -1.0f);
    }

    std::vector<cv::Point2f> src_point = { cv::Point2f(x_cm, y_cm) };
    std::vector<cv::Point2f> dst_point;
    
    // Apply the inverse 3x3 matrix to shift the physical cm coordinate into pixel space
    cv::perspectiveTransform(src_point, dst_point, inv_homography_);
    
    return dst_point[0];
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

    // ---------------------------------------------------------
    // 1. Define Keys (Origin 0,0 is top-left of the first key)
    // ---------------------------------------------------------
    
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


    // ---------------------------------------------------------
    // 2. Define ArUco Markers
    // ---------------------------------------------------------
    // Note: These coordinates estimate their placement based on the image provided.
    // They are placed outside the 15x5u main key grid.
    
    // Top Row (above the keys)
    markers_.push_back({0,  0.0f, -1.0f}); // Above '~' (x=0)
    markers_.push_back({6,  4.0f, -1.0f}); // Above '4' (x=4)
    markers_.push_back({8,  8.0f, -1.0f}); // Above '8' (x=8)
    markers_.push_back({1, 14.0f, -1.0f}); // Above right-edge of Backspace (x=14)

    // Middle Row (Sides)
    markers_.push_back({4, -1.0f, 2.0f});  // Left of Caps Lock (x=-1)
    markers_.push_back({5, 15.0f, 2.0f});  // Right of Enter (x=15)

    // Bottom Row (below the keys)
    markers_.push_back({2,  0.0f, 5.0f});  // Below left Ctrl (x=0)
    markers_.push_back({7,  4.0f, 5.0f});  // Below left side of Space (x=4)
    markers_.push_back({9,  8.0f, 5.0f});  // Below right side of Space (x=8)
    markers_.push_back({3, 14.0f, 5.0f});  // Below right Ctrl (x=14)
}

std::string KeyboardMap::getKeyAt(float x_cm, float y_cm) const {
    // Simple AABB (Axis-Aligned Bounding Box) collision check
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
    return ""; // No key found
}

} // namespace cv_keyboard