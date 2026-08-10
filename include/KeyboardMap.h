#ifndef CV_KEYBOARD_KEYBOARD_MAP_H
#define CV_KEYBOARD_KEYBOARD_MAP_H

#include <string>
#include <vector>
#include <unordered_map>

// OpenCV includes for Homography and ArUco
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/video/tracking.hpp>

namespace cv_keyboard {

/// Base physical unit in centimeters
static constexpr float kBaseUnitCm = 1.7f;

struct KeyDefinition {
    std::string id;
    float x_u;
    float y_u;
    float width_u;
    float height_u;
    
    // Helper to get physical coordinates in cm
    float x_cm() const { return x_u * kBaseUnitCm; }
    float y_cm() const { return y_u * kBaseUnitCm; }
    float width_cm() const { return width_u * kBaseUnitCm; }
    float height_cm() const { return height_u * kBaseUnitCm; }
};

struct ArucoMarkerDef {
    int id;
    float x_u;
    float y_u;
    float size_u = 1.0f;
};

class KeyboardMap {
public:
    KeyboardMap();
    ~KeyboardMap();

    void loadUKLayout();
    bool updateTransform(const cv::Mat& frame);


    cv::Point2f pixelToPhysical(float px, float py) const;
    cv::Point2f physicalToPixel(float x_cm, float y_cm) const;
    std::string getKeyAt(float x_cm, float y_cm) const;

    /// Getters for rendering or homography calculation
    const std::vector<KeyDefinition>& getKeys() const { return keys_; }
    const std::vector<ArucoMarkerDef>& getMarkers() const { return markers_; }
    bool hasValidTransform() const { return valid_pose_; }

private:
    std::vector<KeyDefinition> keys_;
    std::vector<ArucoMarkerDef> markers_;
    
    // Internal helper to place a row of keys sequentially
    void addRow(float start_x, float y, const std::vector<std::pair<std::string, float>>& row_data);

    // Vision / Mapping components
    cv::Mat homography_;     // Pixels -> Physical matrix
    cv::Mat inv_homography_; // Physical -> Pixels matrix

    // Kalman filter for smoothing the homography
    void applyKalmanFilter(cv::Mat& H);
    cv::KalmanFilter kf_;
    bool kf_initialized_ = false;

    // Aruco detection parameters
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
    cv::Ptr<cv::aruco::Board> aruco_board_;

    // Camera Intrinsics
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    // 3D Pose
    cv::Vec3d rvec_, tvec_;
    bool valid_pose_ = false;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_KEYBOARD_MAP_H