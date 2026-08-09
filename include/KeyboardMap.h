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
    std::string id;       // e.g., "A", "Enter_Top", "Space"
    float x_u;            // X position in 'u' units (top-left)
    float y_u;            // Y position in 'u' units (top-left)
    float width_u;        // Width in 'u'
    float height_u;       // Height in 'u' (usually 1.0)
    
    // Helper to get physical coordinates in cm
    float x_cm() const { return x_u * kBaseUnitCm; }
    float y_cm() const { return y_u * kBaseUnitCm; }
    float width_cm() const { return width_u * kBaseUnitCm; }
    float height_cm() const { return height_u * kBaseUnitCm; }
};

struct ArucoMarkerDef {
    int id;               // ArUco dictionary ID (0, 1, 2, etc.)
    float x_u;            // X position in 'u' units (center or top-left, let's use top-left)
    float y_u;            // Y position in 'u' units
    float size_u = 1.0f;  // Markers are 1u x 1u
};

class KeyboardMap {
public:
    KeyboardMap();
    ~KeyboardMap();

    /// Populates the map with the standard UK ISO layout
    void loadUKLayout();

    /// Process the frame to find ArUco markers and update the homography matrix.
    /// Takes the frame by reference to avoid copying memory.
    bool updateTransform(const cv::Mat& frame);

    /// Map a camera pixel coordinate to a physical physical coordinate (in cm)
    /// using the active homography matrix.
    cv::Point2f pixelToPhysical(float px, float py) const;
    cv::Point2f physicalToPixel(float x_cm, float y_cm) const;

    /// Returns the key ID at the given physical coordinates (in cm).
    /// Returns an empty string if no key is found at that location.
    std::string getKeyAt(float x_cm, float y_cm) const;

    /// Getters for rendering or homography calculation
    const std::vector<KeyDefinition>& getKeys() const { return keys_; }
    const std::vector<ArucoMarkerDef>& getMarkers() const { return markers_; }
    bool hasValidTransform() const { return !homography_.empty(); }

private:
    std::vector<KeyDefinition> keys_;
    std::vector<ArucoMarkerDef> markers_;
    
    // Internal helper to place a row of keys sequentially
    void addRow(float start_x, float y, const std::vector<std::pair<std::string, float>>& row_data);

    // Vision / Mapping components
    cv::Mat homography_;     // Pixels -> Physical matrix
    cv::Mat inv_homography_; // Physical -> Pixels matrix

    void applyKalmanFilter(cv::Mat& H);
    cv::KalmanFilter kf_;
    bool kf_initialized_ = false;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_KEYBOARD_MAP_H