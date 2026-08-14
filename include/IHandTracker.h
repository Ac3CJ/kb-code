#ifndef CV_KEYBOARD_IHAND_TRACKER_H
#define CV_KEYBOARD_IHAND_TRACKER_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <array>

namespace cv_keyboard {

/// Single landmark point based on MediaPipe hand detection
struct Landmark {
    float x = 0.0f;       ///< Normalised x-coordinate [0, 1] in image space
    float y = 0.0f;       ///< Normalised y-coordinate [0, 1] in image space
    float z = 0.0f;       ///< Relative depth (wrist-relative, not metric)
    float confidence = 0.0f;
};

/// Complete set of 21 landmarks for one detected hand
struct HandData {
    std::array<Landmark, 21> landmarks;

    float hand_confidence = 0.0f;

    int handedness = 0;
    int64_t timestamp_us = 0;

    HandData() = default;
};

/// Hand landmark indices for finger tips
enum HandLandmark : int {
    WRIST            = 0,
    THUMB_CMC        = 1,
    THUMB_MCP        = 2,
    THUMB_IP         = 3,
    THUMB_TIP        = 4,
    INDEX_FINGER_MCP = 5,
    INDEX_FINGER_PIP = 6,
    INDEX_FINGER_DIP = 7,
    INDEX_FINGER_TIP = 8,
    MIDDLE_FINGER_MCP = 9,
    MIDDLE_FINGER_PIP = 10,
    MIDDLE_FINGER_DIP = 11,
    MIDDLE_FINGER_TIP = 12,
    RING_FINGER_MCP  = 13,
    RING_FINGER_PIP  = 14,
    RING_FINGER_DIP  = 15,
    RING_FINGER_TIP  = 16,
    PINKY_MCP        = 17,
    PINKY_PIP        = 18,
    PINKY_DIP        = 19,
    PINKY_TIP        = 20,
};

/// Finger tip landmark indices (5 per hand = 10 for two hands)
inline constexpr int FINGER_TIP_INDICES[] = {
    THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP
};

/// MediaPipe hand skeleton connection pairs (indices into landmark array)
inline constexpr int HAND_CONNECTIONS[][2] = {
    {0, 1},   {1, 2},   {2, 3},   {3, 4},           // Thumb
    {0, 5},   {5, 6},   {6, 7},   {7, 8},           // Index
    {0, 9},   {9, 10},  {10, 11}, {11, 12},         // Middle
    {0, 13},  {13, 14}, {14, 15}, {15, 16},         // Ring
    {0, 17},  {17, 18}, {18, 19}, {19, 20},         // Pinky
    {5, 9},   {9, 13},  {13, 17}                    // Palm
};

inline constexpr int NUM_HAND_CONNECTIONS = sizeof(HAND_CONNECTIONS) / sizeof(HAND_CONNECTIONS[0]);

/// Wraps hand landmark inference.
class IHandTracker {
public:
    virtual ~IHandTracker() = default;

    virtual bool init() = 0;
    virtual std::shared_ptr<const std::vector<HandData>> detect(const cv::Mat& frame, int64_t timestamp_us) = 0;
    virtual int64_t latestTimestamp() const = 0;
    virtual bool isInitialised() const = 0;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_IHAND_TRACKER_H