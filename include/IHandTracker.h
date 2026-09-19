#ifndef CV_KEYBOARD_IHAND_TRACKER_H
#define CV_KEYBOARD_IHAND_TRACKER_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

#include "HandData.h"

namespace cv_keyboard {

/// Wraps hand landmark inference.
class IHandTracker {
public:
    virtual ~IHandTracker() = default;

    virtual bool init() = 0;
    virtual std::shared_ptr<const std::vector<HandData>> detect(const cv::Mat& frame, int64_t timestamp_us) = 0;
    virtual int64_t latestTimestamp() const = 0;
    virtual bool isInitialised() const = 0;

    virtual double getTrackerTimeMs() const = 0;
    virtual double getFusionTimeMs() const = 0;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_IHAND_TRACKER_H