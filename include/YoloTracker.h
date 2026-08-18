#ifndef CV_KEYBOARD_YOLO_TRACKER_H
#define CV_KEYBOARD_YOLO_TRACKER_H

#include "IHandTracker.h"
#include <opencv2/dnn.hpp>
#include <mutex>

namespace cv_keyboard {

class YoloTracker : public IHandTracker {
public:
    YoloTracker();
    ~YoloTracker() override = default;

    bool init() override;
    std::shared_ptr<const std::vector<HandData>> detect(const cv::Mat& frame, int64_t timestamp_us) override;
    
    int64_t latestTimestamp() const override { return latest_timestamp_us_; }
    bool isInitialised() const override { return initialised_; }
    
    // Performance metrics
    double getTrackerTimeMs() const override { return tracker_time_ms_; }
    double getFusionTimeMs() const override { return fusion_time_ms_; }

private:
    cv::dnn::Net net_;
    bool initialised_ = false;
    
    int64_t latest_timestamp_us_ = 0;
    double tracker_time_ms_ = 0.0;
    double fusion_time_ms_ = 0.0; // Keep at zero until KF is added

    std::shared_ptr<std::vector<HandData>> cached_hands_ = std::make_shared<std::vector<HandData>>();
    std::mutex mutex_;

    // YOLO Configuration
    const float CONFIDENCE_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.4f;
    const cv::Size INPUT_SIZE = cv::Size(640, 640);
    const int NUM_KEYPOINTS = 21;
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_YOLO_TRACKER_H