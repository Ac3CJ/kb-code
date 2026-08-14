#ifndef CV_KEYBOARD_MEDIAPIPE_TRACKER_H
#define CV_KEYBOARD_MEDIAPIPE_TRACKER_H

#include "IHandTracker.h"

namespace cv_keyboard {

class MediaPipeTracker : public IHandTracker {
public:
    MediaPipeTracker();
    ~MediaPipeTracker() override;

    bool init() override; 
    std::shared_ptr<const std::vector<HandData>> detect(const cv::Mat& frame, int64_t timestamp_us) override; 
    int64_t latestTimestamp() const override; 
    bool isInitialised() const override; 

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace cv_keyboard

#endif // CV_KEYBOARD_MEDIAPIPE_TRACKER_H