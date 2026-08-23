#ifndef CV_KEYBOARD_MEDIAPIPE_TRACKER_H
#define CV_KEYBOARD_MEDIAPIPE_TRACKER_H

#include "IHandTracker.h"
#include "HandSmoother.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/gpu/gpu_buffer.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <array>

namespace cv_keyboard {

class MediaPipeTracker : public IHandTracker {
public:
    MediaPipeTracker();
    ~MediaPipeTracker() override;

    bool init() override; 
    std::shared_ptr<const std::vector<HandData>> detect(const cv::Mat& frame, int64_t timestamp_us) override; 
    int64_t latestTimestamp() const override; 
    bool isInitialised() const override; 

    double getTrackerTimeMs() const override { return tracker_time_ms_; }
    double getFusionTimeMs() const override { return fusion_time_ms_; }

private:
    double tracker_time_ms_ = 0.0;
    double fusion_time_ms_ = 0.0;
    
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace cv_keyboard

#endif // CV_KEYBOARD_MEDIAPIPE_TRACKER_H