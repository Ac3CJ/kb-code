#ifndef CV_KEYBOARD_RTMPOSE_TRACKER_H
#define CV_KEYBOARD_RTMPOSE_TRACKER_H

#include "IHandTracker.h"
#include <onnxruntime_cxx_api.h>
#include <mutex>
#include <string>
#include <vector>
#include <memory>

namespace cv_keyboard {

class RTMPoseTracker : public IHandTracker {
public:
    RTMPoseTracker();
    ~RTMPoseTracker() override = default;

    bool init() override;
    std::shared_ptr<const std::vector<HandData>> detect(const cv::Mat& frame, int64_t timestamp_us) override;
    
    int64_t latestTimestamp() const override { return latest_timestamp_us_; }
    bool isInitialised() const override { return initialised_; }
    
    double getTrackerTimeMs() const override { return tracker_time_ms_; }
    double getFusionTimeMs() const override { return fusion_time_ms_; }

    void setUsePreprocessing(bool use) { use_preprocessing_ = use; }
    bool isUsingPreprocessing() const { return use_preprocessing_; }

private:
    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "RTMPoseTracker"};
    Ort::SessionOptions session_options_;

    std::unique_ptr<Ort::Session> detector_session_;
    std::unique_ptr<Ort::Session> pose_session_;
    
    Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    bool initialised_ = false;
    bool use_preprocessing_ = true;

    int64_t latest_timestamp_us_ = 0;
    double tracker_time_ms_ = 0.0;
    double fusion_time_ms_ = 0.0;

    std::shared_ptr<std::vector<HandData>> cached_hands_ = std::make_shared<std::vector<HandData>>();
    std::mutex mutex_;

    // Model tensor shapes
    const int64_t DET_W = 320;
    const int64_t DET_H = 320;
    const int64_t POSE_W = 256;
    const int64_t POSE_H = 256;
    
    const float DET_CONFIDENCE_THRESH = 0.4f;
    const int NUM_KEYPOINTS = 21;
    const float SIMCC_SPLIT_RATIO = 2.0f;

    // Dynamically cached I/O node names
    std::vector<std::string> det_input_names_str_;
    std::vector<std::string> det_output_names_str_;
    std::vector<std::string> pose_input_names_str_;
    std::vector<std::string> pose_output_names_str_;

    cv::Rect getExpandedBox(const cv::Rect& box, int frame_cols, int frame_rows, float scale = 1.25f);
    void matToNCHW(const cv::Mat& src, std::vector<float>& dst, bool to_rgb, const cv::Scalar& mean, const cv::Scalar& std_dev);
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_RTMPOSE_TRACKER_H