#include "RTMPoseTracker.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>

namespace cv_keyboard {

RTMPoseTracker::RTMPoseTracker() = default;

bool RTMPoseTracker::init() {
    std::string det_path = "./rtmdet_onnx/rtmdet_simplified.onnx";
    std::string pose_path = "./rtmpose_onnx/rtmpose_simplified.onnx";
    
    try {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        OrtOpenVINOProviderOptions ov_options;
        ov_options.device_type = "GPU";

        session_options_.AppendExecutionProvider_OpenVINO(ov_options);
        
        detector_session_ = std::make_unique<Ort::Session>(ort_env_, det_path.c_str(), session_options_);
        pose_session_ = std::make_unique<Ort::Session>(ort_env_, pose_path.c_str(), session_options_);
        
        Ort::AllocatorWithDefaultOptions allocator;

        // Query input/output names dynamically for Detector
        size_t num_det_inputs = detector_session_->GetInputCount();
        for (size_t i = 0; i < num_det_inputs; ++i) {
            det_input_names_str_.push_back(detector_session_->GetInputNameAllocated(i, allocator).get());
        }
        size_t num_det_outputs = detector_session_->GetOutputCount();
        for (size_t i = 0; i < num_det_outputs; ++i) {
            det_output_names_str_.push_back(detector_session_->GetOutputNameAllocated(i, allocator).get());
        }

        // Query input/output names dynamically for Pose
        size_t num_pose_inputs = pose_session_->GetInputCount();
        for (size_t i = 0; i < num_pose_inputs; ++i) {
            pose_input_names_str_.push_back(pose_session_->GetInputNameAllocated(i, allocator).get());
        }
        size_t num_pose_outputs = pose_session_->GetOutputCount();
        for (size_t i = 0; i < num_pose_outputs; ++i) {
            pose_output_names_str_.push_back(pose_session_->GetOutputNameAllocated(i, allocator).get());
        }

        initialised_ = true;
        std::cout << "[RTMPoseTracker] ORT Initialised successfully.\n";
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[RTMPoseTracker] ORT Initialization Failed: " << e.what() << "\n";
        return false;
    }
}

cv::Rect RTMPoseTracker::getExpandedBox(const cv::Rect& box, int frame_cols, int frame_rows, float scale) {
    int center_x = box.x + box.width / 2;
    int center_y = box.y + box.height / 2;
    int new_w = static_cast<int>(box.width * scale);
    int new_h = static_cast<int>(box.height * scale);
    
    int new_x = std::max(0, center_x - new_w / 2);
    int new_y = std::max(0, center_y - new_h / 2);
    new_w = std::min(frame_cols - new_x, new_w);
    new_h = std::min(frame_rows - new_y, new_h);
    
    return cv::Rect(new_x, new_y, new_w, new_h);
}

void RTMPoseTracker::matToNCHW(const cv::Mat& src, std::vector<float>& dst, bool to_rgb, 
                               const cv::Scalar& mean, const cv::Scalar& std_dev) {
    cv::Mat converted;
    if (to_rgb) {
        cv::cvtColor(src, converted, cv::COLOR_BGR2RGB);
    } else {
        converted = src;
    }

    int width = converted.cols;
    int height = converted.rows;
    int channels = converted.channels();
    dst.resize(channels * height * width);

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float val = static_cast<float>(converted.at<cv::Vec3b>(h, w)[c]);
                if (use_preprocessing_) {
                    val = (val - static_cast<float>(mean[c])) / static_cast<float>(std_dev[c]);
                } else {
                    val = val / 255.0f;
                }
                dst[c * (height * width) + h * width + w] = val;
            }
        }
    }
}

std::shared_ptr<const std::vector<HandData>> RTMPoseTracker::detect(const cv::Mat& frame, int64_t timestamp_us) {
    if (!initialised_ || frame.empty()) return {};

    int64_t t_start = cv::getTickCount();
    latest_timestamp_us_ = timestamp_us;
    std::vector<HandData> current_hands;

    // ==========================================
    // STAGE 1: RTMDet (Hand Bounding Box)
    // ==========================================
    float scale = std::min(static_cast<float>(DET_W) / frame.cols, static_cast<float>(DET_H) / frame.rows);
    int nw = static_cast<int>(frame.cols * scale);
    int nh = static_cast<int>(frame.rows * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh));

    cv::Mat padded = cv::Mat(DET_H, DET_W, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(0, 0, nw, nh)));

    std::vector<float> det_input_tensor_values;
    matToNCHW(padded, det_input_tensor_values, false, 
              cv::Scalar(103.53, 116.28, 123.675), 
              cv::Scalar(57.375, 57.12, 58.395));

    std::array<int64_t, 4> det_input_shape = {1, 3, DET_H, DET_W};
    Ort::Value det_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, det_input_tensor_values.data(), det_input_tensor_values.size(),
        det_input_shape.data(), det_input_shape.size()
    );

    std::vector<const char*> det_in_names;
    for (const auto& s : det_input_names_str_) det_in_names.push_back(s.c_str());
    std::vector<const char*> det_out_names;
    for (const auto& s : det_output_names_str_) det_out_names.push_back(s.c_str());

    auto det_outputs = detector_session_->Run(
        Ort::RunOptions{nullptr}, det_in_names.data(), &det_input_tensor, 1,
        det_out_names.data(), det_out_names.size()
    );

    // Parse Detections [1, N, 5] -> [x1, y1, x2, y2, score]
    float* det_data = det_outputs[0].GetTensorMutableData<float>();
    auto det_info = det_outputs[0].GetTensorTypeAndShapeInfo();
    auto det_shape = det_info.GetShape();

    int num_boxes = (det_shape.size() == 3) ? static_cast<int>(det_shape[1]) : static_cast<int>(det_shape[0]);
    int elem_per_box = static_cast<int>(det_shape.back());

    std::vector<cv::Rect> valid_boxes;
    for (int i = 0; i < num_boxes; ++i) {
        float* row = det_data + (i * elem_per_box);
        float score = row[4];

        if (score >= DET_CONFIDENCE_THRESH) {
            float x1 = row[0] / scale;
            float y1 = row[1] / scale;
            float x2 = row[2] / scale;
            float y2 = row[3] / scale;

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(frame.cols)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(frame.rows)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(frame.cols)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(frame.rows)));

            if ((x2 - x1) > 10 && (y2 - y1) > 10) {
                valid_boxes.push_back(cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)), 
                                              cv::Point(static_cast<int>(x2), static_cast<int>(y2))));
            }
        }
    }

    // ==========================================
    // STAGE 2: RTMPose (Pose Keypoints)
    // ==========================================
    std::vector<const char*> pose_in_names;
    for (const auto& s : pose_input_names_str_) pose_in_names.push_back(s.c_str());
    std::vector<const char*> pose_out_names;
    for (const auto& s : pose_output_names_str_) pose_out_names.push_back(s.c_str());

    for (const auto& raw_box : valid_boxes) {
        cv::Rect crop_box = getExpandedBox(raw_box, frame.cols, frame.rows, 1.25f);
        if (crop_box.area() <= 0) continue;

        cv::Mat crop = frame(crop_box);
        cv::Mat resized_crop;
        cv::resize(crop, resized_crop, cv::Size(POSE_W, POSE_H));

        std::vector<float> pose_input_tensor_values;
        matToNCHW(resized_crop, pose_input_tensor_values, true,
                  cv::Scalar(123.675, 116.28, 103.53),
                  cv::Scalar(58.395, 57.12, 57.375));

        std::array<int64_t, 4> pose_input_shape = {1, 3, POSE_H, POSE_W};
        Ort::Value pose_input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, pose_input_tensor_values.data(), pose_input_tensor_values.size(),
            pose_input_shape.data(), pose_input_shape.size()
        );

        auto pose_outputs = pose_session_->Run(
            Ort::RunOptions{nullptr}, pose_in_names.data(), &pose_input_tensor, 1,
            pose_out_names.data(), pose_out_names.size()
        );

        HandData hand_data;
        hand_data.timestamp_us = timestamp_us;
        hand_data.hand_confidence = 1.0f;

        if (pose_outputs.size() == 2) {
            // Raw SimCC Outputs: simcc_x [1, 21, 512] and simcc_y [1, 21, 512]
            float* simcc_x = pose_outputs[0].GetTensorMutableData<float>();
            float* simcc_y = pose_outputs[1].GetTensorMutableData<float>();
            int bins = static_cast<int>(pose_outputs[0].GetTensorTypeAndShapeInfo().GetShape().back());

            for (int k = 0; k < NUM_KEYPOINTS; ++k) {
                float* x_ptr = simcc_x + (k * bins);
                float* y_ptr = simcc_y + (k * bins);

                int max_idx_x = static_cast<int>(std::max_element(x_ptr, x_ptr + bins) - x_ptr);
                int max_idx_y = static_cast<int>(std::max_element(y_ptr, y_ptr + bins) - y_ptr);

                float local_x = max_idx_x / SIMCC_SPLIT_RATIO;
                float local_y = max_idx_y / SIMCC_SPLIT_RATIO;

                float global_x = crop_box.x + (local_x / POSE_W) * crop_box.width;
                float global_y = crop_box.y + (local_y / POSE_H) * crop_box.height;

                hand_data.landmarks[k].x = global_x / static_cast<float>(frame.cols);
                hand_data.landmarks[k].y = global_y / static_cast<float>(frame.rows);
                hand_data.landmarks[k].confidence = (x_ptr[max_idx_x] + y_ptr[max_idx_y]) / 2.0f;
            }
        } else {
            // Decoded Output [1, 21, 2] or [1, 21, 3]
            float* kpts = pose_outputs[0].GetTensorMutableData<float>();
            auto shape = pose_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            int step = static_cast<int>(shape.back());

            for (int k = 0; k < NUM_KEYPOINTS; ++k) {
                float local_x = kpts[k * step];
                float local_y = kpts[k * step + 1];

                float global_x = crop_box.x + (local_x / POSE_W) * crop_box.width;
                float global_y = crop_box.y + (local_y / POSE_H) * crop_box.height;

                hand_data.landmarks[k].x = global_x / static_cast<float>(frame.cols);
                hand_data.landmarks[k].y = global_y / static_cast<float>(frame.rows);
                hand_data.landmarks[k].confidence = (step > 2) ? kpts[k * step + 2] : 1.0f;
            }
        }
        current_hands.push_back(hand_data);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    *cached_hands_ = current_hands;

    int64_t t_end = cv::getTickCount();
    tracker_time_ms_ = (t_end - t_start) * 1000.0 / cv::getTickFrequency();
    fusion_time_ms_ = 0.0;

    return cached_hands_;
}

} // namespace cv_keyboard