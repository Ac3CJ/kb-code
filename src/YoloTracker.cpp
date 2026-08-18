#include "YoloTracker.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace cv_keyboard {

YoloTracker::YoloTracker() = default;

bool YoloTracker::init() {
    std::string model_path = "models/hand_pose.onnx"; // Path to your exported YOLO model
    
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
        
        // Optimise for CPU if CUDA isn't available
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); 
        
        initialised_ = true;
        std::cout << "[YoloTracker] Initialised successfully from " << model_path << "\n";
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[YoloTracker] Failed to load ONNX model: " << e.what() << "\n";
        return false;
    }
}

std::shared_ptr<const std::vector<HandData>> YoloTracker::detect(const cv::Mat& frame, int64_t timestamp_us) {
    if (!initialised_ || frame.empty()) return {};

    int64_t t_start = cv::getTickCount();
    latest_timestamp_us_ = timestamp_us;

    // 1. Preprocess: Convert frame to a 640x640 float32 blob (RGB, scaled 0-1)
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, INPUT_SIZE, cv::Scalar(), true, false);
    net_.setInput(blob);

    // 2. Inference
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    // 3. Postprocess: YOLOv8 output is [1, 68, 8400]
    // We transpose it to [8400, 68] for easier row-by-row iteration
    cv::Mat out_tensor = outputs[0]; 
    cv::Mat out_tensor_reshaped = out_tensor.reshape(1, out_tensor.size[1]); // [68, 8400]
    cv::Mat predictions;
    cv::transpose(out_tensor_reshaped, predictions); // [8400, 68]

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<Landmark>> keypoints_list;

    float x_scale = static_cast<float>(frame.cols) / INPUT_SIZE.width;
    float y_scale = static_cast<float>(frame.rows) / INPUT_SIZE.height;

    // 4. Parse the 8400 anchor predictions
    for (int i = 0; i < predictions.rows; ++i) {
        float box_score = predictions.at<float>(i, 4);
        
        if (box_score >= CONFIDENCE_THRESHOLD) {
            // Extract bounding box
            float cx = predictions.at<float>(i, 0);
            float cy = predictions.at<float>(i, 1);
            float w = predictions.at<float>(i, 2);
            float h = predictions.at<float>(i, 3);

            int left = static_cast<int>((cx - 0.5f * w) * x_scale);
            int top = static_cast<int>((cy - 0.5f * h) * y_scale);
            int width = static_cast<int>(w * x_scale);
            int height = static_cast<int>(h * y_scale);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(box_score);
            class_ids.push_back(0); // Only 1 class: Hand

            // Extract 21 Keypoints
            std::vector<Landmark> kpts(NUM_KEYPOINTS);
            for (int k = 0; k < NUM_KEYPOINTS; ++k) {
                int base_idx = 5 + (k * 3);
                // YOLO outputs pixel coordinates relative to the 640x640 input, 
                // so we normalise them back to [0, 1] to match MediaPipe
                kpts[k].x = (predictions.at<float>(i, base_idx) * x_scale) / frame.cols;
                kpts[k].y = (predictions.at<float>(i, base_idx + 1) * y_scale) / frame.rows;
                kpts[k].confidence = predictions.at<float>(i, base_idx + 2);
                kpts[k].z = 0.0f; // YOLO 2D doesn't predict Z
            }
            keypoints_list.push_back(kpts);
        }
    }

    // 5. Apply Non-Maximum Suppression (NMS) to remove duplicate overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    // 6. Build the final HandData output
    std::lock_guard<std::mutex> lock(mutex_);
    cached_hands_->clear();
    cached_hands_->resize(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        
        (*cached_hands_)[i].hand_confidence = confidences[idx];
        (*cached_hands_)[i].handedness = 0; // YOLOv8-pose doesn't natively classify L/R without custom training
        (*cached_hands_)[i].timestamp_us = timestamp_us;

        for (int k = 0; k < NUM_KEYPOINTS; ++k) {
            (*cached_hands_)[i].landmarks[k] = keypoints_list[idx][k];
        }
    }

    int64_t t_end = cv::getTickCount();
    tracker_time_ms_ = (t_end - t_start) * 1000.0 / cv::getTickFrequency();
    fusion_time_ms_ = 0.0; // Update this if KF added

    return cached_hands_;
}

} // namespace cv_keyboard