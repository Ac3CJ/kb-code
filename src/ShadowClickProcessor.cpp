#include "ShadowClickProcessor.h"
#include "HandData.h"
#include <iostream>

namespace cv_keyboard {

void ShadowClickProcessor::resetHistory() {
    finger_states_.clear();
    prev_gray_.release();
}

void ShadowClickProcessor::cycleDebugFinger(int hand_idx, int direction) {
    if (hand_idx < 0 || hand_idx > 1) return;
    
    debug_finger_idx_[hand_idx] = (debug_finger_idx_[hand_idx] + direction + 5) % 5;
    
    const std::string finger_names[] = {"Thumb", "Index Finger", "Middle Finger", "Ring Finger", "Pinky"};
    std::cout << "[Shadow] Hand " << (hand_idx + 1) << " visualizer shifted to: " << finger_names[debug_finger_idx_[hand_idx]] << "\n";
}

void ShadowClickProcessor::drawDebugWindow(const cv::Mat& roi_bgr, const cv::Mat& diff_mask, const cv::Mat& combined_mask, const cv::Mat& flow, const cv::Point2f& true_center, int hand_idx) {
    int scale = 10;
    int w = roi_bgr.cols;
    int h = roi_bgr.rows;

    cv::namedWindow(window_names_[hand_idx], cv::WINDOW_NORMAL | cv::WINDOW_GUI_NORMAL);

    cv::Mat pane1;
    cv::resize(roi_bgr, pane1, cv::Size(w * scale, h * scale), 0, 0, cv::INTER_NEAREST);

    cv::Mat pane2;
    cv::cvtColor(combined_mask, pane2, cv::COLOR_GRAY2BGR);
    cv::resize(pane2, pane2, cv::Size(w * scale, h * scale), 0, 0, cv::INTER_NEAREST);

    cv::Mat pane3;
    cv::Mat gray_roi;
    cv::cvtColor(roi_bgr, gray_roi, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray_roi, pane3, cv::COLOR_GRAY2BGR);
    pane3 *= 0.4; 
    cv::resize(pane3, pane3, cv::Size(w * scale, h * scale), 0, 0, cv::INTER_NEAREST);

    int step = 3; 
    for (int y = 0; y < h; y += step) {
        for (int x = 0; x < w; x += step) {
            if (diff_mask.at<uchar>(y, x) == 0) continue;

            cv::Point2f f = flow.at<cv::Point2f>(y, x);
            if (std::abs(f.x) < 0.3f && std::abs(f.y) < 0.3f) continue;

            cv::Point2f D(x - true_center.x, y - true_center.y);
            float dot = (D.x * f.x) + (D.y * f.y);
            cv::Scalar color = (dot < 0) ? cv::Scalar(0, 50, 255) : cv::Scalar(255, 100, 0); 

            cv::Point pt1(x * scale, y * scale);
            cv::Point pt2(static_cast<int>((x + (f.x * 2.0f)) * scale), static_cast<int>((y + (f.y * 2.0f)) * scale));

            cv::arrowedLine(pane3, pt1, pt2, color, 2, 8, 0, 0.4);
        }
    }

    cv::Mat display;
    std::vector<cv::Mat> panes = {pane1, pane2, pane3};
    cv::hconcat(panes, display);

    const std::string finger_names[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
    std::string title_label = "Hand " + std::to_string(hand_idx + 1) + " " + finger_names[debug_finger_idx_[hand_idx]];

    cv::putText(display, "1. Shifted ROI (" + title_label + ")", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "2. Shadow Mask", cv::Point(w * scale + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "3. Flow (Red=Converging)", cv::Point(w * scale * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    cv::Point crosshair_pt(w * scale * 2 + static_cast<int>(true_center.x * scale), static_cast<int>(true_center.y * scale));
    cv::drawMarker(display, crosshair_pt, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);

    cv::imshow(window_names_[hand_idx], display);
}

void ShadowClickProcessor::detectClicks(const std::vector<HandData>& hands, 
                                        const KeyboardMap& keyboard_map, 
                                        int frame_width, 
                                        int frame_height) {
    if (current_bgr_.empty()) return;

    cv::Mat current_gray;
    cv::cvtColor(current_bgr_, current_gray, cv::COLOR_BGR2GRAY);

    if (prev_gray_.empty()) {
        current_gray.copyTo(prev_gray_);
        return;
    }

    bool debug_rendered_this_frame[2] = {false, false};

    std::vector<HandData> sorted_hands = hands;
    std::sort(sorted_hands.begin(), sorted_hands.end(), [](const HandData& a, const HandData& b) {
        return a.landmarks[0].x < b.landmarks[0].x; 
    });

    int current_hand_idx = 0;

    for (const auto& hand : sorted_hands) {
        if (current_hand_idx > 1) break; // MVP only supports 2 visualizer windows

        for (int tip_idx : FINGER_TIP_INDICES) {
            int finger_id = (current_hand_idx << 8) | tip_idx;
            const auto& tip = hand.landmarks[tip_idx];

            int px = static_cast<int>(tip.x * frame_width);
            int py = static_cast<int>(tip.y * frame_height);
            
            int roi_top_left_x = px - (ROI_WIDTH / 2) + ROI_OFFSET_X;
            int roi_top_left_y = py - (ROI_HEIGHT / 2) + ROI_OFFSET_Y;
            
            cv::Rect roi_rect(roi_top_left_x, roi_top_left_y, ROI_WIDTH, ROI_HEIGHT);
            roi_rect &= cv::Rect(0, 0, current_bgr_.cols, current_bgr_.rows);
            if (roi_rect.width != ROI_WIDTH || roi_rect.height != ROI_HEIGHT) continue; 

            cv::Mat curr_roi_gray = current_gray(roi_rect);
            cv::Mat prev_roi_gray = prev_gray_(roi_rect);
            cv::Mat curr_roi_bgr = current_bgr_(roi_rect);

            cv::Mat prev_roi_blur, curr_roi_blur;
            cv::GaussianBlur(prev_roi_gray, prev_roi_blur, cv::Size(BLUR_SIZE, BLUR_SIZE), 0);
            cv::GaussianBlur(curr_roi_gray, curr_roi_blur, cv::Size(BLUR_SIZE, BLUR_SIZE), 0);

            cv::Mat flow;
            cv::calcOpticalFlowFarneback(prev_roi_blur, curr_roi_blur, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

            cv::Mat lab_roi, l_channel, shadow_mask, diff, diff_mask, combined_mask;
            cv::cvtColor(curr_roi_bgr, lab_roi, cv::COLOR_BGR2Lab);
            cv::extractChannel(lab_roi, l_channel, 0);
            cv::threshold(l_channel, shadow_mask, SHADOW_L_THRESH, 255, cv::THRESH_BINARY_INV);
            
            cv::absdiff(curr_roi_blur, prev_roi_blur, diff);
            cv::threshold(diff, diff_mask, MOTION_THRESH, 255, cv::THRESH_BINARY);
            
            cv::bitwise_and(shadow_mask, diff_mask, combined_mask);

            cv::Point2f finger_pos((ROI_WIDTH / 2.0f) - ROI_OFFSET_X, (ROI_HEIGHT / 2.0f) - ROI_OFFSET_Y);

            int array_idx = -1;
            for (int i = 0; i < 5; ++i) {
                if (FINGER_TIP_INDICES[i] == tip_idx) array_idx = i;
            }

            if (show_debug_window_ && array_idx == debug_finger_idx_[current_hand_idx]) {
                drawDebugWindow(curr_roi_bgr, diff_mask, combined_mask, flow, finger_pos, current_hand_idx);
                debug_rendered_this_frame[current_hand_idx] = true;
            }

            cv::Point2f shadow_vel(0, 0);
            cv::Point2f local_centroid(0, 0);
            int shadow_pixels = 0;

            for (int y = 0; y < combined_mask.rows; ++y) {
                for (int x = 0; x < combined_mask.cols; ++x) {
                    if (combined_mask.at<uchar>(y, x) > 0) {
                        shadow_vel += flow.at<cv::Point2f>(y, x);
                        local_centroid += cv::Point2f(x, y);
                        shadow_pixels++;
                    }
                }
            }

            if (shadow_pixels > 5) {
                shadow_vel /= static_cast<float>(shadow_pixels);
                local_centroid /= static_cast<float>(shadow_pixels);

                cv::Point2f finger_vel(0,0);
                if (finger_pos.x >= 0 && finger_pos.x < ROI_WIDTH && finger_pos.y >= 0 && finger_pos.y < ROI_HEIGHT) {
                    finger_vel = flow.at<cv::Point2f>(static_cast<int>(finger_pos.y), static_cast<int>(finger_pos.x));
                }

                cv::Point2f D = local_centroid - finger_pos;
                cv::Point2f V_rel = shadow_vel - finger_vel;

                float dot_product = (D.x * V_rel.x) + (D.y * V_rel.y);

                if (dot_product < -0.5f) {
                    finger_states_[finger_id] = StrikeState::CONVERGING;
                } 
                else if (dot_product >= -0.1f && finger_states_[finger_id] == StrikeState::CONVERGING) {
                    finger_states_[finger_id] = StrikeState::IDLE;
                    
                    float global_x = roi_rect.x + local_centroid.x;
                    float global_y = roi_rect.y + local_centroid.y;
                    
                    cv::Point2f physical_pt = keyboard_map.pixelToPhysical(global_x, global_y);
                    std::string key = keyboard_map.getKeyAt(physical_pt.x, physical_pt.y);
                    
                    if (!key.empty()) active_clicks_[finger_id] = key;
                }
            } else {
                finger_states_[finger_id] = StrikeState::IDLE; 
            }
        }
        current_hand_idx++;
    }

    if (show_debug_window_) {
        for (int i = 0; i < 2; ++i) {
            if (!debug_rendered_this_frame[i]) {
                cv::Mat blank = cv::Mat::zeros(cv::Size(ROI_WIDTH * 10 * 3, ROI_HEIGHT * 10), CV_8UC3);
                cv::putText(blank, "WAITING FOR TARGET FINGER...", cv::Point(50, (ROI_HEIGHT * 10) / 2), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                cv::namedWindow(window_names_[i], cv::WINDOW_NORMAL | cv::WINDOW_GUI_NORMAL); 
                cv::imshow(window_names_[i], blank);
            }
        }
    }

    current_gray.copyTo(prev_gray_);
}

} // namespace cv_keyboard