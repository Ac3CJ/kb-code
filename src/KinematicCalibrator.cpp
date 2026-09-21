// PLEASE NOTE THAT THE HANDEDNESS IS NOT CONSISTENT BETWEEN MODELS AND CAN CHANGE BASED ON THE MODEL USED. THIS IS A KNOWN ISSUE WITH MEDIAPIPE AND SHOULD BE HANDLED IN THE FUTURE.

#include "KinematicCalibrator.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

namespace cv_keyboard {

namespace {
    // Math helpers for the Custom Knuckle PnP Solver
    cv::Point3f normalizeVector(const cv::Point3f& v) {
        float mag = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        return (mag > 0.0f) ? cv::Point3f(v.x / mag, v.y / mag, v.z / mag) : v;
    }

    float dotProduct(const cv::Point3f& a, const cv::Point3f& b) {
        return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
    }

    // Iterative Gradient Descent Non-Linear Least Squares Solver
    std::vector<cv::Point3f> solveKnucklePnP(
        const std::vector<cv::Point2f>& knuckle_pixels,
        const std::vector<float>& gt_lengths,
        const cv::Mat& camera_matrix,
        float initial_depth_guess_cm) 
    {
        if (camera_matrix.empty()) return std::vector<cv::Point3f>(4, cv::Point3f(0,0,0));

        cv::Mat K_inv = camera_matrix.inv();
        std::vector<cv::Point3f> rays(4);
        
        // 1. Generate 3D Unit Rays from Camera Center
        for (int i = 0; i < 4; ++i) {
            cv::Mat uv = (cv::Mat_<double>(3, 1) << knuckle_pixels[i].x, knuckle_pixels[i].y, 1.0);
            cv::Mat ray_cam = K_inv * uv;
            rays[i] = normalizeVector(cv::Point3f(
                static_cast<float>(ray_cam.at<double>(0)), 
                static_cast<float>(ray_cam.at<double>(1)), 
                static_cast<float>(ray_cam.at<double>(2))
            ));
        }

        std::vector<float> depths(4, initial_depth_guess_cm);
        
        // Mapping matches the 6 GT lengths layout: (5-9, 9-13, 13-17, 5-17, 9-17, 5-13)
        int pairs[6][2] = {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {1, 3}, {0, 2}};

        const int MAX_ITERATIONS = 100;
        const float LEARNING_RATE = 0.05f;

        // 2. Iteratively slide knuckles along their rays to minimize length errors
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            std::vector<float> gradients(4, 0.0f);
            float total_error = 0.0f;

            for (int p = 0; p < 6; ++p) {
                int i = pairs[p][0];
                int j = pairs[p][1];

                cv::Point3f p_i = cv::Point3f(rays[i].x * depths[i], rays[i].y * depths[i], rays[i].z * depths[i]);
                cv::Point3f p_j = cv::Point3f(rays[j].x * depths[j], rays[j].y * depths[j], rays[j].z * depths[j]);

                cv::Point3f diff = cv::Point3f(p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z);
                float dist = std::sqrt(dotProduct(diff, diff));
                
                float error = dist - gt_lengths[p];
                total_error += std::abs(error);

                if (dist > 0.001f) {
                    float err_term = error / dist;
                    gradients[i] += err_term * dotProduct(diff, rays[i]);
                    gradients[j] += err_term * dotProduct(cv::Point3f(-diff.x, -diff.y, -diff.z), rays[j]);
                }
            }

            for (int i = 0; i < 4; ++i) {
                depths[i] -= LEARNING_RATE * gradients[i];
                if (depths[i] < 1.0f) depths[i] = 1.0f; // Prevent snapping behind camera lens
            }

            if (total_error < 0.1f) break; // Early convergence threshold
        }

        // 3. Output localized 3D points relative to the camera sensor
        std::vector<cv::Point3f> camera_space_points(4);
        for (int i = 0; i < 4; ++i) {
            camera_space_points[i] = cv::Point3f(rays[i].x * depths[i], rays[i].y * depths[i], rays[i].z * depths[i]);
        }

        return camera_space_points;
    }
}

void KinematicCalibrator::startCalibration() {
    is_calibrating_ = true;
    is_calibrated_ = false;
    frames_collected_ = 0;
    
    hand0_buffer_.clear();
    hand1_buffer_.clear();
    
    hand0_profile_.is_valid = false;
    hand1_profile_.is_valid = false;
    
    std::cout << "\n[KinematicCalibrator] Calibration Started.\n";
    std::cout << "Place ONE hand flat on the keyboard to avoid marker occlusion...\n";
}

bool KinematicCalibrator::updateCalibration(const std::vector<HandData>& hands, const KeyboardMap& kb_map, int frame_width, int frame_height) {
    if (!is_calibrating_ || !kb_map.hasValidTransform()) return false;
    
    bool captured_data = false;

    for (const auto& hand : hands) {
        if (hand.handedness == 1 && !hand0_profile_.is_valid) { 
            hand0_buffer_.push_back(hand.landmarks);
            captured_data = true;
        } else if (hand.handedness == 2 && !hand1_profile_.is_valid) { 
            hand1_buffer_.push_back(hand.landmarks);
            captured_data = true;
        }
    }

    if (captured_data) {
        frames_collected_++;
    }

    if (frames_collected_ >= MAX_CALIBRATION_FRAMES) {
        finalizeCalibration(kb_map, frame_width, frame_height);
        
        if (hand0_profile_.is_valid && hand1_profile_.is_valid) {
            is_calibrating_ = false;
            is_calibrated_ = true;
            std::cout << "[KinematicCalibrator] Both hands calibrated successfully. Calibration window closed.\n";
            return true; 
        } else {
            frames_collected_ = 0;
            hand0_buffer_.clear();
            hand1_buffer_.clear();
            
            if (!hand0_profile_.is_valid) {
                std::cout << "[KinematicCalibrator] Waiting for hand...\n";
            } else if (!hand1_profile_.is_valid) {
                std::cout << "[KinematicCalibrator] Waiting for other hand...\n";
            }
            
            return false;
        }
    }
    
    return false;
}

void KinematicCalibrator::finalizeCalibration(const KeyboardMap& kb_map, int frame_width, int frame_height) {
    if (!hand0_profile_.is_valid && hand0_buffer_.size() >= static_cast<size_t>(MAX_CALIBRATION_FRAMES / 2)) {
        calculateBoneLengths(hand0_buffer_, hand0_profile_, kb_map, frame_width, frame_height);
        std::cout << "[KinematicCalibrator] Hand profile successfully captured.\n";
    }
    if (!hand1_profile_.is_valid && hand1_buffer_.size() >= static_cast<size_t>(MAX_CALIBRATION_FRAMES / 2)) {
        calculateBoneLengths(hand1_buffer_, hand1_profile_, kb_map, frame_width, frame_height);
        std::cout << "[KinematicCalibrator] Other hand profile successfully captured.\n";
    }
}

void KinematicCalibrator::calculateBoneLengths(const std::vector<std::array<Landmark, 21>>& buffer, HandProfile& profile, const KeyboardMap& kb_map, int frame_width, int frame_height) {
    std::array<cv::Point2f, 21> median_landmarks_cm;
    
    for (int i = 0; i < 21; ++i) {
        std::vector<float> x_vals, y_vals;
        for (const auto& frame_landmarks : buffer) {
            x_vals.push_back(frame_landmarks[i].x);
            y_vals.push_back(frame_landmarks[i].y);
        }
        
        std::sort(x_vals.begin(), x_vals.end());
        std::sort(y_vals.begin(), y_vals.end());
        
        float median_x = x_vals[x_vals.size() / 2];
        float median_y = y_vals[y_vals.size() / 2];

        float px = median_x * frame_width;
        float py = median_y * frame_height;

        median_landmarks_cm[i] = kb_map.pixelToPhysical(px, py);
    }

    profile.bone_lengths_cm.clear();
    profile.angles_rad.clear();

    auto getDist = [&](int i1, int i2) {
        return std::sqrt(std::pow(median_landmarks_cm[i2].x - median_landmarks_cm[i1].x, 2) + 
                         std::pow(median_landmarks_cm[i2].y - median_landmarks_cm[i1].y, 2));
    };

    for (int c = 0; c < NUM_HAND_CONNECTIONS; ++c) {
        int idx1 = HAND_CONNECTIONS[c][0];
        int idx2 = HAND_CONNECTIONS[c][1];
        profile.bone_lengths_cm[c] = getDist(idx1, idx2);
    }

    profile.bone_lengths_cm[23] = getDist(5, 17);  
    profile.bone_lengths_cm[24] = getDist(9, 17);  
    profile.bone_lengths_cm[25] = getDist(5, 13);  

    auto getAngle = [&](int p1, int p2, int p3) {
        float v1x = median_landmarks_cm[p1].x - median_landmarks_cm[p2].x;
        float v1y = median_landmarks_cm[p1].y - median_landmarks_cm[p2].y;
        float v2x = median_landmarks_cm[p3].x - median_landmarks_cm[p2].x;
        float v2y = median_landmarks_cm[p3].y - median_landmarks_cm[p2].y;
        
        float dot = (v1x * v2x) + (v1y * v2y);
        float mag1 = std::sqrt(v1x * v1x + v1y * v1y);
        float mag2 = std::sqrt(v2x * v2x + v2y * v2y);
        
        if (mag1 * mag2 == 0.0f) return 0.0f;
        float val = std::max(-1.0f, std::min(1.0f, dot / (mag1 * mag2)));
        return std::acos(val); 
    };

    profile.angles_rad["wrist_5_to_9"] = getAngle(5, 0, 9);
    profile.angles_rad["wrist_9_to_13"] = getAngle(9, 0, 13);
    profile.angles_rad["wrist_13_to_17"] = getAngle(13, 0, 17);

    profile.angles_rad["knuckle_0_5_9"] = getAngle(0, 5, 9);
    profile.angles_rad["knuckle_5_9_13"] = getAngle(5, 9, 13);
    profile.angles_rad["knuckle_9_13_17"] = getAngle(9, 13, 17);
    profile.angles_rad["knuckle_13_17_0"] = getAngle(13, 17, 0);

    profile.is_valid = true;
}

bool KinematicCalibrator::saveProfile(const std::string& filepath) const {
    std::filesystem::path path(filepath);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    file << "{\n";
    
    auto writeHand = [&](const std::string& hand_name, const HandProfile& profile, bool is_last) {
        file << "  \"" << hand_name << "\": {\n";
        file << "    \"is_valid\": " << (profile.is_valid ? "true" : "false");
        
        if (profile.is_valid) {
            file << ",\n    \"bone_lengths_cm\": {\n";
            int count = 0;
            for (const auto& [conn_idx, length] : profile.bone_lengths_cm) {
                file << "      \"" << conn_idx << "\": " << length;
                if (++count < profile.bone_lengths_cm.size()) file << ",\n";
                else file << "\n";
            }
            file << "    },\n";

            file << "    \"angles_rad\": {\n";
            count = 0;
            for (const auto& [angle_name, rad] : profile.angles_rad) {
                file << "      \"" << angle_name << "\": " << rad;
                if (++count < profile.angles_rad.size()) file << ",\n";
                else file << "\n";
            }
            file << "    }\n";
        } else {
            file << "\n";
        }
        file << "  }" << (is_last ? "\n" : ",\n");
    };

    writeHand("Hand0", hand0_profile_, false);
    writeHand("Hand1", hand1_profile_, true);
    
    file << "}\n";
    return true;
}

bool KinematicCalibrator::loadProfile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    HandProfile* current_profile = nullptr;
    
    enum class ParseState { NONE, BONES, ANGLES };
    ParseState state = ParseState::NONE;

    hand0_profile_ = HandProfile{};
    hand1_profile_ = HandProfile{};
    is_calibrated_ = false;

    while (std::getline(file, line)) {
        if (line.find("\"Hand1\"") != std::string::npos) {
            current_profile = &hand1_profile_;
            continue;
        } else if (line.find("\"Hand0\"") != std::string::npos) {
            current_profile = &hand0_profile_;
            continue;
        }

        if (!current_profile) continue;

        if (line.find("\"is_valid\": true") != std::string::npos) {
            current_profile->is_valid = true;
            continue;
        } else if (line.find("\"is_valid\": false") != std::string::npos) {
            current_profile->is_valid = false;
            continue;
        }

        if (line.find("\"bone_lengths_cm\":") != std::string::npos) {
            state = ParseState::BONES;
            continue;
        } else if (line.find("\"angles_rad\":") != std::string::npos) {
            state = ParseState::ANGLES;
            continue;
        } else if (line.find("}") != std::string::npos && line.find(":") == std::string::npos) {
            if (state != ParseState::NONE) state = ParseState::NONE;
        }

        size_t colon_pos = line.find(':');
        size_t quote1 = line.find('"');
        if (colon_pos != std::string::npos && quote1 != std::string::npos && quote1 < colon_pos) {
            size_t quote2 = line.find('"', quote1 + 1);
            if (quote2 != std::string::npos && quote2 < colon_pos) {
                std::string key_str = line.substr(quote1 + 1, quote2 - quote1 - 1);
                
                try {
                    std::string val_str = line.substr(colon_pos + 1);
                    val_str.erase(std::remove(val_str.begin(), val_str.end(), ','), val_str.end());
                    val_str.erase(std::remove_if(val_str.begin(), val_str.end(), ::isspace), val_str.end());
                    float parsed_val = std::stof(val_str);

                    if (state == ParseState::BONES) {
                        bool is_numeric = !key_str.empty() && std::all_of(key_str.begin(), key_str.end(), ::isdigit);
                        if (is_numeric) {
                            current_profile->bone_lengths_cm[std::stoi(key_str)] = parsed_val;
                        }
                    } else if (state == ParseState::ANGLES) {
                        current_profile->angles_rad[key_str] = parsed_val;
                    }
                } catch (...) { }
            }
        }
    }

    is_calibrated_ = (hand0_profile_.is_valid && hand1_profile_.is_valid);
    return is_calibrated_;
}

std::vector<PhysicalHand> KinematicCalibrator::transform(const std::vector<HandData>& hands, const KeyboardMap& kb_map, int frame_width, int frame_height) {
    std::vector<PhysicalHand> physical_hands;
    if (!kb_map.hasValidTransform()) return physical_hands;

    for (const auto& hand : hands) {
        PhysicalHand phys_hand;
        phys_hand.handedness = hand.handedness;
        phys_hand.timestamp_us = hand.timestamp_us;

        // 1. Initial Planar Projection (Fallback for non-solved joints like the wrist)
        for (int i = 0; i < 21; ++i) {
            const auto& landmark = hand.landmarks[i];
            float px = landmark.x * frame_width;
            float py = landmark.y * frame_height;
            cv::Point2f pt_cm = kb_map.pixelToPhysical(px, py);
            
            phys_hand.landmarks[i].x_cm = pt_cm.x;
            phys_hand.landmarks[i].y_cm = pt_cm.y;
            phys_hand.landmarks[i].z_cm = 0.0f; 

            float prev_px = (landmark.x - landmark.vx) * frame_width;
            float prev_py = (landmark.y - landmark.vy) * frame_height;
            cv::Point2f prev_pt_cm = kb_map.pixelToPhysical(prev_px, prev_py);
            
            phys_hand.landmarks[i].vx_cm = pt_cm.x - prev_pt_cm.x;
            phys_hand.landmarks[i].vy_cm = pt_cm.y - prev_pt_cm.y;
            phys_hand.landmarks[i].vz_cm = 0.0f; 
        }

        const HandProfile* profile = nullptr;
        if (hand.handedness == 1 && hand0_profile_.is_valid) {
            profile = &hand0_profile_;
        } else if (hand.handedness == 2 && hand1_profile_.is_valid) {
            profile = &hand1_profile_;
        }

        // 2. Custom Knuckle PnP Execution
        if (profile && profile->bone_lengths_cm.count(20) && profile->bone_lengths_cm.count(25)) {
            
            // Extract the 2D Pixels for 5, 9, 13, 17
            std::vector<cv::Point2f> knuckle_pixels(4);
            int knuckles[4] = {5, 9, 13, 17};
            for (int i = 0; i < 4; ++i) {
                knuckle_pixels[i] = cv::Point2f(hand.landmarks[knuckles[i]].x * frame_width,
                                                hand.landmarks[knuckles[i]].y * frame_height);
            }

            // Extract the 6 Structural Ground Truth Spans
            std::vector<float> gt_lengths = {
                profile->bone_lengths_cm.at(20), // 5-9
                profile->bone_lengths_cm.at(21), // 9-13
                profile->bone_lengths_cm.at(22), // 13-17
                profile->bone_lengths_cm.at(23), // 5-17
                profile->bone_lengths_cm.at(24), // 9-17
                profile->bone_lengths_cm.at(25)  // 5-13
            };

            // Estimate starting depth from board tracking distance
            float initial_depth = kb_map.getCameraHeightCm();
            if (initial_depth < 1.0f) initial_depth = 40.0f; // Safety fallback

            // Execute the Gradient Descent
            std::vector<cv::Point3f> cam_knuckles = solveKnucklePnP(knuckle_pixels, gt_lengths, kb_map.getCameraMatrix(), initial_depth);

            // 3. Translate Solved Coordinates to Physical Keyboard Space
            cv::Mat R;
            cv::Rodrigues(kb_map.getRvec(), R);
            cv::Mat R_inv = R.t();
            cv::Vec3d tvec = kb_map.getTvec();
            cv::Mat t = (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);

            for (int i = 0; i < 4; ++i) {
                cv::Mat P_cam = (cv::Mat_<double>(3, 1) << cam_knuckles[i].x, cam_knuckles[i].y, cam_knuckles[i].z);
                
                // Project from Camera Space to World Space
                cv::Mat P_world = R_inv * (P_cam - t);
                
                // Overwrite the Homography baseline with the True 3D projection
                phys_hand.landmarks[knuckles[i]].x_cm = P_world.at<double>(0, 0);
                phys_hand.landmarks[knuckles[i]].y_cm = P_world.at<double>(1, 0);
                phys_hand.landmarks[knuckles[i]].z_cm = P_world.at<double>(2, 0);
            }

            // 4. Cascade Fingers relative to the True 3D Knuckles
            int finger_chains[4][4] = {
                {5, 6, 7, 8},     // Index
                {9, 10, 11, 12},  // Middle
                {13, 14, 15, 16}, // Ring
                {17, 18, 19, 20}  // Pinky
            };

            for(int f = 0; f < 4; ++f) {
                for(int j = 0; j < 3; ++j) {
                    int idx1 = finger_chains[f][j];
                    int idx2 = finger_chains[f][j+1];
                    
                    int c_idx = -1;
                    for(int c = 0; c < 20; ++c) {
                        if (HAND_CONNECTIONS[c][0] == idx1 && HAND_CONNECTIONS[c][1] == idx2) {
                            c_idx = c; break;
                        }
                    }

                    if (c_idx != -1) {
                        float dx = phys_hand.landmarks[idx2].x_cm - phys_hand.landmarks[idx1].x_cm;
                        float dy = phys_hand.landmarks[idx2].y_cm - phys_hand.landmarks[idx1].y_cm;
                        float l_obs = std::sqrt(dx*dx + dy*dy);
                        float l_gt = profile->bone_lengths_cm.at(c_idx);
                        
                        float radicand = std::max(0.0f, (l_gt * l_gt) - (l_obs * l_obs));
                        float dz = std::sqrt(radicand);
                        
                        // Fingers curling down toward keys means +Z accumulation
                        phys_hand.landmarks[idx2].z_cm = phys_hand.landmarks[idx1].z_cm + dz;
                    }
                }
            }

            // Triangulate the visual wrist relative to Index and Middle 
            phys_hand.landmarks[0].z_cm = (phys_hand.landmarks[5].z_cm + phys_hand.landmarks[9].z_cm) / 2.0f;
            
            // Cascade Thumb independently 
            int thumb_chain[5] = {0, 1, 2, 3, 4};
            for(int j = 0; j < 4; ++j) {
                int idx1 = thumb_chain[j];
                int idx2 = thumb_chain[j+1];
                
                float dx = phys_hand.landmarks[idx2].x_cm - phys_hand.landmarks[idx1].x_cm;
                float dy = phys_hand.landmarks[idx2].y_cm - phys_hand.landmarks[idx1].y_cm;
                float l_obs = std::sqrt(dx*dx + dy*dy);
                float l_gt = profile->bone_lengths_cm.at(j); // Thumb connections are conveniently 0, 1, 2, 3
                
                float radicand = std::max(0.0f, (l_gt * l_gt) - (l_obs * l_obs));
                float dz = std::sqrt(radicand);
                phys_hand.landmarks[idx2].z_cm = phys_hand.landmarks[idx1].z_cm + dz;
            }
        }

        physical_hands.push_back(phys_hand);
    }

    return physical_hands;
}

} // namespace cv_keyboard