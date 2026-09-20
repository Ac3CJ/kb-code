// PLEASE NOTE THAT THE HANDEDNESS IS NOT CONSISTENT BETWEEN MODELS AND CAN CHANGE BASED ON THE MODEL USED. THIS IS A KNOWN ISSUE WITH MEDIAPIPE AND SHOULD BE HANDLED IN THE FUTURE.

#include "KinematicCalibrator.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace cv_keyboard {

void KinematicCalibrator::startCalibration() {
    is_calibrating_ = true;
    is_calibrated_ = false;
    frames_collected_ = 0;
    
    hand0_buffer_.clear();
    hand1_buffer_.clear();
    
    // Force both hands to be recalibrated to capture any size differences
    hand0_profile_.is_valid = false;
    hand1_profile_.is_valid = false;
    
    std::cout << "\n[KinematicCalibrator] Calibration Started.\n";
    std::cout << "Place ONE hand flat on the keyboard to avoid marker occlusion...\n";
}

bool KinematicCalibrator::updateCalibration(const std::vector<HandData>& hands, const KeyboardMap& kb_map, int frame_width, int frame_height) {
    if (!is_calibrating_ || !kb_map.hasValidTransform()) return false;
    
    bool captured_data = false;

    for (const auto& hand : hands) {
        // Only buffer data for a hand if it hasn't been successfully calibrated yet
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
        
        // System is fully calibrated only when both hands have valid profiles
        if (hand0_profile_.is_valid && hand1_profile_.is_valid) {
            is_calibrating_ = false;
            is_calibrated_ = true;
            std::cout << "[KinematicCalibrator] Both hands calibrated successfully. Calibration window closed.\n";
            return true; // Triggers the Mediator to save the JSON profile
        } else {
            // Reset the buffers to capture the remaining hand
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
    // Only process and validate the hand if it was visible for the majority of the 30-frame window
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
    for (int c = 0; c < NUM_HAND_CONNECTIONS; ++c) {
        int idx1 = HAND_CONNECTIONS[c][0];
        int idx2 = HAND_CONNECTIONS[c][1];
        
        cv::Point2f pt1 = median_landmarks_cm[idx1];
        cv::Point2f pt2 = median_landmarks_cm[idx2];
        
        float dist_cm = std::sqrt(std::pow(pt2.x - pt1.x, 2) + std::pow(pt2.y - pt1.y, 2));
        profile.bone_lengths_cm[c] = dist_cm;
    }
    
    profile.is_valid = true;
}

bool KinematicCalibrator::saveProfile(const std::string& filepath) const {
    // Ensure the profiles/ directory exists
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

    // Reset current state
    hand0_profile_ = HandProfile{};
    hand1_profile_ = HandProfile{};
    is_calibrated_ = false;

    while (std::getline(file, line)) {
        // 1. Determine which hand context we are in
        if (line.find("\"Hand1\"") != std::string::npos) {
            current_profile = &hand1_profile_;
            continue;
        } else if (line.find("\"Hand0\"") != std::string::npos) {
            current_profile = &hand0_profile_;
            continue;
        }

        if (!current_profile) continue;

        // 2. Parse Validity
        if (line.find("\"is_valid\": true") != std::string::npos) {
            current_profile->is_valid = true;
            continue;
        } else if (line.find("\"is_valid\": false") != std::string::npos) {
            current_profile->is_valid = false;
            continue;
        }

        // 3. Parse Bone Lengths (looking for integer keys in quotes, e.g., "0": 2.5)
        size_t colon_pos = line.find(':');
        size_t quote1 = line.find('"');
        if (colon_pos != std::string::npos && quote1 != std::string::npos && quote1 < colon_pos) {
            size_t quote2 = line.find('"', quote1 + 1);
            if (quote2 != std::string::npos && quote2 < colon_pos) {
                std::string key_str = line.substr(quote1 + 1, quote2 - quote1 - 1);
                
                // Ensure the key is purely numeric (our connection indices 0-22)
                bool is_numeric = !key_str.empty() && std::all_of(key_str.begin(), key_str.end(), ::isdigit);
                if (is_numeric) {
                    try {
                        int conn_idx = std::stoi(key_str);
                        std::string val_str = line.substr(colon_pos + 1);
                        
                        // Strip any trailing commas or spaces
                        val_str.erase(std::remove(val_str.begin(), val_str.end(), ','), val_str.end());
                        val_str.erase(std::remove_if(val_str.begin(), val_str.end(), ::isspace), val_str.end());
                        
                        current_profile->bone_lengths_cm[conn_idx] = std::stof(val_str);
                    } catch (...) {
                        // Safely ignore standard parsing faults
                    }
                }
            }
        }
    }

    // The system is only considered fully calibrated if both hands were loaded successfully
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

        // 1. Planar Projection & Velocity Conversion
        for (int i = 0; i < 21; ++i) {
            const auto& landmark = hand.landmarks[i];

            // Map current point to physical CM
            float px = landmark.x * frame_width;
            float py = landmark.y * frame_height;
            cv::Point2f pt_cm = kb_map.pixelToPhysical(px, py);
            
            phys_hand.landmarks[i].x_cm = pt_cm.x;
            phys_hand.landmarks[i].y_cm = pt_cm.y;

            // Extract true physical velocity by mapping the "previous" point
            float prev_px = (landmark.x - landmark.vx) * frame_width;
            float prev_py = (landmark.y - landmark.vy) * frame_height;
            cv::Point2f prev_pt_cm = kb_map.pixelToPhysical(prev_px, prev_py);
            
            phys_hand.landmarks[i].vx_cm = pt_cm.x - prev_pt_cm.x;
            phys_hand.landmarks[i].vy_cm = pt_cm.y - prev_pt_cm.y;
            phys_hand.landmarks[i].vz_cm = 0.0f; // To be calculated by temporal smoothers later if needed
        }

        // 2. Determine which profile to use
        const HandProfile* profile = nullptr;
        if (hand.handedness == 1 && hand0_profile_.is_valid) {
            profile = &hand0_profile_;
        } else if (hand.handedness == 2 && hand1_profile_.is_valid) {
            profile = &hand1_profile_;
        }

        // 3. Z-Estimation via Forward Kinematic Traversal
        if (profile) {
            // 3. Wrist Z-Estimation (The Dynamic Base Anchor)
            float z_wrist = 0.0f;
            
            // Connections 20, 21, 22 are your knuckle spans: (5->9), (9->13), (13->17)
            if (profile && profile->bone_lengths_cm.count(20) && profile->bone_lengths_cm.count(22)) {
                float gt_span = profile->bone_lengths_cm.at(20) + 
                                profile->bone_lengths_cm.at(21) + 
                                profile->bone_lengths_cm.at(22);
                
                float obs_span = 0.0f;
                for (int c = 20; c <= 22; ++c) {
                    int idx1 = HAND_CONNECTIONS[c][0];
                    int idx2 = HAND_CONNECTIONS[c][1];
                    float dx = phys_hand.landmarks[idx2].x_cm - phys_hand.landmarks[idx1].x_cm;
                    float dy = phys_hand.landmarks[idx2].y_cm - phys_hand.landmarks[idx1].y_cm;
                    obs_span += std::sqrt(dx * dx + dy * dy);
                }

                // Ratio > 1.0 means the hand's perspective footprint is larger (floating)
                float ratio = obs_span / gt_span;
                
                // Fetch the mathematically exact camera height dynamically
                float h_cam = kb_map.getCameraHeightCm(); 
                
                if (h_cam > 0.0f && ratio > 1.02f) { // 2% deadzone to prevent micro-jitter
                    // Derived from Perspective Projection: Z = H * (1 - Ratio) / Ratio
                    z_wrist = h_cam * (1.0f - ratio) / ratio; 
                }
                
                // Clamp to 0 to prevent the wrist from digging into the desk (+Z) if the hand rolls
                z_wrist = std::min(0.0f, z_wrist); 
            }

            // Anchor the wrist at the newly calculated floating height
            phys_hand.landmarks[0].z_cm = z_wrist;

            for (int c = 0; c < 20; ++c) {
                int idx1 = HAND_CONNECTIONS[c][0];
                int idx2 = HAND_CONNECTIONS[c][1];

                float dx = phys_hand.landmarks[idx2].x_cm - phys_hand.landmarks[idx1].x_cm;
                float dy = phys_hand.landmarks[idx2].y_cm - phys_hand.landmarks[idx1].y_cm;
                float l_obs = std::sqrt(dx * dx + dy * dy);
                
                float l_gt = profile->bone_lengths_cm.at(c);
                
                // Clamp to prevent NaN when perspective scaling makes l_obs > l_gt
                float radicand = std::max(0.0f, (l_gt * l_gt) - (l_obs * l_obs));
                float dz = std::sqrt(radicand);

                // Heuristic for Z-Down Axis: 
                // Fingers arch UP (negative Z) to the PIP, then curl DOWN (positive Z) to the desk.
                float sign = -1.0f; 
                
                // If this is a PIP->DIP or DIP->TIP connection, the finger is curling DOWN (+Z)
                if (c == 2 || c == 3 ||   // Thumb IP / Tip
                    c == 6 || c == 7 ||   // Index DIP / Tip
                    c == 10 || c == 11 || // Middle DIP / Tip
                    c == 14 || c == 15 || // Ring DIP / Tip
                    c == 18 || c == 19) { // Pinky DIP / Tip
                    sign = 1.0f;
                }

                phys_hand.landmarks[idx2].z_cm = phys_hand.landmarks[idx1].z_cm + (sign * dz);
            }
        } else {
            // Fallback if uncalibrated: flatten the hand to Z=0
            for (int i = 0; i < 21; ++i) {
                phys_hand.landmarks[i].z_cm = 0.0f;
            }
        }

        physical_hands.push_back(phys_hand);
    }

    return physical_hands;
}

} // namespace cv_keyboard