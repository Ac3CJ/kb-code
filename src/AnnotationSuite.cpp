#include "AnnotationSuite.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace fs = std::filesystem;

namespace cv_keyboard {

// ============================================================================
// INTERNAL HELPER FUNCTIONS
// ============================================================================

static std::string computeFastVideoHash(const fs::path& filepath) {
    if (!fs::exists(filepath)) return "MISSING";
    uint64_t hash = 14695981039346656037ULL;
    const uint64_t prime = 1099511628211ULL;

    auto mix = [&](uint64_t val) {
        for (int i = 0; i < 8; ++i) {
            hash ^= (val & 0xFF);
            hash *= prime;
            val >>= 8;
        }
    };

    std::error_code ec;
    uint64_t fsize = fs::file_size(filepath, ec);
    mix(fsize);

    // Hash first 1MB of the video stream for fast content integrity verification
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        const size_t buf_size = 1024 * 1024;
        std::vector<char> buffer(buf_size);
        file.read(buffer.data(), buf_size);
        std::streamsize bytes_read = file.gcount();
        for (std::streamsize i = 0; i < bytes_read; i += 64) { // Stride 64 for sub-ms speed
            hash ^= static_cast<uint8_t>(buffer[i]);
            hash *= prime;
        }
    }

    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << hash;
    return oss.str();
}

static std::string fingerIdToName(int finger_id) {
    const char* hands[] = {"H1 (Left)", "H2 (Right)"};
    const char* fingers[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
    int h = std::clamp(finger_id / 5, 0, 1);
    int f = std::clamp(finger_id % 5, 0, 4);
    return std::string(hands[h]) + " " + fingers[f];
}

static std::string sanitizeKeyString(const std::string& key) {
    // 1. Explicitly map known problematic symbols to safe text
    if (key == "'") return "Apostrophe";
    if (key == "\"") return "Double_Quote";
    if (key == "\\") return "Backslash";
    if (key == "[") return "L_Bracket";
    if (key == "]") return "R_Bracket";
    if (key == "{") return "L_Brace";
    if (key == "}") return "R_Brace";
    if (key == ",") return "Comma";
    if (key == ":") return "Colon";
    if (key == "/") return "Slash";
    if (key == "-") return "Hyphen";
    
    std::string safe = key;
    
    // 2. Strip any remaining rogue characters that break cv::FileStorage's lexer
    // Added: \\ (was missing in your bad_chars), :, /, -, space, and control characters (< 0x20)
    const std::string bad_chars = "\'\"\\/[]{},: -";
    safe.erase(std::remove_if(safe.begin(), safe.end(), 
        [&bad_chars](unsigned char c) { 
            return c < 0x20 || c == 0x7F || bad_chars.find(static_cast<char>(c)) != std::string::npos; 
        }), 
        safe.end());
        
    return safe.empty() ? "NONE" : safe;
}

// ============================================================================
// DATASET CONFIGURATION
// ============================================================================

void DatasetConfig::save(const fs::path& path) const {
    cv::FileStorage fs_out(path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    fs_out << "roi_width" << roi_width;
    fs_out << "roi_height" << roi_height;
    fs_out << "roi_offset_x" << roi_offset_x;
    fs_out << "roi_offset_y" << roi_offset_y;
    fs_out << "pos_frame_before" << pos_frame_before;
    fs_out << "pos_frame_after" << pos_frame_after;
    fs_out << "neg_approach_start" << neg_approach_start;
    fs_out << "neg_approach_end" << neg_approach_end;
    fs_out << "neg_release_start" << neg_release_start;
    fs_out << "neg_release_end" << neg_release_end;
    fs_out << "blur_size" << blur_size;
    fs_out << "shadow_l_thresh" << shadow_l_thresh;
    fs_out << "motion_thresh" << motion_thresh;
    fs_out << "shadow_dot_thresh" << shadow_dot_thresh;
    fs_out << "kinematic_vy_thresh" << kinematic_vy_thresh;
    fs_out << "merge_gap_frames" << merge_gap_frames;
}

bool DatasetConfig::load(const fs::path& path) {
    if (!fs::exists(path)) return false;
    cv::FileStorage fs_in(path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs_in.isOpened()) return false;
    if (!fs_in["roi_width"].empty()) fs_in["roi_width"] >> roi_width;
    if (!fs_in["roi_height"].empty()) fs_in["roi_height"] >> roi_height;
    if (!fs_in["roi_offset_x"].empty()) fs_in["roi_offset_x"] >> roi_offset_x;
    if (!fs_in["roi_offset_y"].empty()) fs_in["roi_offset_y"] >> roi_offset_y;
    if (!fs_in["blur_size"].empty()) fs_in["blur_size"] >> blur_size;
    if (!fs_in["shadow_l_thresh"].empty()) fs_in["shadow_l_thresh"] >> shadow_l_thresh;
    if (!fs_in["motion_thresh"].empty()) fs_in["motion_thresh"] >> motion_thresh;
    if (!fs_in["shadow_dot_thresh"].empty()) fs_in["shadow_dot_thresh"] >> shadow_dot_thresh;
    if (!fs_in["kinematic_vy_thresh"].empty()) fs_in["kinematic_vy_thresh"] >> kinematic_vy_thresh;
    if (!fs_in["merge_gap_frames"].empty()) fs_in["merge_gap_frames"] >> merge_gap_frames;
    return true;
}

// ============================================================================
// ANNOTATION SUITE LIFECYCLE
// ============================================================================

AnnotationSuite::AnnotationSuite(const std::string& input_path,
                                 const std::string& tracker_name,
                                 const std::string& processor_name)
    : tracker_name_(tracker_name),
      processor_name_(processor_name),
      mediator_(tracker_name, processor_name) {

    fs::path p(input_path);
    if (fs::is_directory(p)) {
        videos_dir_ = p;
    } else {
        videos_dir_ = p.parent_path().empty() ? fs::path(".") : p.parent_path();
        initial_video_file_ = p.filename().string();
    }

    annotations_dir_ = videos_dir_.parent_path().empty()
                           ? fs::path("annotations")
                           : videos_dir_.parent_path() / "annotations";
    exports_dir_ = videos_dir_.parent_path().empty()
                       ? fs::path("exports")
                       : videos_dir_.parent_path() / "exports";

    fs::create_directories(annotations_dir_);
    fs::create_directories(exports_dir_);

    config_path_ = exports_dir_ / "dataset_config.json";
    if (!config_.load(config_path_)) {
        config_.save(config_path_);
    }

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL | cv::WINDOW_GUI_NORMAL);
    cv::resizeWindow(window_name_, 1600, 920);
}

AnnotationSuite::~AnnotationSuite() {
    cv::destroyWindow(window_name_);
}

void AnnotationSuite::run() {
    if (!mediator_.init()) {
        std::cerr << "[AnnotationSuite] Failed to initialise Mediator.\n";
        return;
    }
    keyboard_map_.loadUKLayout();

    scanWorkspace();
    if (sessions_.empty()) {
        std::cerr << "[AnnotationSuite] No video files (.mp4, .avi, .mov, .mkv) found in: "
                  << videos_dir_ << "\n";
        return;
    }

    current_session_idx_ = 0;
    if (!initial_video_file_.empty()) {
        for (size_t i = 0; i < sessions_.size(); ++i) {
            if (sessions_[i].video_path.filename() == initial_video_file_) {
                current_session_idx_ = static_cast<int>(i);
                break;
            }
        }
    }

    loadSession(current_session_idx_);

    bool is_paused = true;
    bool force_redraw = true;

    printControlsHelp();

    while (true) {
        if (!is_paused && cap_.isOpened()) {
            cv::Mat next_frame;
            if (cap_.read(next_frame) && !next_frame.empty()) {
                current_frame_idx_++;
                raw_frame_ = next_frame;
                processOrFetchFrame(current_frame_idx_, raw_frame_);
                force_redraw = true;
            } else {
                is_paused = true;
            }
        }

        if (force_redraw && !raw_frame_.empty()) {
            renderWorkspaceUI(is_paused);
            force_redraw = false;
        }

        int key = cv::waitKeyEx(is_paused ? 30 : 15);
        if (key == -1) continue;

        if (awaiting_key_reclassify_) {
            handleKeyReclassificationInput(key);
            force_redraw = true;
            continue;
        }

        // ASCII & Extended Key Handling
        if (key == 27) { // ESC
            saveCurrentSessionJSON();
            break;
        } else if (key == ' ') {
            is_paused = !is_paused;
            force_redraw = true;
        } else if (key == 'd' || key == 'D' || key == 65363 || key == 2555904) {
            is_paused = true;
            seekToFrame(current_frame_idx_ + 1);
            force_redraw = true;
        } else if (key == 'a' || key == 'A' || key == 65361 || key == 2424832) {
            is_paused = true;
            seekToFrame(current_frame_idx_ - 1);
            force_redraw = true;
        } else if (key == 9 || key == 'q' || key == 'Q') {
            is_paused = true;
            jumpToCandidate(1);
            force_redraw = true;
        } else if (key == 'w' || key == 'W' || key == 65056) {
            is_paused = true;
            jumpToCandidate(-1);
            force_redraw = true;
        } else if (key == 13 || key == 10 || key == 'j' || key == 'J') {
            is_paused = true;
            labelCurrentFrame(1); // 1 = Explicit True Positive
            saveCurrentSessionJSON();
            seekToFrame(current_frame_idx_ + 1);
            force_redraw = true;
        } else if (key == 'x' || key == 'X' || key == 'k' || key == 'K') {
            is_paused = true;
            labelCurrentFrame(0); // 0 = Explicit False Positive
            saveCurrentSessionJSON();
            seekToFrame(current_frame_idx_ + 1);
            force_redraw = true;
        } else if (key == 'u' || key == 'U') {
            clearCurrentWindowAnchors();
            saveCurrentSessionJSON();
            force_redraw = true;
        } else if (key == '9' || key == 65362) {
            active_finger_id_ = (active_finger_id_ + 9) % 10;
            force_redraw = true;
        } else if (key == '0' || key == 65364) {
            active_finger_id_ = (active_finger_id_ + 1) % 10;
            force_redraw = true;
        } else if (key == 'c' || key == 'C') {
            awaiting_key_reclassify_ = true;
            status_banner_ = "KEY OVERRIDE: Press new target key (or SPACE / BACKSPACE / ESC)...";
            force_redraw = true;
        } else if (key == 'n' || key == 'N') {
            createManualCandidateAtCurrentFrame();
            saveCurrentSessionJSON();
            force_redraw = true;
        } else if (key == 'm' || key == 'M') {
            fifo_finger_mode_ = !fifo_finger_mode_;
            sortCandidateQueue();
            status_banner_ = std::string("Queue Order switched to: ") +
                             (fifo_finger_mode_ ? "FIFO BY FINGER" : "CHRONOLOGICAL");
            force_redraw = true;
        } else if (key == 'p' || key == 'P') {
            is_paused = true;
            runFullVideoCascadePreScan();
            saveCurrentSessionJSON();
            force_redraw = true;
        } else if (key == 'g' || key == 'G') {
            auto& sess = sessions_[current_session_idx_];
            sess.status = (sess.status == FileStatus::COMPLETED) ? FileStatus::IN_PROGRESS : FileStatus::COMPLETED;
            saveCurrentSessionJSON();
            status_banner_ = (sess.status == FileStatus::COMPLETED) ? "Session marked COMPLETED (Green)." : "Session marked IN_PROGRESS (Yellow).";
            force_redraw = true;
        } else if (key == 'l' || key == 'L') {
            cycleLightingTag();
            saveCurrentSessionJSON();
            force_redraw = true;
        } else if (key == 't' || key == 'T') {
            cycleSurfaceTag();
            saveCurrentSessionJSON();
            force_redraw = true;
        } else if (key == 's' || key == 'S') {
            saveCurrentSessionJSON();
            status_banner_ = "Saved annotation state to JSON.";
            force_redraw = true;
        } else if (key == ']' || key == 65366) {
            is_paused = true;
            saveCurrentSessionJSON();
            if (current_session_idx_ + 1 < static_cast<int>(sessions_.size())) {
                current_session_idx_++;
                loadSession(current_session_idx_);
            }
            force_redraw = true;
        } else if (key == '[' || key == 65365) {
            is_paused = true;
            saveCurrentSessionJSON();
            if (current_session_idx_ > 0) {
                current_session_idx_--;
                loadSession(current_session_idx_);
            }
            force_redraw = true;
        } else if (key == 'e' || key == 'E') {
            is_paused = true;
            saveCurrentSessionJSON();
            runBatchExporter();
            force_redraw = true;
        }
    }
}

// ============================================================================
// WORKSPACE & JSON PERSISTENCE
// ============================================================================

void AnnotationSuite::scanWorkspace() {
    sessions_.clear();
    std::vector<fs::path> video_files;
    const std::set<std::string> valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".MP4"};

    for (const auto& entry : fs::directory_iterator(videos_dir_)) {
        if (entry.is_regular_file() && valid_exts.count(entry.path().extension().string())) {
            video_files.push_back(entry.path());
        }
    }
    std::sort(video_files.begin(), video_files.end());

    for (const auto& vf : video_files) {
        VideoSessionMeta meta;
        meta.video_path = vf;
        meta.json_path = annotations_dir_ / (vf.stem().string() + "_labels.json");
        meta.video_hash = computeFastVideoHash(vf);

        std::string fname = vf.filename().string();
        meta.camera_type = (fname.rfind("w_", 0) == 0) ? "Laptop" : "Phone";

        if (fs::exists(meta.json_path)) {
            cv::FileStorage fs_in(meta.json_path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
            if (fs_in.isOpened()) {
                std::string saved_hash, saved_status;
                fs_in["video_hash"] >> saved_hash;
                fs_in["status"] >> saved_status;
                if (!fs_in["lighting"].empty()) fs_in["lighting"] >> meta.lighting;
                if (!fs_in["surface"].empty()) fs_in["surface"] >> meta.surface;
                if (!fs_in["expected_transcript"].empty()) fs_in["expected_transcript"] >> meta.expected_transcript;

                if (saved_hash != meta.video_hash) {
                    meta.status = FileStatus::DESYNCED;
                } else if (saved_status == "completed") {
                    meta.status = FileStatus::COMPLETED;
                } else {
                    meta.status = FileStatus::IN_PROGRESS;
                }
            }
        } else {
            meta.status = FileStatus::UNANNOTATED;
        }
        sessions_.push_back(meta);
    }
}

void AnnotationSuite::loadSession(int idx) {
    if (idx < 0 || idx >= static_cast<int>(sessions_.size())) return;
    auto& sess = sessions_[idx];

    cap_.release();
    cap_.open(sess.video_path.string());
    if (!cap_.isOpened()) {
        status_banner_ = "ERROR: Failed to open video " + sess.video_path.filename().string();
        return;
    }

    sess.fps = cap_.get(cv::CAP_PROP_FPS);
    if (sess.fps <= 0.0) sess.fps = 60.0;
    sess.frame_width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    sess.frame_height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    sess.total_frames = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));

    CameraSource src = (sess.camera_type == "Laptop") ? CameraSource::Laptop : CameraSource::Phone;
    mediator_.updateCameraIntrinsics(src);
    keyboard_map_.updateCameraIntrinsics(src);
    mediator_.resetClickState();

    hand_cache_.clear();
    candidates_.clear();
    prev_cascade_gray_.release();
    prev_finger_y_.clear();
    prev_finger_vy_.clear();
    highest_frame_processed_ = -1;
    current_frame_idx_ = 0;
    active_candidate_idx_ = -1;

    if (fs::exists(sess.json_path)) {
        loadSessionJSON(sess);
    }

    seekToFrame(0);
    if (!candidates_.empty()) {
        active_candidate_idx_ = 0;
        active_finger_id_ = candidates_[0].finger_id;
        seekToFrame(candidates_[0].peak_frame);
    }

    status_banner_ = "Loaded " + sess.video_path.filename().string() +
                     " | Press 'P' to Pre-Scan all frames or TAB/Q to step candidates.";
}

void AnnotationSuite::loadSessionJSON(VideoSessionMeta& sess) {
    cv::FileStorage fs_in(sess.json_path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs_in.isOpened()) return;

    cv::FileNode cw_node = fs_in["candidates"];
    if (cw_node.type() == cv::FileNode::SEQ) {
        int id_counter = 0;
        for (const auto& item : cw_node) {
            CandidateWindow cw;
            cw.id = id_counter++;
            item["finger_id"] >> cw.finger_id;
            item["start_frame"] >> cw.start_frame;
            item["end_frame"] >> cw.end_frame;
            item["peak_frame"] >> cw.peak_frame;
            int st = 0;
            item["state"] >> st;
            cw.state = static_cast<WindowState>(st);
            item["trig_shadow"] >> cw.trig_shadow;
            item["trig_kinematic"] >> cw.trig_kinematic;
            item["suggested_key"] >> cw.suggested_key;

            cv::FileNode anc_node = item["anchors"];
            if (anc_node.type() == cv::FileNode::SEQ) {
                for (const auto& a : anc_node) {
                    AnchorEvent ev;
                    a["impact_frame"] >> ev.impact_frame;
                    ev.label = 1;
                    if (!a["label"].empty()) a["label"] >> ev.label;
                    a["finger_id"] >> ev.finger_id;
                    a["px"] >> ev.px;
                    a["py"] >> ev.py;
                    a["roi_x"] >> ev.raw_roi.x;
                    a["roi_y"] >> ev.raw_roi.y;
                    a["roi_w"] >> ev.raw_roi.width;
                    a["roi_h"] >> ev.raw_roi.height;
                    a["key_id"] >> ev.key_id;
                    a["heuristic_triggered"] >> ev.heuristic_triggered;
                    cw.anchors.push_back(ev);
                }
            }
            candidates_.push_back(cw);
        }
    }
    sortCandidateQueue();
}

void AnnotationSuite::saveCurrentSessionJSON() {
    if (sessions_.empty()) return;
    auto& sess = sessions_[current_session_idx_];

    if (sess.status == FileStatus::UNANNOTATED && !candidates_.empty()) {
        sess.status = FileStatus::IN_PROGRESS;
    }
    sess.video_hash = computeFastVideoHash(sess.video_path);

    cv::FileStorage fs_out(sess.json_path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    fs_out << "video_file" << sess.video_path.filename().string();
    fs_out << "video_hash" << sess.video_hash;
    std::string status_str = (sess.status == FileStatus::COMPLETED) ? "completed" : "in_progress";
    fs_out << "status" << status_str;

    fs_out << "camera_type" << sess.camera_type;
    fs_out << "fps" << sess.fps;
    fs_out << "frame_width" << sess.frame_width;
    fs_out << "frame_height" << sess.frame_height;
    fs_out << "total_frames" << sess.total_frames;
    fs_out << "lighting" << sess.lighting;
    fs_out << "surface" << sess.surface;
    fs_out << "expected_transcript" << sess.expected_transcript;

    fs_out << "candidates" << "[";
    for (const auto& cw : candidates_) {
        fs_out << "{";
        fs_out << "finger_id" << cw.finger_id;
        fs_out << "start_frame" << cw.start_frame;
        fs_out << "end_frame" << cw.end_frame;
        fs_out << "peak_frame" << cw.peak_frame;
        fs_out << "state" << static_cast<int>(cw.state);
        fs_out << "trig_shadow" << cw.trig_shadow;
        fs_out << "trig_kinematic" << cw.trig_kinematic;
        
        // Sanitize the suggested key
        fs_out << "suggested_key" << sanitizeKeyString(cw.suggested_key); 
        
        fs_out << "anchors" << "[";
        for (const auto& ev : cw.anchors) {
            fs_out << "{";
            fs_out << "impact_frame" << ev.impact_frame;
            fs_out << "label" << ev.label;
            fs_out << "finger_id" << ev.finger_id;
            fs_out << "px" << ev.px;
            fs_out << "py" << ev.py;
            fs_out << "roi_x" << ev.raw_roi.x;
            fs_out << "roi_y" << ev.raw_roi.y;
            fs_out << "roi_w" << ev.raw_roi.width;
            fs_out << "roi_h" << ev.raw_roi.height;
            fs_out << "heuristic_triggered" << ev.heuristic_triggered;
            
            // Sanitize the anchor key
            fs_out << "key_id" << sanitizeKeyString(ev.key_id); 
            
            fs_out << "}";
        }
        fs_out << "]";
        fs_out << "}";
    }
    fs_out << "]";
}

// ============================================================================
// DUAL-GATE CASCADE CLASSIFIER
// ============================================================================

std::vector<HandData> AnnotationSuite::getSpatiallySortedHands(
    std::shared_ptr<const std::vector<HandData>> hands_ptr) {
    if (!hands_ptr || hands_ptr->empty()) return {};
    std::vector<HandData> sorted = *hands_ptr;
    std::sort(sorted.begin(), sorted.end(), [](const HandData& a, const HandData& b) {
        return a.landmarks[0].x < b.landmarks[0].x;
    });
    return sorted;
}

void AnnotationSuite::processOrFetchFrame(int frame_idx, const cv::Mat& frame) {
    keyboard_map_.updateTransform(frame);

    if (frame_idx > highest_frame_processed_) {
        mediator_.processFrame(frame);
        auto latest = mediator_.latestHands();
        if (latest) {
            hand_cache_[frame_idx] = std::make_shared<std::vector<HandData>>(*latest);
        } else {
            hand_cache_[frame_idx] = nullptr;
        }
        runDualGateCascadeStep(frame_idx, frame, hand_cache_[frame_idx]);
        highest_frame_processed_ = frame_idx;
    } else if (hand_cache_.count(frame_idx)) {
        mediator_.injectCachedHands(hand_cache_[frame_idx], frame);
    } else {
        mediator_.processFrame(frame);
        auto latest = mediator_.latestHands();
        hand_cache_[frame_idx] = latest ? std::make_shared<std::vector<HandData>>(*latest) : nullptr;
    }
}

void AnnotationSuite::runDualGateCascadeStep(int frame_idx,
                                             const cv::Mat& bgr_frame,
                                             std::shared_ptr<const std::vector<HandData>> hands_ptr) {
    if (bgr_frame.empty()) return;

    cv::Mat current_gray;
    cv::cvtColor(bgr_frame, current_gray, cv::COLOR_BGR2GRAY);

    if (prev_cascade_gray_.empty() || prev_cascade_gray_.size() != current_gray.size()) {
        current_gray.copyTo(prev_cascade_gray_);
        return;
    }

    auto sorted_hands = getSpatiallySortedHands(hands_ptr);
    int w = bgr_frame.cols;
    int h = bgr_frame.rows;

    for (size_t h_idx = 0; h_idx < std::min<size_t>(sorted_hands.size(), 2); ++h_idx) {
        const auto& hand = sorted_hands[h_idx];
        for (int f_idx = 0; f_idx < 5; ++f_idx) {
            int tip_idx = FINGER_TIP_INDICES[f_idx];
            int finger_id = static_cast<int>(h_idx * 5 + f_idx);
            const auto& tip = hand.landmarks[tip_idx];

            int px = static_cast<int>(tip.x * w);
            int py = static_cast<int>(tip.y * h);

            bool gate_kinematic = false;
            float curr_y = tip.y;
            if (prev_finger_y_.count(finger_id)) {
                float curr_vy = curr_y - prev_finger_y_[finger_id];
                float prev_vy = prev_finger_vy_[finger_id];
                if (prev_vy > config_.kinematic_vy_thresh && curr_vy <= 0.0015f) {
                    gate_kinematic = true;
                }
                prev_finger_vy_[finger_id] = curr_vy;
            }
            prev_finger_y_[finger_id] = curr_y;

            bool gate_shadow = false;
            int roi_x = px - (config_.roi_width / 2) + config_.roi_offset_x;
            int roi_y = py - (config_.roi_height / 2) + config_.roi_offset_y;
            cv::Rect roi_rect(roi_x, roi_y, config_.roi_width, config_.roi_height);
            roi_rect &= cv::Rect(0, 0, w, h);

            if (roi_rect.width == config_.roi_width && roi_rect.height == config_.roi_height) {
                cv::Mat curr_roi_gray = current_gray(roi_rect);
                cv::Mat prev_roi_gray = prev_cascade_gray_(roi_rect);
                cv::Mat curr_roi_bgr = bgr_frame(roi_rect);

                int bsize = (config_.blur_size % 2 == 1) ? config_.blur_size : config_.blur_size + 1;
                cv::Mat prev_blur, curr_blur;
                cv::GaussianBlur(prev_roi_gray, prev_blur, cv::Size(bsize, bsize), 0);
                cv::GaussianBlur(curr_roi_gray, curr_blur, cv::Size(bsize, bsize), 0);

                cv::Mat lab_roi, l_chan, shadow_mask, diff, diff_mask, combined_mask;
                cv::cvtColor(curr_roi_bgr, lab_roi, cv::COLOR_BGR2Lab);
                cv::extractChannel(lab_roi, l_chan, 0);
                cv::threshold(l_chan, shadow_mask, config_.shadow_l_thresh, 255, cv::THRESH_BINARY_INV);

                cv::absdiff(curr_blur, prev_blur, diff);
                cv::threshold(diff, diff_mask, config_.motion_thresh, 255, cv::THRESH_BINARY);
                cv::bitwise_and(shadow_mask, diff_mask, combined_mask);

                int active_px = cv::countNonZero(combined_mask);
                if (active_px > 5) {
                    cv::Mat flow;
                    cv::calcOpticalFlowFarneback(prev_blur, curr_blur, flow, 0.5, 2, 13, 2, 5, 1.1, 0);

                    cv::Point2f shadow_vel(0, 0), centroid(0, 0);
                    int count = 0;
                    for (int y = 0; y < combined_mask.rows; ++y) {
                        const uchar* m_row = combined_mask.ptr<uchar>(y);
                        const cv::Point2f* f_row = flow.ptr<cv::Point2f>(y);
                        for (int x = 0; x < combined_mask.cols; ++x) {
                            if (m_row[x] > 0) {
                                shadow_vel += f_row[x];
                                centroid += cv::Point2f(static_cast<float>(x), static_cast<float>(y));
                                count++;
                            }
                        }
                    }
                    if (count > 5) {
                        shadow_vel /= static_cast<float>(count);
                        centroid /= static_cast<float>(count);
                        cv::Point2f finger_pos(
                            (config_.roi_width / 2.0f) - config_.roi_offset_x,
                            (config_.roi_height / 2.0f) - config_.roi_offset_y);
                        cv::Point2f D = centroid - finger_pos;
                        float dot = (D.x * shadow_vel.x) + (D.y * shadow_vel.y);
                        if (dot < config_.shadow_dot_thresh) {
                            gate_shadow = true;
                        }
                    }
                }
            }

            if (gate_kinematic || gate_shadow) {
                std::string guessed_key = guessKeyAtPixel(px, py);
                registerCascadeTrigger(frame_idx, finger_id, gate_shadow, gate_kinematic, guessed_key);
            }
        }
    }

    current_gray.copyTo(prev_cascade_gray_);
}

std::string AnnotationSuite::guessKeyAtPixel(int px, int py) const {
    if (!keyboard_map_.hasValidTransform()) return "UNKNOWN";
    cv::Point2f phys = keyboard_map_.pixelToPhysical(static_cast<float>(px), static_cast<float>(py));
    std::string k = keyboard_map_.getKeyAt(phys.x, phys.y);
    return k.empty() ? "NONE" : k;
}

void AnnotationSuite::registerCascadeTrigger(int frame_idx,
                                             int finger_id,
                                             bool trig_shadow,
                                             bool trig_kinematic,
                                             const std::string& suggested_key) {
    int total_f = sessions_[current_session_idx_].total_frames;
    int w_start = std::max(0, frame_idx - 8);
    int w_end = std::min(std::max(0, total_f - 1), frame_idx + 8);

    for (auto& cw : candidates_) {
        if (cw.finger_id == finger_id &&
            w_start <= (cw.end_frame + config_.merge_gap_frames) &&
            w_end >= (cw.start_frame - config_.merge_gap_frames)) {
            cw.start_frame = std::min(cw.start_frame, w_start);
            cw.end_frame = std::max(cw.end_frame, w_end);
            cw.trig_shadow = cw.trig_shadow || trig_shadow;
            cw.trig_kinematic = cw.trig_kinematic || trig_kinematic;
            if (cw.suggested_key == "NONE" || cw.suggested_key == "UNKNOWN") {
                cw.suggested_key = suggested_key;
            }
            return;
        }
    }

    CandidateWindow nw;
    nw.id = static_cast<int>(candidates_.size());
    nw.finger_id = finger_id;
    nw.start_frame = w_start;
    nw.end_frame = w_end;
    nw.peak_frame = frame_idx;
    nw.trig_shadow = trig_shadow;
    nw.trig_kinematic = trig_kinematic;
    nw.suggested_key = suggested_key;
    nw.state = WindowState::PENDING;
    candidates_.push_back(nw);
    sortCandidateQueue();
}

void AnnotationSuite::sortCandidateQueue() {
    if (fifo_finger_mode_) {
        std::sort(candidates_.begin(), candidates_.end(),
                  [](const CandidateWindow& a, const CandidateWindow& b) {
                      if (a.finger_id != b.finger_id) return a.finger_id < b.finger_id;
                      return a.start_frame < b.start_frame;
                  });
    } else {
        std::sort(candidates_.begin(), candidates_.end(),
                  [](const CandidateWindow& a, const CandidateWindow& b) {
                      if (a.start_frame != b.start_frame) return a.start_frame < b.start_frame;
                      return a.finger_id < b.finger_id;
                  });
    }
    for (size_t i = 0; i < candidates_.size(); ++i) {
        candidates_[i].id = static_cast<int>(i);
    }
}

void AnnotationSuite::runFullVideoCascadePreScan() {
    if (!cap_.isOpened()) return;
    int total_f = sessions_[current_session_idx_].total_frames;
    status_banner_ = "Running Dual-Gate Cascade Pre-Scan across all frames...";
    renderWorkspaceUI(true);
    cv::waitKey(1);

    for (int f = highest_frame_processed_ + 1; f < total_f; ++f) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, f);
        cv::Mat frame;
        if (!cap_.read(frame) || frame.empty()) break;
        processOrFetchFrame(f, frame);

        if (f % 15 == 0) {
            current_frame_idx_ = f;
            raw_frame_ = frame;
            status_banner_ = "Pre-Scanning Frame " + std::to_string(f) + " / " +
                             std::to_string(total_f) + " | Candidates found: " +
                             std::to_string(candidates_.size());
            renderWorkspaceUI(true);
            if (cv::waitKey(1) == 27) break;
        }
    }

    if (!candidates_.empty()) {
        active_candidate_idx_ = 0;
        active_finger_id_ = candidates_[0].finger_id;
        seekToFrame(candidates_[0].peak_frame);
    }
    status_banner_ = "Pre-Scan Complete! Total Merged Candidate Windows: " +
                     std::to_string(candidates_.size());
}

// ============================================================================
// NAVIGATION & LABELING ACTIONS
// ============================================================================

void AnnotationSuite::seekToFrame(int target_frame) {
    if (!cap_.isOpened()) return;
    int total_f = sessions_[current_session_idx_].total_frames;
    target_frame = std::clamp(target_frame, 0, std::max(0, total_f - 1));

    current_frame_idx_ = target_frame;
    cap_.set(cv::CAP_PROP_POS_FRAMES, current_frame_idx_);
    cap_.read(raw_frame_);
    if (!raw_frame_.empty()) {
        processOrFetchFrame(current_frame_idx_, raw_frame_);
    }
}

void AnnotationSuite::jumpToCandidate(int direction) {
    if (candidates_.empty()) {
        status_banner_ = "No candidate windows yet. Press 'P' to Pre-Scan or 'N' to add manually.";
        return;
    }
    int n = static_cast<int>(candidates_.size());
    if (active_candidate_idx_ < 0) {
        active_candidate_idx_ = 0;
    } else {
        active_candidate_idx_ = (active_candidate_idx_ + direction + n) % n;
    }
    const auto& cw = candidates_[active_candidate_idx_];
    active_finger_id_ = cw.finger_id;
    int target = cw.anchors.empty() ? cw.peak_frame : cw.anchors.back().impact_frame;
    seekToFrame(target);

    status_banner_ = "Candidate #" + std::to_string(active_candidate_idx_ + 1) + "/" +
                     std::to_string(n) + " [" + fingerIdToName(cw.finger_id) +
                     "] Frames " + std::to_string(cw.start_frame) + "-" +
                     std::to_string(cw.end_frame);
}

bool AnnotationSuite::getFingerPixelCoords(int frame_idx, int finger_id, int& out_px, int& out_py) const {
    if (!hand_cache_.count(frame_idx) || !hand_cache_.at(frame_idx)) return false;
    auto sorted_hands = getSpatiallySortedHands(hand_cache_.at(frame_idx));
    size_t h_idx = static_cast<size_t>(finger_id / 5);
    int f_idx = finger_id % 5;
    if (h_idx >= sorted_hands.size()) return false;

    int tip_idx = FINGER_TIP_INDICES[f_idx];
    const auto& tip = sorted_hands[h_idx].landmarks[tip_idx];
    const auto& sess = sessions_[current_session_idx_];
    out_px = static_cast<int>(tip.x * sess.frame_width);
    out_py = static_cast<int>(tip.y * sess.frame_height);
    return true;
}

void AnnotationSuite::labelCurrentFrame(int label_val) {
    int px = -1, py = -1;
    
    // Do not abort! Just check if it succeeded.
    bool tracker_found = getFingerPixelCoords(current_frame_idx_, active_finger_id_, px, py);

    AnchorEvent ev;
    ev.impact_frame = current_frame_idx_;
    ev.label = label_val;
    ev.finger_id = active_finger_id_;
    ev.px = px;
    ev.py = py;

    if (tracker_found) {
        ev.raw_roi = cv::Rect(
            px - (config_.roi_width / 2) + config_.roi_offset_x,
            py - (config_.roi_height / 2) + config_.roi_offset_y,
            config_.roi_width,
            config_.roi_height);
        ev.key_id = guessKeyAtPixel(px, py);
    } else {
        // Tracker failed due to occlusion/edge case. Save as placeholder.
        ev.raw_roi = cv::Rect(0, 0, 0, 0); 
        ev.key_id = "TRACKER_FAILED";
    }

    int matched_idx = -1;
    if (active_candidate_idx_ >= 0 &&
        active_candidate_idx_ < static_cast<int>(candidates_.size())) {
        auto& cw = candidates_[active_candidate_idx_];
        if (cw.finger_id == active_finger_id_ &&
            current_frame_idx_ >= cw.start_frame - 5 &&
            current_frame_idx_ <= cw.end_frame + 5) {
            matched_idx = active_candidate_idx_;
        }
    }

    if (matched_idx == -1) {
        createManualCandidateAtCurrentFrame();
        matched_idx = active_candidate_idx_;
        ev.heuristic_triggered = false;
    }

    if (matched_idx >= 0) {
        auto& cw = candidates_[matched_idx];
        if (ev.key_id == "UNKNOWN" || ev.key_id == "NONE") {
            ev.key_id = cw.suggested_key;
        }
        
        cw.anchors.erase(
            std::remove_if(cw.anchors.begin(), cw.anchors.end(),
                           [&](const AnchorEvent& a) { return a.impact_frame == ev.impact_frame; }),
            cw.anchors.end());
            
        cw.anchors.push_back(ev);
        cw.state = WindowState::ACCEPTED;

        std::string type_str = (label_val == 1) ? "TP (Touch)" : "FP (Hover)";
        
        if (tracker_found) {
            status_banner_ = "FRAME " + std::to_string(ev.impact_frame) + " LABELED " + type_str + 
                             " [" + fingerIdToName(ev.finger_id) + "] Key: " + ev.key_id;
        } else {
            status_banner_ = "FRAME " + std::to_string(ev.impact_frame) + " LABELED " + type_str + 
                             " (TRACKER FAILED - Coords saved as placeholders)";
        }
    }
}

void AnnotationSuite::clearCurrentWindowAnchors() {
    if (active_candidate_idx_ < 0 ||
        active_candidate_idx_ >= static_cast<int>(candidates_.size())) {
        return;
    }
    auto& cw = candidates_[active_candidate_idx_];
    cw.anchors.clear();
    cw.state = WindowState::PENDING;
    status_banner_ = "Cleared anchors in current window.";
}

void AnnotationSuite::createManualCandidateAtCurrentFrame() {
    int total_f = sessions_[current_session_idx_].total_frames;
    CandidateWindow nw;
    nw.id = static_cast<int>(candidates_.size());
    nw.finger_id = active_finger_id_;
    nw.start_frame = std::max(0, current_frame_idx_ - 8);
    nw.end_frame = std::min(std::max(0, total_f - 1), current_frame_idx_ + 8);
    nw.peak_frame = current_frame_idx_;
    nw.trig_shadow = false;
    nw.trig_kinematic = false;
    int px = 0, py = 0;
    if (getFingerPixelCoords(current_frame_idx_, active_finger_id_, px, py)) {
        nw.suggested_key = guessKeyAtPixel(px, py);
    } else {
        nw.suggested_key = "NONE";
    }
    nw.state = WindowState::PENDING;
    candidates_.push_back(nw);
    sortCandidateQueue();

    for (size_t i = 0; i < candidates_.size(); ++i) {
        if (candidates_[i].finger_id == active_finger_id_ &&
            candidates_[i].peak_frame == current_frame_idx_) {
            active_candidate_idx_ = static_cast<int>(i);
            break;
        }
    }
    status_banner_ = "Created manual candidate window at F" + std::to_string(current_frame_idx_);
}

void AnnotationSuite::handleKeyReclassificationInput(int key) {
    awaiting_key_reclassify_ = false;
    if (key == 27) {
        status_banner_ = "Key override cancelled.";
        return;
    }
    std::string new_key;
    if (key == ' ') new_key = "SPACE";
    else if (key == 8 || key == 127 || key == 65288) new_key = "BACKSPACE";
    else if (key == 13 || key == 10) new_key = "ENTER";
    else if (key >= 33 && key <= 126) {
        char c = static_cast<char>( std::toupper(key) );
        new_key = std::string(1, c);
    } else {
        status_banner_ = "Unrecognised keycode for override.";
        return;
    }

    if (active_candidate_idx_ >= 0 &&
        active_candidate_idx_ < static_cast<int>(candidates_.size())) {
        auto& cw = candidates_[active_candidate_idx_];
        cw.suggested_key = new_key;
        for (auto& a : cw.anchors) {
            a.key_id = new_key;
        }
        saveCurrentSessionJSON();
        status_banner_ = "Updated Target Key to: '" + new_key + "'";
    }
}

void AnnotationSuite::cycleLightingTag() {
    const std::vector<std::string> opts = {"overhead", "diffuse", "side_lit", "low_light", "dynamic_exposure"};
    auto& cur = sessions_[current_session_idx_].lighting;
    auto it = std::find(opts.begin(), opts.end(), cur);
    int idx = (it == opts.end()) ? 0 : static_cast<int>(std::distance(opts.begin(), it) + 1) % opts.size();
    cur = opts[idx];
    status_banner_ = "Lighting tag set to: " + cur;
}

void AnnotationSuite::cycleSurfaceTag() {
    const std::vector<std::string> opts = {"matte_light", "matte_dark", "glossy", "wood_grain"};
    auto& cur = sessions_[current_session_idx_].surface;
    auto it = std::find(opts.begin(), opts.end(), cur);
    int idx = (it == opts.end()) ? 0 : static_cast<int>(std::distance(opts.begin(), it) + 1) % opts.size();
    cur = opts[idx];
    status_banner_ = "Surface tag set to: " + cur;
}

// ============================================================================
// BATCH EXPORTER
// ============================================================================

void AnnotationSuite::runBatchExporter() {
    config_.save(config_path_);
    int exported_sessions = 0;
    size_t total_samples_exported = 0;

    for (const auto& sess : sessions_) {
        if (!fs::exists(sess.json_path)) continue;

        VideoSessionMeta temp_sess = sess;
        std::vector<CandidateWindow> sess_candidates;
        cv::FileStorage fs_in(sess.json_path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!fs_in.isOpened()) continue;

        cv::FileNode cw_node = fs_in["candidates"];
        if (cw_node.type() != cv::FileNode::SEQ) continue;

        struct FrameTarget {
                int label = 0;
                int finger_id = 0;
                std::string key_id = "NONE";
                bool is_explicit = false; // Protects manual overrides
            };
        std::map<int, std::map<int, FrameTarget>> frame_schedule;

        for (const auto& item : cw_node) {
            int finger_id = 0;
            item["finger_id"] >> finger_id;
            
            cv::FileNode anc_node = item["anchors"];
            if (anc_node.type() != cv::FileNode::SEQ) continue;

            // Phase 1: Expand Procedural Data
            for (const auto& a : anc_node) {
                int t0 = 0, label_val = 1;
                std::string k_id;
                a["impact_frame"] >> t0;
                if (!a["label"].empty()) a["label"] >> label_val;
                a["key_id"] >> k_id;

                if (label_val == 1) {
                    for (int d = config_.neg_approach_start; d <= config_.neg_approach_end; ++d) {
                        if (!frame_schedule[t0 + d].count(finger_id)) {
                            frame_schedule[t0 + d][finger_id] = {0, finger_id, k_id, false};
                        }
                    }
                    for (int d = config_.neg_release_start; d <= config_.neg_release_end; ++d) {
                        if (!frame_schedule[t0 + d].count(finger_id)) {
                            frame_schedule[t0 + d][finger_id] = {0, finger_id, k_id, false};
                        }
                    }
                    for (int d = -config_.pos_frame_before; d <= config_.pos_frame_after; ++d) {
                        if (!frame_schedule[t0 + d].count(finger_id) || !frame_schedule[t0 + d][finger_id].is_explicit) {
                            frame_schedule[t0 + d][finger_id] = {1, finger_id, k_id, false};
                        }
                    }
                }
            }

            // Phase 2: Explicit Frames Overwrite Everything
            for (const auto& a : anc_node) {
                int t0 = 0, label_val = 1;
                std::string k_id;
                a["impact_frame"] >> t0;
                if (!a["label"].empty()) a["label"] >> label_val;
                a["key_id"] >> k_id;

                frame_schedule[t0][finger_id] = {label_val, finger_id, k_id, true};
            }
        }

        if (frame_schedule.empty()) continue;

        cv::VideoCapture vcap(sess.video_path.string());
        if (!vcap.isOpened()) continue;

        fs::path bin_path = exports_dir_ / (sess.video_path.stem().string() + ".bin");
        std::ofstream bin_out(bin_path, std::ios::binary);
        if (!bin_out.is_open()) continue;

        BinaryCropHeader hdr;
        hdr.roi_w = static_cast<uint32_t>(config_.roi_width);
        hdr.roi_h = static_cast<uint32_t>(config_.roi_height);
        hdr.num_samples = 0;
        bin_out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        Mediator export_mediator(tracker_name_, processor_name_);
        export_mediator.init();
        uint32_t written_count = 0;

        for (const auto& [f_idx, finger_map] : frame_schedule) {
            vcap.set(cv::CAP_PROP_POS_FRAMES, f_idx);
            cv::Mat frame;
            if (!vcap.read(frame) || frame.empty()) continue;

            export_mediator.processFrame(frame);
            auto sorted_hands = getSpatiallySortedHands(export_mediator.latestHands());
            if (sorted_hands.empty()) continue;

            for (const auto& [f_id, target] : finger_map) {
                size_t h_idx = static_cast<size_t>(f_id / 5);
                int tip_slot = f_id % 5;
                if (h_idx >= sorted_hands.size()) continue;

                const auto& tip = sorted_hands[h_idx].landmarks[FINGER_TIP_INDICES[tip_slot]];
                int px = static_cast<int>(tip.x * frame.cols);
                int py = static_cast<int>(tip.y * frame.rows);

                int rx = px - (config_.roi_width / 2) + config_.roi_offset_x;
                int ry = py - (config_.roi_height / 2) + config_.roi_offset_y;
                cv::Rect roi_rect(rx, ry, config_.roi_width, config_.roi_height);
                cv::Rect clamped = roi_rect & cv::Rect(0, 0, frame.cols, frame.rows);
                if (clamped.width != config_.roi_width || clamped.height != config_.roi_height) {
                    continue;
                }

                cv::Mat crop = frame(clamped).clone();

                BinarySampleMeta smeta{};
                smeta.frame_index = f_idx;
                smeta.label = target.label;
                smeta.finger_id = f_id;
                smeta.roi_x = clamped.x;
                smeta.roi_y = clamped.y;
                smeta.roi_w = clamped.width;
                smeta.roi_h = clamped.height;
                std::strncpy(smeta.key_id, target.key_id.c_str(), sizeof(smeta.key_id) - 1);

                bin_out.write(reinterpret_cast<const char*>(&smeta), sizeof(smeta));
                bin_out.write(reinterpret_cast<const char*>(crop.data),
                              crop.total() * crop.elemSize());
                written_count++;
            }
        }

        hdr.num_samples = written_count;
        bin_out.seekp(0, std::ios::beg);
        bin_out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        bin_out.close();

        exported_sessions++;
        total_samples_exported += written_count;
    }

    if (fs::exists("export_to_hdf5.py")) {
        std::string cmd = "python3 export_to_hdf5.py \"" + exports_dir_.string() + "\" \"" +
                          annotations_dir_.string() + "\"";
        int ret = std::system(cmd.c_str());
        (void)ret;
    }

    status_banner_ = "BATCH EXPORT COMPLETE: " + std::to_string(exported_sessions) +
                     " session(s), " + std::to_string(total_samples_exported) +
                     " ROI crops written to " + exports_dir_.string();
}

// ============================================================================
// UI RENDERING
// ============================================================================

void AnnotationSuite::renderWorkspaceUI(bool is_paused) {
    const int W = 1600;
    const int H = 920;
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(24, 24, 28));

    const int SIDEBAR_W = 310;
    const int RIGHT_PANEL_W = 380;
    const int TIMELINE_H = 175;
    const int VIEWPORT_W = W - SIDEBAR_W - RIGHT_PANEL_W;
    const int VIEWPORT_H = H - TIMELINE_H;

    drawSidebar(canvas, cv::Rect(0, 0, SIDEBAR_W, H - TIMELINE_H));
    drawCenterViewport(canvas, cv::Rect(SIDEBAR_W, 0, VIEWPORT_W, VIEWPORT_H), is_paused);
    drawRightInspector(canvas, cv::Rect(SIDEBAR_W + VIEWPORT_W, 0, RIGHT_PANEL_W, VIEWPORT_H));
    drawBottomTimeline(canvas, cv::Rect(0, VIEWPORT_H, W, TIMELINE_H));

    cv::imshow(window_name_, canvas);
}

void AnnotationSuite::drawSidebar(cv::Mat& canvas, const cv::Rect& bounds) {
    cv::rectangle(canvas, bounds, cv::Scalar(32, 33, 38), -1);
    cv::line(canvas, cv::Point(bounds.width - 1, 0),
             cv::Point(bounds.width - 1, bounds.height), cv::Scalar(60, 60, 70), 1);

    cv::putText(canvas, "WORKSPACE FILES", cv::Point(15, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(220, 220, 230), 2);
    cv::putText(canvas, "[ / ] : Prev / Next File", cv::Point(15, 52),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(150, 150, 165), 1);

    int y = 80;
    for (size_t i = 0; i < sessions_.size() && y < bounds.height - 140; ++i) {
        const auto& s = sessions_[i];
        bool selected = (static_cast<int>(i) == current_session_idx_);

        if (selected) {
            cv::rectangle(canvas, cv::Rect(6, y - 18, bounds.width - 12, 28),
                          cv::Scalar(55, 65, 90), -1);
        }

        cv::Scalar badge_color(60, 60, 230);
        std::string badge_txt = "NEW";
        if (s.status == FileStatus::IN_PROGRESS) {
            badge_color = cv::Scalar(0, 210, 255);
            badge_txt = "WIP";
        } else if (s.status == FileStatus::COMPLETED) {
            badge_color = cv::Scalar(70, 210, 70);
            badge_txt = "OK";
        } else if (s.status == FileStatus::DESYNCED) {
            badge_color = cv::Scalar(0, 130, 255);
            badge_txt = "HASH!";
        }

        cv::circle(canvas, cv::Point(20, y - 4), 6, badge_color, -1);
        std::string label = s.video_path.filename().string();
        if (label.size() > 24) label = label.substr(0, 21) + "...";

        cv::putText(canvas, label, cv::Point(35, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.43,
                    selected ? cv::Scalar(255, 255, 255) : cv::Scalar(185, 185, 195), 1);
        cv::putText(canvas, badge_txt, cv::Point(bounds.width - 52, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.36, badge_color, 1);
        y += 32;
    }

    int fy = bounds.height - 125;
    cv::line(canvas, cv::Point(10, fy - 15), cv::Point(bounds.width - 10, fy - 15),
             cv::Scalar(60, 60, 70), 1);
    const auto& cur = sessions_[current_session_idx_];
    cv::putText(canvas, "SESSION METADATA", cv::Point(15, fy),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 215, 255), 1);
    cv::putText(canvas, "Hash : " + cur.video_hash.substr(0, 12), cv::Point(15, fy + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 190), 1);
    cv::putText(canvas, "Cam  : " + cur.camera_type + " (" + std::to_string(static_cast<int>(cur.fps)) + " FPS)",
                cv::Point(15, fy + 42), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 190), 1);
    cv::putText(canvas, "Light: " + cur.lighting + " (Press L)", cv::Point(15, fy + 62),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 190), 1);
    cv::putText(canvas, "Surf : " + cur.surface + " (Press T)", cv::Point(15, fy + 82),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 190), 1);
}

void AnnotationSuite::drawCenterViewport(cv::Mat& canvas, const cv::Rect& bounds, bool is_paused) {
    cv::Mat display_frame;
    mediator_.renderOverlay(raw_frame_, display_frame);

    int px = 0, py = 0;
    if (getFingerPixelCoords(current_frame_idx_, active_finger_id_, px, py)) {
        int rx = px - (config_.roi_width / 2) + config_.roi_offset_x;
        int ry = py - (config_.roi_height / 2) + config_.roi_offset_y;
        cv::Rect roi_rect(rx, ry, config_.roi_width, config_.roi_height);
        cv::rectangle(display_frame, roi_rect, cv::Scalar(0, 255, 255), 2);
        cv::circle(display_frame, cv::Point(px, py), 6, cv::Scalar(0, 0, 255), -1);
        cv::putText(display_frame, fingerIdToName(active_finger_id_),
                    cv::Point(rx, std::max(20, ry - 8)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2);
    }

    double scale = std::min(static_cast<double>(bounds.width - 20) / display_frame.cols,
                            static_cast<double>(bounds.height - 60) / display_frame.rows);
    int rw = static_cast<int>(display_frame.cols * scale);
    int rh = static_cast<int>(display_frame.rows * scale);
    int ox = bounds.x + (bounds.width - rw) / 2;
    int oy = bounds.y + 45 + (bounds.height - 55 - rh) / 2;

    cv::Mat resized;
    cv::resize(display_frame, resized, cv::Size(rw, rh));
    resized.copyTo(canvas(cv::Rect(ox, oy, rw, rh)));

    const auto& sess = sessions_[current_session_idx_];
    std::string hdr = "Frame: " + std::to_string(current_frame_idx_) + " / " +
                      std::to_string(sess.total_frames) +
                      (is_paused ? " [PAUSED]" : " [PLAYING]") +
                      "   |   Active Finger: " + fingerIdToName(active_finger_id_) +
                      " (Keys 9 / 0)";
    cv::putText(canvas, hdr, cv::Point(bounds.x + 15, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(240, 240, 250), 2);
}

void AnnotationSuite::drawRightInspector(cv::Mat& canvas, const cv::Rect& bounds) {
    cv::rectangle(canvas, bounds, cv::Scalar(32, 33, 38), -1);
    cv::line(canvas, cv::Point(bounds.x, 0),
             cv::Point(bounds.x, bounds.height), cv::Scalar(60, 60, 70), 1);

    int x0 = bounds.x + 15;
    cv::putText(canvas, "150x75 ROI INSPECTOR", cv::Point(x0, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(220, 220, 230), 2);

    int scale = 2;
    int preview_w = config_.roi_width * scale;
    int preview_h = config_.roi_height * scale;
    cv::Mat roi_preview = cv::Mat::zeros(preview_h, preview_w, CV_8UC3);

    int px = 0, py = 0;
    std::string live_key = "NONE";
    if (getFingerPixelCoords(current_frame_idx_, active_finger_id_, px, py)) {
        live_key = guessKeyAtPixel(px, py);
        int rx = px - (config_.roi_width / 2) + config_.roi_offset_x;
        int ry = py - (config_.roi_height / 2) + config_.roi_offset_y;
        cv::Rect r(rx, ry, config_.roi_width, config_.roi_height);
        cv::Rect clamped = r & cv::Rect(0, 0, raw_frame_.cols, raw_frame_.rows);
        if (clamped.width == config_.roi_width && clamped.height == config_.roi_height) {
            cv::resize(raw_frame_(clamped), roi_preview,
                       cv::Size(preview_w, preview_h), 0, 0, cv::INTER_NEAREST);
            int cx = static_cast<int>(((config_.roi_width / 2.0f) - config_.roi_offset_x) * scale);
            int cy = static_cast<int>(((config_.roi_height / 2.0f) - config_.roi_offset_y) * scale);
            cv::drawMarker(roi_preview, cv::Point(cx, cy),
                           cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 16, 2);
        } else {
            cv::putText(roi_preview, "ROI OUT OF BOUNDS", cv::Point(25, preview_h / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }
    } else {
        cv::putText(roi_preview, "FINGER NOT VISIBLE", cv::Point(25, preview_h / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }

    roi_preview.copyTo(canvas(cv::Rect(x0, 45, preview_w, preview_h)));
    cv::rectangle(canvas, cv::Rect(x0, 45, preview_w, preview_h), cv::Scalar(90, 90, 105), 1);

    int y = 45 + preview_h + 28;
    cv::putText(canvas, "Spatial Key Guess: [" + live_key + "] (Press C to override)",
                cv::Point(x0, y), cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(0, 230, 255), 1);

    y += 32;
    cv::line(canvas, cv::Point(x0, y - 14), cv::Point(bounds.x + bounds.width - 15, y - 14),
             cv::Scalar(60, 60, 70), 1);
    cv::putText(canvas, "ACTIVE CANDIDATE WINDOW", cv::Point(x0, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(220, 220, 230), 1);

    y += 24;
    if (active_candidate_idx_ >= 0 &&
        active_candidate_idx_ < static_cast<int>(candidates_.size())) {
        const auto& cw = candidates_[active_candidate_idx_];
        std::string gates = std::string(cw.trig_shadow ? "[Shadow] " : "") +
                            (cw.trig_kinematic ? "[Kinematic]" : "");
        if (gates.empty()) gates = "[Manual]";

        std::string st_str = (cw.state == WindowState::ACCEPTED)   ? "ACCEPTED (Green)"
                             : (cw.state == WindowState::REJECTED) ? "REJECTED (Red)"
                                                                   : "PENDING (Yellow)";

        cv::putText(canvas, "Index : #" + std::to_string(active_candidate_idx_ + 1) +
                            " / " + std::to_string(candidates_.size()),
                    cv::Point(x0, y), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(190, 190, 200), 1);
        cv::putText(canvas, "Range : F" + std::to_string(cw.start_frame) + " -> F" +
                            std::to_string(cw.end_frame) + " (Peak F" + std::to_string(cw.peak_frame) + ")",
                    cv::Point(x0, y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(190, 190, 200), 1);
        cv::putText(canvas, "Gates : " + gates,
                    cv::Point(x0, y + 40), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(190, 190, 200), 1);
        cv::putText(canvas, "State : " + st_str,
                    cv::Point(x0, y + 60), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(190, 190, 200), 1);
        cv::putText(canvas, "Key   : " + cw.suggested_key + " | Anchors: " +
                            std::to_string(cw.anchors.size()),
                    cv::Point(x0, y + 80), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(0, 255, 180), 1);
    } else {
        cv::putText(canvas, "No active candidate selected.",
                    cv::Point(x0, y), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(140, 140, 150), 1);
    }

    y += 115;
    cv::line(canvas, cv::Point(x0, y - 14), cv::Point(bounds.x + bounds.width - 15, y - 14),
             cv::Scalar(60, 60, 70), 1);
    cv::putText(canvas, "SPEED KEYBINDS (NO MOUSE)", cv::Point(x0, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 215, 255), 1);

    const std::vector<std::string> binds = {
        "ENTER / J : Drop Anchor (TP) & Next Frame",
        "X / K     : Reject Window (FP) & Next Frame",
        "TAB / Q   : Jump Next Candidate",
        "W         : Jump Prev Candidate",
        "A / D     : Step -1 / +1 Frame",
        "SPACE     : Play / Pause",
        "9 / 0     : Cycle Target Finger",
        "C         : Override Key ID",
        "N         : Add Manual Candidate (FN)",
        "P         : Full Video Pre-Scan",
        "M         : Toggle FIFO Finger / Time",
        "G         : Mark File Completed",
        "E         : Batch Export to .bin/.h5"
    };
    y += 22;
    for (const auto& b : binds) {
        cv::putText(canvas, b, cv::Point(x0, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(200, 200, 210), 1);
        y += 20;
    }
}

void AnnotationSuite::drawBottomTimeline(cv::Mat& canvas, const cv::Rect& bounds) {
    cv::rectangle(canvas, bounds, cv::Scalar(20, 20, 24), -1);
    cv::line(canvas, cv::Point(0, bounds.y),
             cv::Point(bounds.width, bounds.y), cv::Scalar(70, 70, 85), 1);

    int pad_x = 25;
    int bar_w = bounds.width - 2 * pad_x;
    int total_f = std::max(1, sessions_[current_session_idx_].total_frames);

    auto frameToX = [&](int f) {
        return pad_x + static_cast<int>((static_cast<double>(std::clamp(f, 0, total_f)) / total_f) * bar_w);
    };

    int track1_y = bounds.y + 35;
    cv::rectangle(canvas, cv::Rect(pad_x, track1_y, bar_w, 24), cv::Scalar(42, 44, 52), -1);

    if (highest_frame_processed_ > 0) {
        int proc_x = frameToX(highest_frame_processed_);
        cv::rectangle(canvas, cv::Rect(pad_x, track1_y + 20, proc_x - pad_x, 4),
                      cv::Scalar(90, 130, 60), -1);
    }

    for (size_t i = 0; i < candidates_.size(); ++i) {
        const auto& cw = candidates_[i];
        int x1 = frameToX(cw.start_frame);
        int x2 = std::max(x1 + 3, frameToX(cw.end_frame));

        cv::Scalar col(0, 200, 255);
        if (cw.state == WindowState::ACCEPTED) col = cv::Scalar(60, 220, 60);
        else if (cw.state == WindowState::REJECTED) col = cv::Scalar(60, 60, 220);

        int top_offset = (cw.finger_id == active_finger_id_) ? 0 : 8;
        cv::rectangle(canvas, cv::Rect(x1, track1_y + top_offset, x2 - x1, 20 - top_offset), col, -1);

        if (static_cast<int>(i) == active_candidate_idx_) {
            cv::rectangle(canvas, cv::Rect(x1 - 1, track1_y - 2, (x2 - x1) + 2, 28),
                          cv::Scalar(255, 255, 255), 2);
        }
    }

    int ph_x = frameToX(current_frame_idx_);
    cv::line(canvas, cv::Point(ph_x, track1_y - 6),
             cv::Point(ph_x, track1_y + 30), cv::Scalar(0, 80, 255), 2);

    int track2_y = bounds.y + 88;
    int zoom_radius = 20;
    int cell_w = bar_w / (2 * zoom_radius + 1);

    std::map<int, int> local_labels;
    for (const auto& cw : candidates_) {
        if (cw.finger_id != active_finger_id_ || cw.state != WindowState::ACCEPTED) continue;
        
        // Phase 1: Procedural Expansion (Only triggers from TPs)
        for (const auto& a : cw.anchors) {
            if (a.label != 1) continue; 
            int t0 = a.impact_frame;
            for (int d = config_.neg_approach_start; d <= config_.neg_approach_end; ++d) {
                if (!local_labels.count(t0 + d)) local_labels[t0 + d] = 0;
            }
            for (int d = config_.neg_release_start; d <= config_.neg_release_end; ++d) {
                if (!local_labels.count(t0 + d)) local_labels[t0 + d] = 0;
            }
            for (int d = -config_.pos_frame_before; d <= config_.pos_frame_after; ++d) {
                local_labels[t0 + d] = 1;
            }
        }
        
        // Phase 2: Explicit Labels Guarantee UI Override
        for (const auto& a : cw.anchors) {
            local_labels[a.impact_frame] = a.label;
        }
    }

    cv::putText(canvas,
                "PROCEDURAL EXPANSION PREVIEW (Green = Touch [1], Red = Hard Negative [0]):",
                cv::Point(pad_x, track2_y - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 195), 1);

    for (int d = -zoom_radius; d <= zoom_radius; ++d) {
        int f = current_frame_idx_ + d;
        int cx = pad_x + (d + zoom_radius) * cell_w;
        cv::Rect cell(cx, track2_y, cell_w - 2, 26);

        if (f < 0 || f >= total_f) {
            cv::rectangle(canvas, cell, cv::Scalar(15, 15, 18), -1);
            continue;
        }

        cv::Scalar cell_col(48, 50, 58);
        if (local_labels.count(f)) {
            cell_col = (local_labels[f] == 1) ? cv::Scalar(40, 190, 40)
                                              : cv::Scalar(45, 45, 180);
        }
        cv::rectangle(canvas, cell, cell_col, -1);

        if (d == 0) {
            cv::rectangle(canvas, cell, cv::Scalar(255, 255, 255), 2);
        }
    }

    cv::putText(canvas, status_banner_, cv::Point(pad_x, bounds.y + bounds.height - 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 230, 255), 1);
}

void AnnotationSuite::printControlsHelp() const {
    std::cout << "\n======================================================\n"
              << "   CV KEYBOARD ACTIVE-LEARNING ANNOTATION SUITE\n"
              << "======================================================\n"
              << "  ENTER / J : Drop Anchor (True Positive) & Next Frame\n"
              << "  X / K     : Reject Window (False Positive) & Next Frame\n"
              << "  TAB / Q   : Jump to Next Candidate Window\n"
              << "  W         : Jump to Previous Candidate Window\n"
              << "  A / D     : Step -1 / +1 Frame\n"
              << "  SPACE     : Play / Pause\n"
              << "  9 / 0     : Cycle Active Finger (H1 Thumb..H2 Pinky)\n"
              << "  C         : Override Key ID for Active Window\n"
              << "  N         : Create Manual Candidate Window (FN Fix)\n"
              << "  P         : Full Video Dual-Gate Pre-Scan\n"
              << "  M         : Toggle Queue Order (FIFO Finger / Time)\n"
              << "  G         : Toggle Session Status (WIP / COMPLETED)\n"
              << "  [ / ]     : Save & Load Prev / Next Video File\n"
              << "  E         : Batch Export Dataset (.bin + .h5)\n"
              << "  ESC       : Save & Quit\n"
              << "======================================================\n";
}

} // namespace cv_keyboard

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./annotation_suite <path_to_videos_dir_or_video.mp4> [--tracker <rtmpose|mediapipe>]\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string tracker = "rtmpose";
    std::string processor = "shadow";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tracker" && i + 1 < argc) {
            tracker = argv[++i];
        } else if (arg == "--processor" && i + 1 < argc) {
            processor = argv[++i];
        }
    }

    cv_keyboard::AnnotationSuite suite(input_path, tracker, processor);
    suite.run();
    return 0;
}