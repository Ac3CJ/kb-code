#ifndef CV_KEYBOARD_ANNOTATION_SUITE_H
#define CV_KEYBOARD_ANNOTATION_SUITE_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "HandData.h"
#include "KeyboardMap.h"
#include "Mediator.h"
#include "Settings.h"

namespace cv_keyboard {

// ============================================================================
// 1. CONFIGURATION & DATA STRUCTURES
// ============================================================================

struct DatasetConfig {
    int roi_width = 150;
    int roi_height = 75;
    int roi_offset_x = 0;
    int roi_offset_y = 20;

    // Procedural Label Expansion Bounds (Relative to Anchor Frame t0)
    int pos_frame_before = 1;      // [t0 - 1, t0 + 1] -> Label 1 (Touch)
    int pos_frame_after = 1;
    int neg_approach_start = -6;   // [t0 - 6, t0 - 2] -> Label 0 (Hard Negative)
    int neg_approach_end = -2;
    int neg_release_start = 3;     // [t0 + 3, t0 + 7] -> Label 0 (Release Negative)
    int neg_release_end = 7;

    // Dual-Gate Cascade Classifier Hyper-Sensitive Thresholds
    int blur_size = 31;
    int shadow_l_thresh = 115;     // Slightly higher L threshold for higher recall
    int motion_thresh = 12;        // Lower motion threshold for higher sensitivity
    float shadow_dot_thresh = -0.25f;
    float kinematic_vy_thresh = 0.008f; // Normalized downward velocity gate
    int merge_gap_frames = 10;     // Merge same-finger windows if <= 10 frames apart

    void save(const std::filesystem::path& path) const;
    bool load(const std::filesystem::path& path);
};

enum class FileStatus {
    UNANNOTATED, // Red: No JSON exists
    IN_PROGRESS, // Yellow: JSON exists, status == "in_progress", hash matches
    COMPLETED,   // Green: JSON exists, status == "completed", hash matches
    DESYNCED     // Orange: JSON exists, but video hash changed
};

enum class WindowState {
    PENDING,
    ACCEPTED,
    REJECTED
};

struct AnchorEvent {
    int impact_frame = 0;
    int label = 1;
    int finger_id = 0;        // 0..4 = Hand 1 (Thumb..Pinky), 5..9 = Hand 2 (Thumb..Pinky)
    int px = 0;               // Raw fingertip X in full video pixels
    int py = 0;               // Raw fingertip Y in full video pixels
    cv::Rect raw_roi;         // Exact [X, Y, W, H] at time of annotation
    std::string key_id = "";  // Semantic key (e.g., "E", "SPACE")
    bool heuristic_triggered = true;
};

struct CandidateWindow {
    int id = 0;
    int finger_id = 0;        // 0..9
    int start_frame = 0;
    int end_frame = 0;
    int peak_frame = 0;       // Frame where heuristic first triggered
    bool trig_shadow = false;
    bool trig_kinematic = false;
    std::string suggested_key = "";
    WindowState state = WindowState::PENDING;
    std::vector<AnchorEvent> anchors; // Supports multiple anchors if windows merged
};

struct VideoSessionMeta {
    std::filesystem::path video_path;
    std::filesystem::path json_path;
    std::string video_hash;
    FileStatus status = FileStatus::UNANNOTATED;

    std::string camera_type = "Phone";
    double fps = 30.0;
    int frame_width = 1920;
    int frame_height = 1080;
    int total_frames = 0;
    std::string lighting = "overhead";
    std::string surface = "matte_light";
    std::string expected_transcript = "";
};

// Binary Header for Fast C++ -> Python HDF5 Export
#pragma pack(push, 1)
struct BinaryCropHeader {
    char magic[4] = {'C', 'V', 'K', 'B'};
    uint32_t version = 1;
    uint32_t num_samples = 0;
    uint32_t roi_w = 150;
    uint32_t roi_h = 75;
    uint32_t channels = 3;
};

struct BinarySampleMeta {
    int32_t frame_index;
    int32_t label;            // 1 = Touch, 0 = Hard Negative
    int32_t finger_id;        // 0..9
    int32_t roi_x;
    int32_t roi_y;
    int32_t roi_w;
    int32_t roi_h;
    char key_id[16];          // Null-terminated key string
};
#pragma pack(pop)

// ============================================================================
// 2. MAIN ANNOTATION SUITE CLASS
// ============================================================================

class AnnotationSuite {
public:
    AnnotationSuite(const std::string& input_path,
                    const std::string& tracker_name,
                    const std::string& processor_name);
    ~AnnotationSuite();

    void run();

private:
    // Workspace & JSON Persistence
    void scanWorkspace();
    void loadSession(int idx);
    void loadSessionJSON(VideoSessionMeta& sess);
    void saveCurrentSessionJSON();

    // Dual-Gate Cascade Classifier (Shadow Convergence OR Kinematic Zero-Crossing)
    static std::vector<HandData> getSpatiallySortedHands(
        std::shared_ptr<const std::vector<HandData>> hands_ptr);
    void processOrFetchFrame(int frame_idx, const cv::Mat& frame);
    void runDualGateCascadeStep(int frame_idx,
                                const cv::Mat& bgr_frame,
                                std::shared_ptr<const std::vector<HandData>> hands_ptr);
    std::string guessKeyAtPixel(int px, int py) const;
    void registerCascadeTrigger(int frame_idx,
                                int finger_id,
                                bool trig_shadow,
                                bool trig_kinematic,
                                const std::string& suggested_key);
    void sortCandidateQueue();
    void runFullVideoCascadePreScan();

    // Navigation & Anchor Labeling Actions
    void seekToFrame(int target_frame);
    void jumpToCandidate(int direction);
    bool getFingerPixelCoords(int frame_idx, int finger_id, int& out_px, int& out_py) const;
    void labelCurrentFrame(int label_val);
    void clearCurrentWindowAnchors();
    void createManualCandidateAtCurrentFrame();
    void handleKeyReclassificationInput(int key);
    void cycleLightingTag();
    void cycleSurfaceTag();

    // Batch Dataset Exporter
    void runBatchExporter();

    // UI Rendering
    void renderWorkspaceUI(bool is_paused);
    void drawSidebar(cv::Mat& canvas, const cv::Rect& bounds);
    void drawCenterViewport(cv::Mat& canvas, const cv::Rect& bounds, bool is_paused);
    void drawRightInspector(cv::Mat& canvas, const cv::Rect& bounds);
    void drawBottomTimeline(cv::Mat& canvas, const cv::Rect& bounds);
    void printControlsHelp() const;

    // Member Variables
    std::string tracker_name_;
    std::string processor_name_;
    Mediator mediator_;
    KeyboardMap keyboard_map_;
    DatasetConfig config_;

    std::filesystem::path videos_dir_;
    std::filesystem::path annotations_dir_;
    std::filesystem::path exports_dir_;
    std::filesystem::path config_path_;
    std::string initial_video_file_;

    std::vector<VideoSessionMeta> sessions_;
    int current_session_idx_ = 0;

    cv::VideoCapture cap_;
    cv::Mat raw_frame_;
    int current_frame_idx_ = 0;
    int highest_frame_processed_ = -1;

    std::map<int, std::shared_ptr<const std::vector<HandData>>> hand_cache_;
    std::vector<CandidateWindow> candidates_;
    int active_candidate_idx_ = -1;
    int active_finger_id_ = 1; // Default to Hand 1 Index Finger
    bool fifo_finger_mode_ = true;
    bool awaiting_key_reclassify_ = false;

    // Dual-Gate Cascade Internal State
    cv::Mat prev_cascade_gray_;
    std::map<int, float> prev_finger_y_;
    std::map<int, float> prev_finger_vy_;

    std::string window_name_ = "CV Keyboard - Active Learning Annotation Suite";
    std::string status_banner_ = "Ready.";
};

} // namespace cv_keyboard

#endif // CV_KEYBOARD_ANNOTATION_SUITE_H