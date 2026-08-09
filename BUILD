load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "cv-keyboard",
    srcs = [
        "src/main.cpp",
        "src/Application.cpp",
        "src/HandTracker.cpp",
        "src/Mediator.cpp",
        "src/KeyboardMap.cpp",
        # Move headers into srcs for cc_binary
        "include/Application.h",
        "include/HandTracker.h",
        "include/Mediator.h",
        "include/Settings.h",
        "include/KeyboardMap.h",
    ],
    # Tells Bazel where to find headers so `#include "HandTracker.h"` works seamlessly
    includes = ["include"],
    data = [
        # Custom graph definition (loaded at runtime via runfiles)
        "//graphs:hand_landmark_tracker.pbtxt",

        # MediaPipe hand landmark model files (bundled via mediapipe_files rule)
        "@mediapipe//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "@mediapipe//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
        # MediaPipe Core & Hand Tracking Graph Execution
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:file_helpers",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/formats:landmark_cc_proto",
        "@mediapipe//mediapipe/framework/formats:classification_cc_proto",

        "@mediapipe//mediapipe/modules/hand_landmark:hand_landmark_tracking_gpu",
        "@mediapipe//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
        "@mediapipe//mediapipe/gpu:gpu_shared_data_internal",
        "@mediapipe//mediapipe/gpu:gpu_buffer",


        # Our custom graph subgraph (links in HandLandmarkTrackingCpu)
        "//graphs:hand_landmark_tracker",

        # OpenCV for video capture and frame processing
        "@mediapipe//third_party:opencv",
    ],
)