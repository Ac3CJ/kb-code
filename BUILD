load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

# ==============================================================================
# THE CORE ENGINE (Reusable Library)
# ==============================================================================
cc_library(
    name = "cv_keyboard_core",
    srcs = [
        "src/MediaPipeTracker.cpp",
        "src/Mediator.cpp",
        "src/KeyboardMap.cpp",
        "src/ZeroCrossingProcessor.cpp",
        "src/InterpolationProcessor.cpp",
        "src/BaseClickProcessor.cpp",
        "src/RTMPoseTracker.cpp",
        "src/HandSmoother.cpp",
        "src/TypingEngine.cpp",

        "include/MediaPipeTracker.h",
        "include/IHandTracker.h",
        "include/Mediator.h",
        "include/Settings.h",
        "include/KeyboardMap.h",
        "include/IClickProcessor.h",
        "include/ZeroCrossingProcessor.h",
        "include/InterpolationProcessor.h",
        "include/BaseClickProcessor.h",
        "include/RTMPoseTracker.h",
        "include/HandSmoother.h",
        "include/TypingEngine.h",
    ],
    includes = ["include"],
    linkopts = [
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_video",
        "-lopencv_aruco",
        "-lopencv_calib3d",
        "-lopencv_dnn",
    ],
    data = [
        "//graphs:hand_landmark_tracker.pbtxt",
        "@mediapipe//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "@mediapipe//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
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
        "//graphs:hand_landmark_tracker",
        "@mediapipe//third_party:opencv",
        "//third_party/onnxruntime:onnxruntime",
    ],
)

# ==============================================================================
# THE LIVE APPLICATION (Webcam / scrcpy)
# ==============================================================================
cc_binary(
    name = "cv-keyboard",
    srcs = [
        "src/main.cpp",
        "src/Application.cpp",
        "include/Application.h",
    ],
    includes = ["include"],
    deps = [
        ":cv_keyboard_core",
    ],
    linkopts = [
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_video",
        "-lopencv_aruco",
        "-lopencv_calib3d",
        "-lopencv_dnn",
    ],
)

# ==============================================================================
# THE OFFLINE TESTER (Pre-recorded Video processing)
# ==============================================================================
cc_binary(
    name = "offline_tester",
    srcs = [
        "src/offline_tester_main.cpp",
    ],
    includes = ["include"],
    deps = [
        ":cv_keyboard_core", 
    ],
    linkopts = [
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_video",
        "-lopencv_aruco",
        "-lopencv_calib3d",
        "-lopencv_dnn",
    ],
)