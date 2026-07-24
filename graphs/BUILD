# Custom graph definitions for CV-Keyboard.

load(
    "@mediapipe//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# Expose the graph pbtxt for bundling as data.
exports_files(glob(["*.pbtxt"]))

# Custom hand landmark tracking graph (no rendering overhead).
mediapipe_simple_subgraph(
    name = "hand_landmark_tracker",
    graph = "hand_landmark_tracker.pbtxt",
    register_as = "HandLandmarkTracker",
    deps = [
        "@mediapipe//mediapipe/calculators/core:constant_side_packet_calculator",
        "@mediapipe//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
    ],
)