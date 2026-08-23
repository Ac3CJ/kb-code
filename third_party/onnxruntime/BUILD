load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "onnxruntime_so",
    shared_library = "lib/libonnxruntime.so",
)

cc_library(
    name = "onnxruntime",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    deps = [":onnxruntime_so"],
    data = glob(["lib/*.so*"]), 
)
