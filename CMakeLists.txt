cmake_minimum_required(VERSION 3.16)
project(cv-keyboard VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ---------------------------------------------------------------------------
# MediaPipe — built externally with Bazel
# Set MediaPipe_DIR to the directory containing MediaPipeConfig.cmake,
# or point MEDIAPIPE_ROOT to the built MediaPipe installation prefix.
# ---------------------------------------------------------------------------
if(NOT DEFINED MEDIAPIPE_ROOT)
    set(MEDIAPIPE_ROOT "/usr/local" CACHE PATH "MediaPipe installation prefix")
endif()

# MediaPipe library paths
set(MEDIAPIPE_LIBS
    ${MEDIAPIPE_ROOT}/lib/libmediapipe_framework.a
    ${MEDIAPIPE_ROOT}/lib/libmediapipe_hand_tracking.a
)

# MediaPipe include paths
set(MEDIAPIPE_INCLUDE_DIRS
    ${MEDIAPIPE_ROOT}/include
    ${MEDIAPIPE_ROOT}/include/mediapipe
)

# Proto dependency (MediaPipe uses protobuf)
find_package(Protobuf REQUIRED)

# Abseil (MediaPipe depends on Abseil)
find_package(absl REQUIRED)

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(SOURCES
    src/Application.cpp
    src/HandTracker.cpp
    src/Mediator.cpp
    src/main.cpp
)

add_executable(cv-keyboard ${SOURCES})

target_include_directories(cv-keyboard PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${MEDIAPIPE_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(cv-keyboard PRIVATE
    ${OpenCV_LIBS}
    ${MEDIAPIPE_LIBS}
    ${Protobuf_LIBRARIES}
    absl::base
    absl::strings
    absl::synchronization
    absl::flat_hash_map
)

# ---------------------------------------------------------------------------
# Copy MediaPipe graph config next to the binary for runtime discovery
# ---------------------------------------------------------------------------
add_custom_command(TARGET cv-keyboard POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${MEDIAPIPE_ROOT}/graphs/hand_tracking"
        "$<TARGET_FILE_DIR:cv-keyboard>/mediapipe/graphs/hand_tracking"
    COMMENT "Copying MediaPipe hand tracking graph configs"
)


