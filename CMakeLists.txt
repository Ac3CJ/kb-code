cmake_minimum_required(VERSION 3.16)
project(cv-keyboard VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Source files
set(SOURCES
    src/Application.cpp
    src/main.cpp
)

add_executable(cv-keyboard ${SOURCES})

target_include_directories(cv-keyboard PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cv-keyboard PRIVATE ${OpenCV_LIBS})