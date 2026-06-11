cmake_minimum_required(VERSION 3.16)
project(kfs_realsense_detect LANGUAGES CXX CUDA)

# ============================================================
# C++17 标准
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# 编译选项: DEBUG / RELEASE
# ============================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Debug 宏传给代码
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG_MODE)
endif()

# ============================================================
# 依赖查找
# ============================================================

# --- yaml-cpp (配置文件解析) ---
find_package(yaml-cpp REQUIRED)
message(STATUS "yaml-cpp found")

# --- OpenCV (显示 + 预处理) ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# --- ONNX Runtime ---
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_DIR}/include
          /usr/local/include/onnxruntime
          /usr/include/onnxruntime)

find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS ${ONNXRUNTIME_DIR}/lib
          /usr/local/lib
          /usr/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime 未找到！\n"
        "请设置 ONNXRUNTIME_DIR 或安装:\n"
        "  sudo apt install libonnxruntime-dev\n"
        "或从 https://github.com/microsoft/onnxruntime/releases 下载")
endif()
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib:     ${ONNXRUNTIME_LIB}")

# --- librealsense2 ---
find_package(realsense2 QUIET)
if(NOT realsense2_FOUND)
    find_path(REALSENSE_INCLUDE_DIR librealsense2/rs.hpp
        HINTS /usr/local/include
              /usr/include)

    find_library(REALSENSE_LIB realsense2
        HINTS /usr/local/lib
              /usr/lib)

    if(REALSENSE_INCLUDE_DIR AND REALSENSE_LIB)
        set(realsense2_FOUND TRUE)
        set(realsense2_INCLUDE_DIR ${REALSENSE_INCLUDE_DIR})
        set(realsense2_LIBRARY   ${REALSENSE_LIB})
    endif()
endif()

if(realsense2_FOUND)
    message(STATUS "RealSense2 include: ${realsense2_INCLUDE_DIR}")
    message(STATUS "RealSense2 lib:     ${realsense2_LIBRARY}")
    add_definitions(-DHAS_REALSENSE)
else()
    message(WARNING
        "librealsense2 未找到！RealSense D415 功能不可用。\n"
        "USB 摄像头 (--camera usb) 仍可正常使用。\n"
        "如需 RealSense 支持: sudo apt install librealsense2-dev")
endif()

# ============================================================
# 可执行目标
# ============================================================
set(KFS_SOURCES
    src/main.cpp
    src/yolo_detector.cpp
    src/usb_capture.cpp
)

if(realsense2_FOUND)
    list(APPEND KFS_SOURCES src/rs_capture.cpp)
endif()

add_executable(kfs_detect ${KFS_SOURCES})

target_include_directories(kfs_detect PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

if(realsense2_FOUND)
    target_include_directories(kfs_detect PRIVATE ${realsense2_INCLUDE_DIR})
endif()

target_link_libraries(kfs_detect PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    yaml-cpp
    pthread
)

if(realsense2_FOUND)
    target_link_libraries(kfs_detect PRIVATE ${realsense2_LIBRARY})
endif()

# ============================================================
# 安装规则
# ============================================================
install(TARGETS kfs_detect RUNTIME DESTINATION bin)

# ============================================================
# 帮助信息
# ============================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  KFS RealSense D415 推理模块")
message(STATUS "  3 类检测: R1 / T / F")
message(STATUS "  配置: config/kfs_config.yaml (YAML)")
message(STATUS "========================================")
message(STATUS "  编译: mkdir build && cd build")
message(STATUS "         cmake .. -DCMAKE_BUILD_TYPE=Release")
message(STATUS "         make -j$(nproc)")
message(STATUS "  运行: ./kfs_detect [--debug] [--model path]")
message(STATUS "========================================")
