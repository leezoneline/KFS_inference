# KFS 实时目标检测 (C++)

基于 **Intel RealSense D415** 或 **普通 USB 摄像头** 的实时 YOLO11 三分类目标检测推理模块。

## 🏷️ 检测类别 (3 类)

| Class ID | 名称 | 含义 |
|---|---|---|
| 0 | R1 | (R_R1 + B_R1) |
| 1 | T | (T_03 ~ T_17) |
| 2 | F | (F_18 ~ F_32) |

## 📁 目录结构

```
realsense_inference/
├── CMakeLists.txt              # CMake 构建配置
├── models/                     # ONNX 模型存放
│   └── kfs_yolo11_3class.onnx
├── include/
│   ├── yolo_detector.h         # 检测器头文件 (统一推理)
│   ├── rs_capture.h            # RealSense D415 驱动
│   └── usb_capture.h           # USB /dev/video* 驱动
├── src/
│   ├── main.cpp                # 主程序 (debug 可视化 + 控制台输出)
│   ├── yolo_detector.cpp       # YOLO11 ONNX 推理核心
│   ├── rs_capture.cpp          # RealSense D415 采集
│   └── usb_capture.cpp         # USB 摄像头采集
├── scripts/
│   └── export_onnx.py          # 导出训练模型为 ONNX (详细帮助见下文)
└── README.md
```

## 🔧 依赖安装

```bash
# OpenCV (≥4.x) — 必需
sudo apt install libopencv-dev

# ONNX Runtime — 必需
# 方式 A: apt
sudo apt install libonnxruntime-dev

# 方式 B: 手动 (推荐, 版本可控)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar xzf onnxruntime-linux-x64-1.19.2.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.19.2

# librealsense2 — 可选 (仅 RealSense D415 需要)
sudo apt install librealsense2-dev
```

> **注意**: 没有 RealSense D415 也可以编译运行，只需使用 `--camera usb` 即可驱动普通 USB 摄像头。

## 🚀 快速开始

### 1. 导出 ONNX 模型

```bash
cd ~/KFS_training/realsense_inference

# 自动查找 best.pt 并导出 (推荐)
python scripts/export_onnx.py

# 列出项目下所有可用的 .pt 文件
python scripts/export_onnx.py --list-pt

# 指定 .pt 路径
python scripts/export_onnx.py --pt ~/KFS_training/training_output/runs/KFS_full_3class/weights/best.pt

# 导出并验证
python scripts/export_onnx.py --verify

# 导出 FP16 半精度 (更小更快, 需 GPU 推理)
python scripts/export_onnx.py --half

# 导出其他格式
python scripts/export_onnx.py --format openvino   # Intel OpenVINO
python scripts/export_onnx.py --format engine      # NVIDIA TensorRT

# 查看完整帮助
python scripts/export_onnx.py --help
```

### 2. 编译 C++ 推理模块

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

如果 ONNX Runtime 是手动安装的:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.19.2
```

### 3. 运行

#### RealSense D415 (默认)

```bash
# Debug 模式 (显示推理画面 + 角点)
./kfs_detect --camera realsense --debug

# 无 GUI 模式 (仅终端输出)
./kfs_detect --camera realsense --no-debug
```

#### USB 摄像头 (/dev/videoN)

```bash
# 列出所有可用摄像头及其支持的分辨率
./kfs_detect --list-cameras

# 默认 /dev/video0, 自动检测最佳分辨率
./kfs_detect --camera usb --debug

# 指定设备号和分辨率
./kfs_detect --camera usb --device 2 --width 1280 --height 720 --debug

# 指定帧率
./kfs_detect --camera usb --device 0 --width 1920 --height 1080 --fps 30 --debug

# 无 GUI 模式
./kfs_detect --camera usb --device 0 --no-debug
```

#### 自定义参数

```bash
./kfs_detect --camera usb --debug --conf 0.5 --iou 0.4 --model ../models/custom.onnx
```

## 🎮 Debug 模式操作

| 按键 | 功能 |
|---|---|
| `q` / `ESC` | 退出 |
| `s` | 截图保存为 PNG |

## 📊 输出格式

每个检测框包含:

```cpp
struct Detection {
    int class_id;           // 0=R1, 1=T, 2=F
    string class_name;      // "R1", "T", "F"
    float confidence;       // [0, 1]

    // 4 个角点 (像素坐标)
    Point2f corner_tl;      // 左上
    Point2f corner_tr;      // 右上
    Point2f corner_br;      // 右下
    Point2f corner_bl;      // 左下
};
```

### 终端输出示例

```
═══════════════════════════════════════════
  KFS 实时检测结果
  帧尺寸: 1920×1080 | 推理: 8.2 ms
───────────────────────────────────────────
  # | 类别 | 置信度 | 左上角点 (x,y) | 右下角点 (x,y)
───────────────────────────────────────────
  0 |   R1 |  0.92 | ( 320, 240) | ( 580, 460)
  1 |    T |  0.87 | ( 640, 300) | ( 920, 520)
  2 |    F |  0.76 | (1000, 400) | (1280, 620)
═══════════════════════════════════════════
```

## 📝 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--camera TYPE` | `realsense` | 相机类型: `realsense` 或 `usb` |
| `--list-cameras` | - | 列出所有 USB 摄像头及支持的分辨率 |
| `--device N` | `0` | USB 设备号 /dev/videoN |
| `--width W` | `1920` | USB 相机分辨率宽 |
| `--height H` | `1080` | USB 相机分辨率高 |
| `--fps N` | `30` | USB 相机帧率 |
| `--debug` | 开启 | 显示 OpenCV 可视化窗口 |
| `--no-debug` | - | 仅终端文本输出 |
| `--model PATH` | `models/kfs_yolo11_3class.onnx` | ONNX 模型路径 |
| `--conf VALUE` | `0.25` | 置信度阈值 |
| `--iou VALUE` | `0.45` | NMS IoU 阈值 |
| `--size VALUE` | `640` | 模型输入尺寸 |
| `--help, -h` | - | 显示帮助信息 |

## ⚙️ 相机权限 (Linux)

### USB 摄像头

```bash
# 将当前用户加入 video 组
sudo usermod -aG video $USER
# 重新登录生效
```

### RealSense D415

```bash
# 添加 udev 规则
sudo bash -c 'echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"8086\", MODE=\"0666\"" > /etc/udev/rules.d/99-realsense.rules'
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## 🧪 测试 (无相机时用视频/图片)

如果没有连接摄像头，可以写一个简单的测试程序用图片或视频文件验证 ONNX 推理:

```cpp
// test_standalone.cpp
#include "yolo_detector.h"
#include <opencv2/highgui.hpp>

int main() {
    YoloDetector detector("models/kfs_yolo11_3class.onnx");
    cv::Mat img = cv::imread("test.jpg");
    auto result = detector.detect(img);

    for (auto& d : result.detections) {
        printf("[%s] conf=%.2f  tl=(%.0f,%.0f) br=(%.0f,%.0f)\n",
               d.class_name.c_str(), d.confidence,
               d.corner_tl.x, d.corner_tl.y,
               d.corner_br.x, d.corner_br.y);
    }
    return 0;
}
```

## � ONNX 导出脚本详细帮助

```bash
cd ~/KFS_training/realsense_inference
python scripts/export_onnx.py --help
```

支持的导出格式:
- `onnx` — ONNX (CPU/GPU 通用)
- `engine` — NVIDIA TensorRT (最快, 需 GPU + TensorRT)
- `openvino` — Intel OpenVINO (Intel CPU/GPU 优化)
- `tflite` — TensorFlow Lite (移动端/嵌入式)
- `coreml` — Apple Core ML (macOS/iOS)

### 常见问题

| 问题 | 解决方法 |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| ONNX 简化失败 | 加 `--no-simplify` 或 `pip install onnx-simplifier` |
| opset 版本不兼容 | 尝试 `--opset 11` |
| .pt 文件找不到 | 先运行 `train_yolo11_3class.py` 训练, 或 `--list-pt` 查看
