# KFS 实时目标检测 (C++)

基于 **Intel RealSense D415** 或 **普通 USB 摄像头** 的实时 YOLO11 三分类目标检测推理模块。

## 检测类别 (3 类)

| Class ID | 名称 | 含义 |
|----------|------|------|
| 0        | R1   | (R_R1 + B_R1) |
| 1        | T    | (T_03 ~ T_17) |
| 2        | F    | (F_18 ~ F_32) |

---

## 依赖

### 系统依赖 (C++ 编译)

| 依赖 | 版本要求 | 必需 | 安装命令 |
|------|----------|------|----------|
| CMake | ≥ 3.16 | ✅ | `sudo apt install cmake` |
| GCC / Clang (C++17) | ≥ 8 / ≥ 7 | ✅ | `sudo apt install build-essential` |
| OpenCV | ≥ 4.x | ✅ | `sudo apt install libopencv-dev` |
| ONNX Runtime | ≥ 1.16 | ✅ | 见下方安装说明 |
| yaml-cpp | 任意 | ✅ | `sudo apt install libyaml-cpp-dev` |
| librealsense2 | ≥ 2.50 | ❌ | `sudo apt install librealsense2-dev` |

> ❌ = 可选, 仅 RealSense D415 需要。没有 D415 可用 USB 摄像头正常编译运行。

### Python 依赖 (仅导出 ONNX 时需要)

```bash
pip install ultralytics onnx onnx-simplifier
```

### 一键安装

```bash
# 必需项
sudo apt install -y cmake build-essential libopencv-dev libyaml-cpp-dev

# ONNX Runtime (推荐手动安装以控制版本)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar xzf onnxruntime-linux-x64-1.19.2.tgz
echo 'export ONNXRUNTIME_DIR='"$(pwd)"'/onnxruntime-linux-x64-1.19.2' >> ~/.bashrc

# 可选 — RealSense D415
sudo apt install -y librealsense2-dev
```

---

## 部署

### 1. 导出 ONNX 模型

```bash
cd realsense_inference
python scripts/export_onnx.py                    # 自动查找 best.pt 导出
python scripts/export_onnx.py --pt /path/to/best.pt   # 指定 .pt 路径
```

### 2. 编译

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

如果 ONNX Runtime 是手动安装的：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.19.2
```

### 3. 配置

编辑 `config/kfs_config.yaml`：

```yaml
camera:
  type: usb             # realsense 或 usb
  usb:
    device: 0
    width: 640
    height: 480
    fps: 60
    fourcc: "MJPG"
model:
  path: models/kfs_yolo11_3class.onnx
  input_size: 640
  conf_threshold: 0.25
  iou_threshold: 0.30
display:
  debug: true           # false = 仅终端输出, 无 GUI
```

### 4. 运行

```bash
./build/kfs_detect                          # 使用默认配置
./build/kfs_detect --config my_config.yaml  # 指定配置
./build/kfs_detect --list-cameras           # 列出 USB 摄像头
```

---

## Debug 模式按键

| 按键 | 功能 |
|------|------|
| `q` / `ESC` | 退出 |
| `s` | 截图保存为 PNG |
| `SPACE` | 触发单次推理 / 切换连续模式 |
| `m` | 切换 IDLE / CONTINUOUS 模式 |

---

## 输出格式

```cpp
struct Detection {
    int class_id;           // 0=R1, 1=T, 2=F
    string class_name;      // "R1", "T", "F"
    float confidence;       // [0, 1]
    Point2f corner_tl;      // 左上角 (像素坐标)
    Point2f corner_tr;      // 右上角
    Point2f corner_br;      // 右下角
    Point2f corner_bl;      // 左下角
};
```

终端输出示例：

```
═══════════════════════════════════════════
  KFS 实时检测结果
  帧尺寸: 640×480 | 推理: 8.2 ms
───────────────────────────────────────────
  # | 类别 | 置信度 | 左上角点 (x,y) | 右下角点 (x,y)
───────────────────────────────────────────
  0 |   R1 |  0.92 | ( 320, 240) | ( 580, 460)
  1 |    T |  0.87 | ( 640, 300) | ( 920, 520)
  2 |    F |  0.76 | (1000, 400) | (1280, 620)
═══════════════════════════════════════════
```

---

## 目录结构

```
realsense_inference/
├── CMakeLists.txt
├── config/
│   └── kfs_config.yaml         # 全局 YAML 配置
├── models/
│   └── kfs_yolo11_3class.onnx  # ONNX 模型
├── include/
│   ├── yolo_detector.h         # 推理核心
│   ├── rs_capture.h            # RealSense D415 采集
│   └── usb_capture.h           # USB 摄像头采集
├── src/
│   ├── main.cpp                # 主程序
│   ├── yolo_detector.cpp       # ONNX 推理实现
│   ├── rs_capture.cpp          # D415 实现
│   └── usb_capture.cpp         # USB 实现
├── scripts/
│   └── export_onnx.py          # .pt → .onnx 导出
└── README.md
```
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

## ❔ ONNX 导出脚本详细帮助

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
