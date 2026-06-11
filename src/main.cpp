/**
 * @file main.cpp
 * @brief KFS 目标检测 — 实时推理
 *
 * 规格:
 *   - 相机: Intel RealSense D415  或  USB /dev/video* 摄像头
 *   - 模型: YOLO11 3 类 (R1 / T / F)
 *   - 输出: 类别 + 4 角点坐标
 *   - Debug: 显示推理画面 (OpenCV imshow)
 *
 * 用法:
 *   # RealSense D415
 *   ./kfs_detect --camera realsense --debug
 *
 *   # USB 摄像头 (默认 /dev/video0, 自动检测分辨率)
 *   ./kfs_detect --camera usb --debug
 *
 *   # USB 摄像头 - 指定分辨率和设备号
 *   ./kfs_detect --camera usb --device 2 --width 1280 --height 720
 *
 *   # 列出系统所有 USB 摄像头及支持的分辨率
 *   ./kfs_detect --list-cameras
 *
 * 按键 (debug 模式):
 *   q / ESC — 退出
 *   s      — 截图保存
 */

#include "yolo_detector.h"
#ifdef HAS_REALSENSE
#include "rs_capture.h"
#endif
#include "usb_capture.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <csignal>
#include <atomic>
#include <thread>
#include <cstdio>

#include <yaml-cpp/yaml.h>

// ============================================================
// 全局标志
// ============================================================
static std::atomic<bool> g_running{true};

static void signalHandler(int) {
    g_running = false;
}

// ============================================================
// YAML 配置结构
// ============================================================
struct Config {
    std::string cameraType = "usb";
    std::string modelPath  = "models/kfs_yolo11_3class.onnx";
    bool        debug      = true;
    float       confThresh = 0.25f;
    float       iouThresh  = 0.30f;
    int         inputSize  = 640;

    // USB 相机参数
    int         usbDevice  = 0;
    int         usbWidth   = 1280;
    int         usbHeight  = 1024;
    int         usbFPS     = 30;
    std::string usbFourcc  = "";

    // 相机控制 (V4L2)
    int  autoExposure  = -1;
    int  exposure      = -1;
    int  gain          = -1;
    int  brightness    = -1;
    int  contrast      = -1;
    int  saturation    = -1;
    int  whiteBalance  = -1;
    int  sharpness     = -1;
};

// YAML 辅助: 获取节点值 (缺失时返回默认值)
static int yamlInt(const YAML::Node& node, const std::string& key, int defval) {
    if (node && node[key]) return node[key].as<int>();
    return defval;
}
static float yamlFloat(const YAML::Node& node, const std::string& key, float defval) {
    if (node && node[key]) return node[key].as<float>();
    return defval;
}
static std::string yamlStr(const YAML::Node& node, const std::string& key, const std::string& defval) {
    if (node && node[key]) return node[key].as<std::string>();
    return defval;
}
static bool yamlBool(const YAML::Node& node, const std::string& key, bool defval) {
    if (node && node[key]) return node[key].as<bool>();
    return defval;
}

static Config loadYamlConfig(const std::string& path) {
    Config cfg;
    try {
        YAML::Node root = YAML::LoadFile(path);

        // camera
        if (root["camera"]) {
            auto cam = root["camera"];
            cfg.cameraType = yamlStr(cam, "type", cfg.cameraType);

            if (cam["usb"]) {
                auto usb = cam["usb"];
                cfg.usbDevice = yamlInt(usb, "device", cfg.usbDevice);
                cfg.usbWidth  = yamlInt(usb, "width",  cfg.usbWidth);
                cfg.usbHeight = yamlInt(usb, "height", cfg.usbHeight);
                cfg.usbFPS    = yamlInt(usb, "fps",    cfg.usbFPS);
                cfg.usbFourcc = yamlStr(usb, "fourcc", cfg.usbFourcc);
            }

            if (cam["controls"]) {
                auto ctrl = cam["controls"];
                cfg.autoExposure = yamlInt(ctrl, "auto_exposure", cfg.autoExposure);
                cfg.exposure     = yamlInt(ctrl, "exposure",      cfg.exposure);
                cfg.gain         = yamlInt(ctrl, "gain",          cfg.gain);
                cfg.brightness   = yamlInt(ctrl, "brightness",    cfg.brightness);
                cfg.contrast     = yamlInt(ctrl, "contrast",      cfg.contrast);
                cfg.saturation   = yamlInt(ctrl, "saturation",    cfg.saturation);
                cfg.whiteBalance = yamlInt(ctrl, "white_balance", cfg.whiteBalance);
                cfg.sharpness    = yamlInt(ctrl, "sharpness",     cfg.sharpness);
            }
        }

        // model
        if (root["model"]) {
            auto mdl = root["model"];
            cfg.modelPath  = yamlStr(mdl, "path",           cfg.modelPath);
            cfg.inputSize  = yamlInt(mdl, "input_size",     cfg.inputSize);
            cfg.confThresh = yamlFloat(mdl, "conf_threshold", cfg.confThresh);
            cfg.iouThresh  = yamlFloat(mdl, "iou_threshold",  cfg.iouThresh);
        }

        // display
        if (root["display"]) {
            auto disp = root["display"];
            cfg.debug = yamlBool(disp, "debug", cfg.debug);
        }

    } catch (const std::exception& e) {
        std::cout << "[ERROR] YAML 解析失败: " << e.what() << "\n";
        exit(1);
    }
    return cfg;
}

static void printUsage() {
    std::cout << "用法: ./kfs_detect [--config PATH] [--list-cameras] [--help]\n\n"
              << "== 选项 ==\n"
              << "  --config PATH       指定配置文件 (默认: config/kfs_config.yaml)\n"
              << "  --list-cameras      列出所有 USB 摄像头及支持的分辨率\n"
              << "  --help / -h         显示帮助\n\n"
              << "== 配置文件格式 ==\n"
              << "  所有参数统一在 YAML 文件中设置: config/kfs_config.yaml\n"
              << "  包含: 相机类型/分辨率/帧率/编码/V4L2控制/模型参数/显示\n\n"
              << "== 示例 ==\n"
              << "  ./build/kfs_detect                                    # 使用默认配置\n"
              << "  ./build/kfs_detect --config my_config.yaml            # 指定配置\n"
              << "  ./build/kfs_detect --list-cameras                    # 查看可用摄像头\n\n";
    exit(0);
}

// ============================================================
// 检测模式
// ============================================================
enum class DetectMode {
    IDLE,          // 仅显示视频流, 不推理
    SINGLE_SHOT,   // 触发一次推理, 完成后回到 IDLE
    CONTINUOUS     // 持续推理每一帧
};

static const char* modeLabel(DetectMode m) {
    switch (m) {
        case DetectMode::IDLE:       return "IDLE (press SPACE to detect)";
        case DetectMode::SINGLE_SHOT:return "SINGLE-SHOT";
        case DetectMode::CONTINUOUS: return "CONTINUOUS";
    }
    return "";
}

// ============================================================
// Debug 可视化
// ============================================================

// 类别对应颜色 (BGR)
static const cv::Scalar CLASS_COLORS[] = {
    cv::Scalar(0, 255, 0),   // R1 — 绿色
    cv::Scalar(255, 0, 0),   // T  — 蓝色
    cv::Scalar(0, 0, 255),   // F  — 红色
};

static void drawDebug(cv::Mat& frame, const FrameResult& result, double fps,
                      DetectMode mode) {
    const int thickness = 2;
    const double fontScale = 0.7;
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;

    for (const auto& det : result.detections) {
        const auto& color = CLASS_COLORS[det.class_id % 3];

        // 绘制 4 条边 (用 4 个角点)
        std::vector<cv::Point> corners = {
            cv::Point(static_cast<int>(det.corner_tl.x), static_cast<int>(det.corner_tl.y)),
            cv::Point(static_cast<int>(det.corner_tr.x), static_cast<int>(det.corner_tr.y)),
            cv::Point(static_cast<int>(det.corner_br.x), static_cast<int>(det.corner_br.y)),
            cv::Point(static_cast<int>(det.corner_bl.x), static_cast<int>(det.corner_bl.y)),
        };
        cv::polylines(frame, corners, true, color, thickness);

        // 绘制 4 个角点小圆
        for (const auto& pt : corners) {
            cv::circle(frame, pt, 4, color, -1);
        }

        // 标签文本
        std::ostringstream label;
        label << det.class_name << " " << std::fixed << std::setprecision(2)
              << det.confidence;

        cv::Size textSz = cv::getTextSize(label.str(), fontFace, fontScale, 1, nullptr);
        int labelY = std::max(corners[0].y - 8, textSz.height + 4);

        cv::rectangle(frame,
                      cv::Point(corners[0].x, labelY - textSz.height - 4),
                      cv::Point(corners[0].x + textSz.width + 4, labelY + 2),
                      color, -1);
        cv::putText(frame, label.str(),
                    cv::Point(corners[0].x + 2, labelY),
                    fontFace, fontScale, cv::Scalar(255, 255, 255), 1);
    }

    // 左上角状态信息
    std::ostringstream ss;
    ss << "FPS: " << std::fixed << std::setprecision(1) << fps
       << " | Inference: " << result.inference_ms << " ms"
       << " | Detections: " << result.detections.size();
    cv::putText(frame, ss.str(),
                cv::Point(10, 30), fontFace, 0.6,
                cv::Scalar(0, 255, 255), 1);

    // 底部: 模式 + 按键提示
    cv::Scalar modeColor = (mode == DetectMode::IDLE) ? cv::Scalar(100, 200, 100)
                           : (mode == DetectMode::CONTINUOUS) ? cv::Scalar(0, 200, 255)
                           : cv::Scalar(0, 255, 255);
    cv::putText(frame, modeLabel(mode),
                cv::Point(10, frame.rows - 36), fontFace, 0.55,
                modeColor, 1);
    cv::putText(frame, "SPACE: detect once | D: toggle continuous | S: screenshot | Q/ESC: quit",
                cv::Point(10, frame.rows - 12), fontFace, 0.45,
                cv::Scalar(200, 200, 200), 1);
}

// ============================================================
// 终端输出
// ============================================================

static void printResults(const FrameResult& result) {
    std::cout << "\033[2J\033[H";  // 清屏
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "  KFS RealSense D415 检测结果\n";
    std::cout << "  帧尺寸: " << result.frame_size.width
              << "×" << result.frame_size.height
              << " | 推理: " << std::fixed << std::setprecision(1)
              << result.inference_ms << " ms\n";
    std::cout << "───────────────────────────────────────────\n";

    if (result.detections.empty()) {
        std::cout << "  (无检测)\n";
    } else {
        std::cout << "  # | 类别 | 置信度 | 左上角点 (x,y) | 右下角点 (x,y)\n";
        std::cout << "───────────────────────────────────────────\n";
        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& d = result.detections[i];
            std::cout << "  " << std::setw(2) << i
                      << " | " << std::setw(4) << d.class_name
                      << " | " << std::fixed << std::setprecision(2)
                      << std::setw(5) << d.confidence
                      << " | (" << std::setw(4) << static_cast<int>(d.corner_tl.x)
                      << ","  << std::setw(4) << static_cast<int>(d.corner_tl.y)
                      << ") | (" << std::setw(4) << static_cast<int>(d.corner_br.x)
                      << ","  << std::setw(4) << static_cast<int>(d.corner_br.y)
                      << ")\n";
        }
    }
    std::cout << "═══════════════════════════════════════════\n";
}

// ============================================================
// 相机抽象: 统一 RealSense 与 USB 相机接口
// ============================================================

enum class CameraType { RealSense, USB };

struct CameraHandle {
    CameraType type = CameraType::USB;
    USBCapture* usb = nullptr;
#ifdef HAS_REALSENSE
    RealSenseCapture* rs = nullptr;
#endif

    ~CameraHandle() { stop(); }

    bool isRunning() const {
#ifdef HAS_REALSENSE
        if (type == CameraType::RealSense && rs) return rs->isRunning();
#endif
        if (type == CameraType::USB      && usb) return usb->isRunning();
        return false;
    }

    bool getFrame(cv::Mat& frame) {
#ifdef HAS_REALSENSE
        if (type == CameraType::RealSense && rs) return rs->getFrame(frame);
#endif
        if (type == CameraType::USB      && usb) return usb->getFrame(frame);
        return false;
    }

    void stop() {
#ifdef HAS_REALSENSE
        if (rs) { rs->stop(); delete rs; rs = nullptr; }
#endif
        if (usb) { usb->stop(); delete usb; usb = nullptr; }
    }
};

// ============================================================
// 主函数
// ============================================================

int main(int argc, char** argv) {
    // 重定向 stderr 到 /dev/null, 抑制 libjpeg "Corrupt JPEG data" 警告
    if (freopen("/dev/null", "w", stderr) == nullptr) { /* 忽略 */ }

    // 信号处理
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    // 解析命令行 → 仅 --config 和 --list-cameras / --help
    std::string configPath = "config/kfs_config.yaml";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--list-cameras") {
            std::cout << "═══════════════════════════════════════════\n";
            std::cout << "  系统 USB 摄像头列表\n";
            std::cout << "═══════════════════════════════════════════\n\n";
            auto devices = USBCapture::listDevices();
            if (devices.empty()) {
                std::cout << "  未检测到 USB 摄像头\n\n";
                std::cout << "  提示:\n";
                std::cout << "    - 检查 USB 连接: ls /dev/video*\n";
                std::cout << "    - 检查权限: sudo usermod -aG video $USER\n";
            } else {
                for (int dev : devices) {
                    std::cout << "  /dev/video" << dev << ":\n";
                    auto resList = USBCapture::listResolutions(dev);
                    if (resList.empty()) {
                        std::cout << "    (无法获取分辨率列表)\n";
                    } else {
                        for (const auto& r : resList) {
                            std::cout << "    - " << r.width << "×" << r.height
                                      << " @ " << static_cast<int>(r.fps) << " fps\n";
                        }
                    }
                }
            }
            std::cout << "\n═══════════════════════════════════════════\n";
            exit(0);
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
        } else {
            std::cout << "[WARN] 未知参数: " << arg << " (用 --help 查看用法)\n";
        }
    }

    // 加载 YAML 配置
    Config cfg = loadYamlConfig(configPath);
    std::cout << "[CONFIG] 已加载: " << configPath << std::endl;

    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   KFS 目标检测 — 实时推理               ║\n";
    std::cout << "║   类别: R1 / T / F                      ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    // 1) 加载 YOLO 检测器
    YoloDetector detector(cfg.modelPath, cfg.inputSize,
                          cfg.confThresh, cfg.iouThresh);

    // 2) 启动相机 (RealSense 或 USB)
    CameraHandle camera;
    std::string  windowName;

    if (cfg.cameraType == "usb") {
        camera.type = CameraType::USB;
        camera.usb  = new USBCapture(cfg.usbDevice,
                                     cfg.usbWidth, cfg.usbHeight, cfg.usbFPS,
                                     cfg.usbFourcc);

        // 先设置 V4L2 控制 (在 start() 内部通过 v4l2-ctl 在 OpenCV 打开前应用)
        USBCapture::CameraControls ctrl;
        ctrl.autoExposure = cfg.autoExposure;
        ctrl.exposure     = cfg.exposure;
        ctrl.gain         = cfg.gain;
        ctrl.brightness   = cfg.brightness;
        ctrl.contrast     = cfg.contrast;
        ctrl.saturation   = cfg.saturation;
        ctrl.whiteBalance = cfg.whiteBalance;
        ctrl.sharpness    = cfg.sharpness;
        camera.usb->applyControls(ctrl);

        if (!camera.usb->start()) {
            std::cout << "[ERROR] 无法启动 USB 摄像头 /dev/video" << cfg.usbDevice << "\n"
                      << "  请检查: 1) USB 连接  2) 权限 (sudo usermod -aG video $USER)\n"
                      << "  提示: 运行 ./kfs_detect --list-cameras 查看可用摄像头\n";
            return 1;
        }

        windowName = "KFS Detection - USB Camera";

        std::cout << "[INFO] 实际分辨率: "
                  << camera.usb->getWidth() << "×"
                  << camera.usb->getHeight() << " @ "
                  << camera.usb->getFPS() << " fps\n";
    }
#ifdef HAS_REALSENSE
    else {
        camera.type = CameraType::RealSense;
        camera.rs   = new RealSenseCapture(1920, 1080, 30);

        if (!camera.rs->start()) {
            std::cout << "[ERROR] 无法启动 RealSense D415 相机！\n"
                      << "  请检查: 1) USB 连接  2) udev 规则  3) 权限\n"
                      << "  提示: 使用 --camera usb 切换到普通 USB 摄像头\n";
            return 1;
        }

        windowName = "KFS Detection - RealSense D415";

        auto intrinsics = camera.rs->getIntrinsics();
        std::cout << "[INFO] 相机内参: fx=" << intrinsics.fx
                  << " fy=" << intrinsics.fy
                  << " cx=" << intrinsics.cx
                  << " cy=" << intrinsics.cy << std::endl;
    }
#else
    else {
        std::cout << "[ERROR] 此编译版本不支持 RealSense (未链接 librealsense2)\n"
                  << "  请使用 --camera usb 或重新编译带 RealSense 支持的版本\n";
        return 1;
    }
#endif

    // 3) 主循环
    cv::Mat frame;
    int    frameCount = 0;
    auto   tStart     = std::chrono::high_resolution_clock::now();
    double fps        = 0.0;

    DetectMode detectMode = DetectMode::IDLE;
    FrameResult lastResult;  // 保存最后一次推理结果用于叠加显示

    // 提前创建窗口 (只一次, 否则每帧重建严重拖慢渲染)
    if (cfg.debug) {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    }

    std::cout << "\n[INFO] YOLO 模型已加载, 等待触发推理...\n"
              << (cfg.debug ? "  按 SPACE 单次推理 | 按 D 持续推理 | Q/ESC 退出\n"
                            : "  按 Ctrl+C 退出\n")
              << "\n";

    while (g_running) {
        // 取帧 (先检查退出标志, 避免阻塞在 read 中无法退出)
        if (!g_running) break;
        if (!camera.getFrame(frame) || frame.empty()) {
            if (!g_running) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 根据模式决定是否推理
        bool shouldDetect = (detectMode == DetectMode::CONTINUOUS)
                         || (detectMode == DetectMode::SINGLE_SHOT);

        if (shouldDetect) {
            lastResult = detector.detect(frame);
            if (detectMode == DetectMode::SINGLE_SHOT) {
                detectMode = DetectMode::IDLE;  // 单次完成, 回到 IDLE
            }
        }

        // FPS 计算
        frameCount++;
        {
            auto tFps = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(tFps - tStart).count();
            if (elapsed >= 1.0) {
                fps = frameCount / elapsed;
                frameCount = 0;
                tStart = tFps;
            }
        }

        if (cfg.debug) {
            // === Debug 模式: 显示画面 ===
            drawDebug(frame, lastResult, fps, detectMode);
            cv::imshow(windowName, frame);

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {  // q 或 ESC
                g_running = false;
            } else if (key == ' ') {  // 空格: 单次推理
                detectMode = DetectMode::SINGLE_SHOT;
                std::cout << "[INFO] 触发单次推理" << std::endl;
            } else if (key == 'd' || key == 'D') {  // D: 切换持续推理
                detectMode = (detectMode == DetectMode::CONTINUOUS)
                           ? DetectMode::IDLE : DetectMode::CONTINUOUS;
                std::cout << "[INFO] 检测模式: " << modeLabel(detectMode) << std::endl;
            } else if (key == 's') {
                // 截图
                auto now = std::chrono::system_clock::now();
                auto ts  = std::chrono::duration_cast<std::chrono::seconds>(
                    now.time_since_epoch()).count();
                std::string fname = "screenshot_" + std::to_string(ts) + ".png";
                cv::imwrite(fname, frame);
                std::cout << "[SAVE] 截图已保存: " << fname << std::endl;
            }
        } else {
            // === 非 Debug: 仅终端输出 (仅推理时打印) ===
            if (shouldDetect) {
                printResults(lastResult);
            }
        }
    }

    // 4) 清理
    camera.stop();
    if (cfg.debug) {
        cv::destroyAllWindows();
    }

    std::cout << "\n[INFO] 程序正常退出\n";
    return 0;
}
