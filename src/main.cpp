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

// ============================================================
// 全局标志
// ============================================================
static std::atomic<bool> g_running{true};

static void signalHandler(int) {
    g_running = false;
}

// ============================================================
// 命令行参数解析
// ============================================================
struct Args {
    std::string cameraType = "realsense";  // "realsense" 或 "usb"
    std::string modelPath  = "models/kfs_yolo11_3class.onnx";
    std::string configPath;                // 配置文件路径
    bool        debug      = true;
    float       confThresh = 0.25f;
    float       iouThresh  = 0.30f;  // 更激进的 NMS, 减少重复框
    int         inputSize  = 640;

    // USB 相机参数
    int         usbDevice  = 0;
    int         usbWidth   = 1920;
    int         usbHeight  = 1080;
    int         usbFPS     = 30;
    std::string usbFourcc  = "MJPG";

    // 相机控制 (V4L2)
    int  autoExposure  = -1;    // -1=自动, 1=手动
    int  exposure      = -1;    // 手动曝光值
    int  gain          = -1;
    int  brightness    = -1;
    int  contrast      = -1;
    int  saturation    = -1;
    int  whiteBalance  = -1;    // -1=自动
    int  sharpness     = -1;
};

// 简单配置文件解析 (key = value 格式)
static void loadConfig(const std::string& path, Args& args) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[WARN] 无法打开配置文件: " << path << std::endl;
        return;
    }
    std::string line;
    while (std::getline(f, line)) {
        // 去注释和空白
        auto pos = line.find('#');
        if (pos != std::string::npos) line = line.substr(0, pos);
        // 去首尾空白
        while (!line.empty() && std::isspace(line.front())) line.erase(0, 1);
        while (!line.empty() && std::isspace(line.back()))  line.pop_back();
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        // 去空白
        while (!key.empty() && std::isspace(key.back())) key.pop_back();
        while (!val.empty() && std::isspace(val.front())) val.erase(0, 1);

        if      (key == "device")        args.usbDevice     = std::stoi(val);
        else if (key == "width")         args.usbWidth      = std::stoi(val);
        else if (key == "height")        args.usbHeight     = std::stoi(val);
        else if (key == "fps")           args.usbFPS        = std::stoi(val);
        else if (key == "fourcc")        args.usbFourcc     = val;
        else if (key == "auto_exposure") args.autoExposure  = std::stoi(val);
        else if (key == "exposure")      args.exposure      = std::stoi(val);
        else if (key == "gain")          args.gain          = std::stoi(val);
        else if (key == "brightness")    args.brightness    = std::stoi(val);
        else if (key == "contrast")      args.contrast      = std::stoi(val);
        else if (key == "saturation")    args.saturation    = std::stoi(val);
        else if (key == "white_balance") args.whiteBalance  = std::stoi(val);
        else if (key == "sharpness")     args.sharpness     = std::stoi(val);
    }
    std::cout << "[CONFIG] 已加载: " << path << std::endl;
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--camera" && i + 1 < argc) {
            args.cameraType = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            args.configPath = argv[++i];
            loadConfig(args.configPath, args);
        } else if (arg == "--debug") {
            args.debug = true;
        } else if (arg == "--no-debug") {
            args.debug = false;
        } else if (arg == "--model" && i + 1 < argc) {
            args.modelPath = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            args.confThresh = std::stof(argv[++i]);
        } else if (arg == "--iou" && i + 1 < argc) {
            args.iouThresh = std::stof(argv[++i]);
        } else if (arg == "--size" && i + 1 < argc) {
            args.inputSize = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            args.usbDevice = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            args.usbWidth = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            args.usbHeight = std::stoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            args.usbFPS = std::stoi(argv[++i]);
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

            // 同时提示 RealSense 检测
            std::cout << "\n  提示: RealSense D415 请用 --camera realsense\n";
            std::cout << "  使用 --camera usb --device N 指定摄像头\n\n";
            exit(0);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: ./kfs_detect [选项]\n\n"
                      << "== 相机选择 ==\n"
                      << "  --camera TYPE       相机类型: realsense (默认) 或 usb\n"
                      << "  --list-cameras      列出所有 USB 摄像头及支持的分辨率\n\n"
                      << "== USB 相机参数 ==\n"
                      << "  --device N          设备号 /dev/videoN (默认: 0)\n"
                      << "  --width W           分辨率宽 (默认: 1920)\n"
                      << "  --height H          分辨率高 (默认: 1080)\n"
                      << "  --fps N             帧率 (默认: 30)\n"
                      << "  --config PATH       相机配置文件 (覆盖命令行参数)\n\n"
                      << "== 推理参数 ==\n"
                      << "  --model PATH        ONNX 模型路径\n"
                      << "  --conf VALUE        置信度阈值 (默认: 0.25)\n"
                      << "  --iou VALUE         NMS IOU 阈值 (默认: 0.45)\n"
                      << "  --size VALUE        模型输入尺寸 (默认: 640)\n\n"
                      << "== 显示 ==\n"
                      << "  --debug             开启 debug 可视化 (默认)\n"
                      << "  --no-debug          仅终端输出\n\n"
                      << "== 示例 ==\n"
                      << "  ./kfs_detect --camera usb --config config/usb_camera.conf --debug\n"
                      << "  ./kfs_detect --camera usb --device 0 --width 1280 --height 720 --debug\n"
                      << "  ./kfs_detect --camera realsense --debug --conf 0.5\n\n";
            exit(0);
        }
    }
    return args;
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

static void drawDebug(cv::Mat& frame, const FrameResult& result, double fps) {
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

    cv::putText(frame, "DEBUG MODE | Press Q/ESC to quit | S to screenshot",
                cv::Point(10, frame.rows - 12), fontFace, 0.5,
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
    // 信号处理
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    // 解析命令行
    Args args = parseArgs(argc, argv);

    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   KFS 目标检测 — 实时推理               ║\n";
    std::cout << "║   类别: R1 / T / F                      ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    // 1) 加载 YOLO 检测器
    YoloDetector detector(args.modelPath, args.inputSize,
                          args.confThresh, args.iouThresh);

    // 2) 启动相机 (RealSense 或 USB)
    CameraHandle camera;
    std::string  windowName;

    if (args.cameraType == "usb") {
        camera.type = CameraType::USB;
        camera.usb  = new USBCapture(args.usbDevice,
                                     args.usbWidth, args.usbHeight, args.usbFPS,
                                     args.usbFourcc);

        // 先设置 V4L2 控制 (在 start() 内部通过 v4l2-ctl 在 OpenCV 打开前应用)
        USBCapture::CameraControls ctrl;
        ctrl.autoExposure = args.autoExposure;
        ctrl.exposure     = args.exposure;
        ctrl.gain         = args.gain;
        ctrl.brightness   = args.brightness;
        ctrl.contrast     = args.contrast;
        ctrl.saturation   = args.saturation;
        ctrl.whiteBalance = args.whiteBalance;
        ctrl.sharpness    = args.sharpness;
        camera.usb->applyControls(ctrl);

        if (!camera.usb->start()) {
            std::cerr << "[ERROR] 无法启动 USB 摄像头 /dev/video" << args.usbDevice << "\n"
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
            std::cerr << "[ERROR] 无法启动 RealSense D415 相机！\n"
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
        std::cerr << "[ERROR] 此编译版本不支持 RealSense (未链接 librealsense2)\n"
                  << "  请使用 --camera usb 或重新编译带 RealSense 支持的版本\n";
        return 1;
    }
#endif

    // 3) 主循环
    cv::Mat frame;
    int    frameCount = 0;
    auto   tStart     = std::chrono::high_resolution_clock::now();
    double fps        = 0.0;

    std::cout << "\n[INFO] 开始推理... "
              << (args.debug ? "(DEBUG 可视化)" : "(控制台输出)")
              << "\n  按 Ctrl+C 退出\n\n";

    while (g_running) {
        // 取帧
        if (!camera.getFrame(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 推理
        auto result = detector.detect(frame);

        // FPS 计算
        frameCount++;
        auto tNow = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(tNow - tStart).count();
        if (elapsed >= 1.0) {
            fps = frameCount / elapsed;
            frameCount = 0;
            tStart = tNow;
        }

        if (args.debug) {
            // === Debug 模式: 显示画面 ===
            drawDebug(frame, result, fps);
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
            cv::imshow(windowName, frame);

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {  // q 或 ESC
                g_running = false;
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
            // === 非 Debug: 仅终端输出 ===
            printResults(result);
        }
    }

    // 4) 清理
    camera.stop();
    if (args.debug) {
        cv::destroyAllWindows();
    }

    std::cout << "\n[INFO] 程序正常退出\n";
    return 0;
}
