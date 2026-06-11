#include "usb_capture.h"

#include <opencv2/videoio.hpp>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <dirent.h>

// ============================================================
// V4L2 控制必须在 OpenCV 打开设备之前用 v4l2-ctl 设置
// OpenCV 打开后设备流锁定, set() 可能不生效
// ============================================================
static void runV4L2Ctl(int deviceId, const std::string& args) {
    std::ostringstream cmd;
    cmd << "v4l2-ctl -d /dev/video" << deviceId << " " << args << " 2>/dev/null";
    (void)system(cmd.str().c_str());
}

struct USBCapture::Impl {
    cv::VideoCapture cap;
    int  deviceId;
    int  width, height, fps;
    int  realWidth, realHeight, realFPS;
    int  fourccCode;
    bool running = false;
    CameraControls pendingCtrls;

    Impl(int devId, int w, int h, int f, const std::string& fourcc)
        : deviceId(devId), width(w), height(h), fps(f)
    {
        if (fourcc.empty() || fourcc == "MJPG" || fourcc == "mjpg") {
            fourccCode = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        } else if (fourcc == "YUYV" || fourcc == "yuyv") {
            fourccCode = cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V');
        } else {
            fourccCode = -1;
        }
    }

    // 在 OpenCV 打开设备之前通过 v4l2-ctl 设置相机控制
    void applyV4L2BeforeOpen() {
        // 曝光: 强制手动模式, 避免默认的 Aperture Priority(3) 导致过曝
        if (pendingCtrls.autoExposure >= 0) {
            if (pendingCtrls.autoExposure == 1) {
                runV4L2Ctl(deviceId, "-c auto_exposure=1");
                if (pendingCtrls.exposure >= 0)
                    runV4L2Ctl(deviceId, "-c exposure_time_absolute=" + std::to_string(pendingCtrls.exposure));
                std::cout << "[USB] 曝光: 手动, time=" << pendingCtrls.exposure << std::endl;
            } else {
                runV4L2Ctl(deviceId, "-c auto_exposure=0");
                std::cout << "[USB] 曝光: 自动" << std::endl;
            }
        }
        if (pendingCtrls.brightness > -64)
            runV4L2Ctl(deviceId, "-c brightness=" + std::to_string(pendingCtrls.brightness));
        if (pendingCtrls.contrast >= 0)
            runV4L2Ctl(deviceId, "-c contrast=" + std::to_string(pendingCtrls.contrast));
        if (pendingCtrls.saturation >= 0)
            runV4L2Ctl(deviceId, "-c saturation=" + std::to_string(pendingCtrls.saturation));
        if (pendingCtrls.sharpness >= 0)
            runV4L2Ctl(deviceId, "-c sharpness=" + std::to_string(pendingCtrls.sharpness));
        if (pendingCtrls.gain >= 0)
            runV4L2Ctl(deviceId, "-c gamma=" + std::to_string(pendingCtrls.gain + 100));
        if (pendingCtrls.whiteBalance >= 0) {
            runV4L2Ctl(deviceId, "-c white_balance_automatic=0");
            runV4L2Ctl(deviceId, "-c white_balance_temperature=" + std::to_string(pendingCtrls.whiteBalance));
        }
    }

    bool start() {
        if (running) return true;

        // 1) 先设 V4L2 控制
        applyV4L2BeforeOpen();

        // 2) 再打开摄像头
        bool ok = cap.open(deviceId, cv::CAP_V4L2);
        if (!ok) ok = cap.open(deviceId);
        if (!ok) {
            std::cerr << "[USB] 无法打开 /dev/video" << deviceId << std::endl;
            return false;
        }

        // 3) 设置 FOURCC (必须在分辨率之前, V4L2 要求)
        if (fourccCode != -1) {
            cap.set(cv::CAP_PROP_FOURCC, static_cast<double>(fourccCode));
            int actualFourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
            std::cout << "[USB] 编码: " << (char)(actualFourcc & 0xFF)
                      << (char)((actualFourcc >> 8) & 0xFF)
                      << (char)((actualFourcc >> 16) & 0xFF)
                      << (char)((actualFourcc >> 24) & 0xFF) << std::endl;
        }

        // 4) 设置分辨率
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  static_cast<double>(width));
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(height));

        // 5) 设置帧率 (FOURCC 和分辨率之后)
        cap.set(cv::CAP_PROP_FPS, static_cast<double>(fps));

        // 读取实际值
        realWidth  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        realHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        realFPS    = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

        if (realWidth <= 0 || realHeight <= 0) {
            realWidth  = 640;
            realHeight = 480;
        }
        if (realFPS <= 0) realFPS = fps;

        running = true;
        std::cout << "[USB] /dev/video" << deviceId
                  << " 已启动: " << realWidth << "×" << realHeight
                  << " @ " << realFPS << " fps" << std::endl;
        return true;
    }

    void stop() {
        if (!running) return;
        cap.release();
        running = false;
        std::cout << "[USB] 摄像头已释放" << std::endl;
    }
};

// ============================================================
// USBCapture 公共接口
// ============================================================

USBCapture::USBCapture(int deviceId, int width, int height, int fps,
                       const std::string& fourcc)
    : pImpl(std::make_unique<Impl>(deviceId, width, height, fps, fourcc))
{}

USBCapture::~USBCapture() { pImpl->stop(); }

bool USBCapture::start()          { return pImpl->start(); }
void USBCapture::stop()           { pImpl->stop(); }
bool USBCapture::isRunning() const { return pImpl->running; }

bool USBCapture::getFrame(cv::Mat& frame) {
    if (!pImpl->running) return false;
    return pImpl->cap.read(frame);
}

int USBCapture::getWidth()  const { return pImpl->realWidth; }
int USBCapture::getHeight() const { return pImpl->realHeight; }
int USBCapture::getFPS()    const { return pImpl->realFPS; }
int USBCapture::getDeviceId() const { return pImpl->deviceId; }

// applyControls: 保存参数, 在 start() 中通过 v4l2-ctl 应用
void USBCapture::applyControls(const CameraControls& ctrl) {
    pImpl->pendingCtrls = ctrl;
}

USBCapture::Intrinsics USBCapture::getIntrinsics() const {
    Intrinsics intr{};
    intr.width  = pImpl->realWidth;
    intr.height = pImpl->realHeight;
    // 粗略近似 (无标定)
    intr.fx = static_cast<float>(pImpl->realWidth)  * 0.5f;
    intr.fy = static_cast<float>(pImpl->realHeight) * 0.5f;
    intr.cx = static_cast<float>(pImpl->realWidth)  * 0.5f;
    intr.cy = static_cast<float>(pImpl->realHeight) * 0.5f;
    return intr;
}

// ============================================================
// 静态工具方法
// ============================================================

static bool isVideoDevice(const std::string& name) {
    if (name.find("video") == std::string::npos) return false;
    for (char c : name) {
        if (c >= '0' && c <= '9') return true;
    }
    return false;
}

std::vector<int> USBCapture::listDevices() {
    std::vector<int> devices;

    DIR* dir = opendir("/dev");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name = entry->d_name;
            if (!isVideoDevice(name)) continue;

            std::string numStr;
            for (char c : name) {
                if (c >= '0' && c <= '9') numStr += c;
            }
            if (!numStr.empty()) {
                int n = std::stoi(numStr);
                cv::VideoCapture test(n, cv::CAP_V4L2);
                if (test.isOpened()) {
                    devices.push_back(n);
                    test.release();
                }
            }
        }
        closedir(dir);
    }

    return devices;
}

std::vector<CameraResolution> USBCapture::listResolutions(int deviceId) {
    std::vector<CameraResolution> resList;

    cv::VideoCapture cap(deviceId, cv::CAP_V4L2);
    if (!cap.isOpened()) return resList;

    static const int commonRes[][2] = {
        {1920, 1080}, {1280, 720}, {960, 540},
        {800, 600},   {640, 480},  {480, 360}, {320, 240},
        {3840, 2160}, {2560, 1440}, {2592, 1944},
        {2048, 1536}, {1600, 1200},
    };

    double origW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double origH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    for (const auto& r : commonRes) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  static_cast<double>(r[0]));
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(r[1]));

        int gotW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int gotH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        if (gotW == r[0] && gotH == r[1]) {
            double gotFPS = cap.get(cv::CAP_PROP_FPS);
            if (gotFPS <= 0) gotFPS = 30.0;

            bool dup = false;
            for (auto& x : resList) {
                if (x.width == gotW && x.height == gotH) { dup = true; break; }
            }
            if (!dup) {
                resList.push_back({gotW, gotH, gotFPS});
            }
        }
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH,  origW);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, origH);
    cap.release();

    return resList;
}
