#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief 摄像头分辨率描述
 */
struct CameraResolution {
    int width;
    int height;
    double fps;
};

/**
 * @brief USB / Video0 摄像头捕获器 (基于 OpenCV VideoCapture)
 *
 * 支持任意 V4L2 USB 摄像头, 可配置分辨率、帧率、设备号、曝光等参数。
 * 与 RealSenseCapture 保持一致的接口风格，方便在 main.cpp 中切换。
 */
class USBCapture {
public:
    /**
     * @brief 相机控制参数
     * 值 < 0 表示保持自动/默认, >= 0 则手动设置
     */
    struct CameraControls {
        int autoExposure = -1;   // -1=自动, 1=手动
        int exposure     = 120;  // 手动曝光值 (0~10000), 推荐 50~500
        int gain         = 20;   // 增益 (0~100)
        int brightness   = 128;  // 亮度 (0~255)
        int contrast     = 128;  // 对比度 (0~255)
        int saturation   = 128;  // 饱和度 (0~255)
        int whiteBalance = -1;   // -1=自动, 2000~6500
        int sharpness    = 128;  // 锐度 (0~255)
    };

    /**
     * @param deviceId  /dev/videoN 编号 (默认 0)
     * @param width     期望分辨率宽 (默认 1920)
     * @param height    期望分辨率高 (默认 1080)
     * @param fps       期望帧率 (默认 30)
     * @param fourcc    编码格式, "MJPG" 或 "" (自动), 推荐 MJPG 获得高帧率
     */
    USBCapture(int deviceId = 0,
               int width    = 1920,
               int height   = 1080,
               int fps      = 30,
               const std::string& fourcc = "");

    ~USBCapture();

    // 不可拷贝
    USBCapture(const USBCapture&) = delete;
    USBCapture& operator=(const USBCapture&) = delete;

    /// 启动摄像头
    bool start();

    /// 停止摄像头
    void stop();

    /// 是否正在运行
    bool isRunning() const;

    /**
     * @brief 获取最新一帧
     * @param frame  输出 BGR 图像
     * @return       成功获取返回 true
     */
    bool getFrame(cv::Mat& frame);

    /// 获取实际分辨率 (可能与请求值不同)
    int getWidth()  const;
    int getHeight() const;
    int getFPS()    const;
    int getDeviceId() const;

    /// 设置相机控制参数 (曝光/增益/亮度等), 在 start() 后调用
    void applyControls(const CameraControls& ctrl);

    /// 获取相机内参 (USB 相机无标定, 返回默认值)
    struct Intrinsics {
        float fx, fy;    // 焦距 (默认用 width/2 近似)
        float cx, cy;    // 主点 (默认用 width/2, height/2)
        int   width, height;
    };
    Intrinsics getIntrinsics() const;

    // ----------------------------------------------------------
    // 静态工具方法
    // ----------------------------------------------------------

    /**
     * @brief 列出指定摄像头支持的分辨率
     * @param deviceId 摄像头设备号
     * @return 支持的分辨率列表
     */
    static std::vector<CameraResolution> listResolutions(int deviceId = 0);

    /**
     * @brief 列出系统所有可用的摄像头
     * @return 可用的 /dev/videoN 编号列表
     */
    static std::vector<int> listDevices();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
