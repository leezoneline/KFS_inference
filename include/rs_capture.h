#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>

/**
 * @brief RealSense D415 彩色流捕获器
 *
 * 规格:
 *   - 分辨率: 1920×1080
 *   - 帧率:   30 fps
 *   - 传感器: Rolling Shutter, 2MP
 *   - FOV:    69°×42°
 */
class RealSenseCapture {
public:
    /**
     * @param width   彩色流宽 (默认 1920)
     * @param height  彩色流高 (默认 1080)
     * @param fps     帧率 (默认 30)
     */
    RealSenseCapture(int width  = 1920,
                     int height = 1080,
                     int fps    = 30);

    ~RealSenseCapture();

    // 不可拷贝
    RealSenseCapture(const RealSenseCapture&) = delete;
    RealSenseCapture& operator=(const RealSenseCapture&) = delete;

    /// 启动相机流
    bool start();

    /// 停止相机流
    void stop();

    /// 是否正在运行
    bool isRunning() const;

    /**
     * @brief 获取最新一帧 (阻塞)
     * @param frame  输出 BGR 图像
     * @return       成功获取返回 true
     */
    bool getFrame(cv::Mat& frame);

    /// 获取相机内参 (用于坐标系转换)
    struct Intrinsics {
        float fx, fy;    // 焦距
        float cx, cy;    // 主点
        int   width, height;
    };
    Intrinsics getIntrinsics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
