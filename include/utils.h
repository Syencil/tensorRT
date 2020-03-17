// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/10

#ifndef TENSORRT_7_UTILS_H
#define TENSORRT_7_UTILS_H

#include <NvInfer.h>
#include <dirent.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>

std::vector<float> imagePreprocess(const std::vector<cv::Mat> &images, const int &image_h, const int &image_w, bool is_padding=true, float(*pFun)(unsigned char)=nullptr, bool HWC=true);

std::vector<std::pair<std::string, std::string>> searchDirectory(const std::vector<std::string> &directory, const std::vector<std::string> &suffix=std::vector<std::string>{".jpg", ".png"});

cv::Mat renderBoundingBox(cv::Mat image, const std::vector<std::vector<float>> &bboxes);

cv::Mat renderKeypoint(cv::Mat image, const std::vector<std::vector<float>> &keypoints);

template <typename T>
T clip(const T &n, const T &lower, const T &upper){
    return std::max(lower, std::min(n, upper));
}

#endif //TENSORRT_7_UTILS_H
