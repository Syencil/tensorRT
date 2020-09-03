// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/14

#ifndef TENSORRT_YOLOV3_H
#define TENSORRT_YOLOV3_H

#include <thread>
#include <mutex>

#include "tensorrt.h"
#include "utils.h"

class Yolov3 : private DetectionTRT{
private:

    void postProcessParall(unsigned long start, unsigned long length, float postThres, const float *origin_output, std::vector<common::Bbox> *bboxes);

    inline void safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox);

public:

    //! Initializing
    Yolov3(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams);

    //! Read images into buffer
    std::vector<float> preProcess(const std::vector<cv::Mat> &images) override;

    //! Infer
    float infer(const std::vector<std::vector<float>>&InputDatas, common::BufferManager &bufferManager, cudaStream_t stream=nullptr) const override;

    //! Post Process for Yolov3
    std::vector<common::Bbox> postProcess(common::BufferManager &bufferManager, float postThres=-1, float nmsThres=-1) override;

    //! Transform
    void transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes, bool is_padding) override ;

    //! Init Inference Session
    bool initSession(int initOrder) override;

    //! Pred One Image
    std::vector<common::Bbox> predOneImage(const cv::Mat &image, float postThres=-1, float nmsThres=-1) override ;

};

#endif //TENSORRT_YOLOV3_H
