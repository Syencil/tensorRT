// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2021/1/26

#ifndef TENSORRT_YOLO_DARKNET_H
#define TENSORRT_YOLO_DARKNET_H

#include "tensorrt.h"

class Darknet : public DetectionTRT{
private:

    void postProcessParall(unsigned long start, unsigned long length, float postThres, const float *bbox_ptr, const float *conf_ptr, std::vector<common::Bbox> *bboxes);

    void safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox);

public:

    //! Initializing
    Darknet(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams);

    //! Read images into buffer
    std::vector<float> preProcess(const std::vector<cv::Mat> &images) override;

    //! Infer
    float infer(const std::vector<std::vector<float>>&InputDatas, common::BufferManager &bufferManager, cudaStream_t stream=nullptr) const override;

    //! Post Process
    std::vector<common::Bbox> postProcess(common::BufferManager &bufferManager, float postThres=-1, float nmsThres=-1) override;

    //! Transform
    void transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes, bool is_padding) override ;

    //! Init Inference Session
    bool initSession(int initOrder) override;

    //! Pred One Image
    std::vector<common::Bbox> predOneImage(const cv::Mat &image, float postThres=-1, float nmsThres=-1) override ;

};

#endif //TENSORRT_YOLO_DARKNET_H
