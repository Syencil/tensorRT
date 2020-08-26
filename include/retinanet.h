// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/15

#ifndef TENSORRT_RETINANET_H
#define TENSORRT_RETINANET_H

#include "tensorrt.h"
#include "utils.h"

class RetinaNet : private DetectionTRT{
public:

    //! Initializing
    RetinaNet(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams detectParams);

    //! Read images into buffer
    std::vector<float> preProcess(const std::vector<cv::Mat> &images) const override ;

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


#endif //TENSORRT_RETINANET_H
