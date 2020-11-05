// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/7/1

#ifndef TENSORRT_PSENET_H
#define TENSORRT_PSENET_H

#include <thread>
#include <mutex>
#include <map>

#include "tensorrt.h"
#include "utils.h"

class Psenet : private SegmentationTRT{
public:

    //! Initializing
    Psenet(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams);

    //! Read images into buffer
    std::vector<float> preProcess(const std::vector<cv::Mat> &images) override;

    //! Infer
    float infer(const std::vector<std::vector<float>>&InputDatas, common::BufferManager &bufferManager, cudaStream_t stream) const override ;

    //! Post Process for Psenetv2
    //! \return cv::Mat CV8U    point_value = 0, 1, 2... cid (0 for background)
    cv::Mat postProcess(common::BufferManager &bufferManager, float postThres=-1) override;

    //! transform
    void transform(const int &ih, const int &iw, const int &oh, const int &ow, cv::Mat &mask, bool is_padding) override ;

    //! Init Inference Session
    //! \param initOrder 0==========> init from SerializedPath. If failed, init from onnxPath.
    //!                             1 ==========> init from onnxPath and save the session into SerializedPath if it doesnt exist.
    //!                             2 ==========> init from onnxPath and force to save the session into SerializedPath.
    //! \return true if no errors happened.
    bool initSession(int initOrder) override;

    //! Pred One Image
    //! \param image
    //! \param posThres Post process threshold.
    //! \return Segmentation Mask CV8U
    cv::Mat predOneImage(const cv::Mat &image, float postThres=-1) override;

};


#endif //TENSORRT_PSENET_H
