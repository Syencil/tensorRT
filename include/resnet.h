// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/6

#ifndef TENSORRT_RESNET_H
#define TENSORRT_RESNET_H

#include "tensorrt.h"
#include "utils.h"

class Resnet : private TensorRT{
private:
    common::ClassificationParams mClassifactionParams;
public:
    //! Initializing
    //! \param inputParams
    //! \param trtParams
    //! \param ImageClassificationParams
    Resnet(common::InputParams inputParams, common::TrtParams trtParams, common::ClassificationParams classifactionParams);

    //! Read images into buffer
    //! \param images
    //! \return Float32 file data
    std::vector<float> preProcess(const std::vector<cv::Mat> &images) const;

    //! Post Process for Yolov3
    //! \param bufferManager It contains inference result
    //! \param postThres
    //! \param nms
    //! \return [xmin, ymin, xmax, ymax]
    std::vector<float> postProcess(common::BufferManager &bufferManager) const;

    //! Init Inference Session
    //! \param initOrder 0 (default) ===> init from SerializedPath. If failed, init from onnxPath.
    //!                             1 ==========> init from onnxPath and save the session into SerializedPath if it doesnt exist.
    //!                             2 ==========> init from onnxPath and force to save the session into SerializedPath.
    //! \return true if no errors happened.
    bool initSession(int initOrder=0);

    //!
    //! \param image
    //! \param posThres Post process threshold.
    //! \param nmsThres NMS Threshold
    //! \return [xmin, ymin, xmax, ymax]
    std::vector<float> predOneImage(const cv::Mat &image);
};

#endif //TENSORRT_RESNET_H
