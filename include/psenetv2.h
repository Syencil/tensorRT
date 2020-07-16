// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/7/13

#ifndef TENSORRT_PANNET_H
#define TENSORRT_PANNET_H

#include <thread>
#include <mutex>

#include "tensorrt.h"
#include "utils.h"

class Psenetv2 : private TensorRT{
private:
    common::DetectParams mDetectParams;
    std::mutex mMutex;
private:

    //! If the image is padded, bboxes need to be restored.
    //! \param ih Input image height
    //! \param iw Input image width
    //! \param oh Output image height
    //! \param ow Output image width
    //! \param mask cv::Mat
    cv::Mat transformBbx(const int &ih, const int &iw, const int &oh, const int &ow, cv::Mat &mask, bool is_padding=true);

public:

    //! Initializing
    //! \param inputParams
    //! \param trtParams
    //! \param yoloParams
    Psenetv2(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams);

    //! Read images into buffer
    //! \param images
    //! \return Float32 file data
    std::vector<float> preProcess(const std::vector<cv::Mat> &images) const;

    //! Post Process for Psenetv2
    //! \param bufferManager It contains inference result
    //! \param postThres
    //! \param nms
    //! \return cv::Mat CV8U    point_value = 0, 1, 2... cid (0 for background)
    cv::Mat postProcess(common::BufferManager &bufferManager, float postThres=-1);

    //! Init Inference Session
    //! \param initOrder 0==========> init from SerializedPath. If failed, init from onnxPath.
    //!                             1 ==========> init from onnxPath and save the session into SerializedPath if it doesnt exist.
    //!                             2 ==========> init from onnxPath and force to save the session into SerializedPath.
    //! \return true if no errors happened.
    bool initSession(int initOrder) override;

    //!
    //! \param image
    //! \param posThres Post process threshold.
    //! \return Segmentation Mask CV8U
    cv::Mat predOneImage(const cv::Mat &image, float postThres=-1);

};

#endif //TENSORRT_PANNET_H
