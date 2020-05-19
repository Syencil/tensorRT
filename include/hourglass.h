// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/16

#ifndef TENSORRT_HOURGLASS_H
#define TENSORRT_HOURGLASS_H

#include "tensorrt.h"

class Hourglass : private TensorRT{
private:
    common::KeypointParams keypointParams;
private:

    //! If the image is padded, bboxes need to be restored.
    //! \param ih Input image height
    //! \param iw Input image width
    //! \param oh Output image height
    //! \param ow Output image width
    //! \param [x, y, cid, prob]
    void transformPoint(const int &h, const int &w, std::vector<common::Keypoint> &keypoints);

public:

    //! Initializing
    //! \param inputParams
    //! \param trtParams
    //! \param hourglassParams
    Hourglass(common::InputParams inputParams, common::TrtParams trtParams, common::KeypointParams hourglassParams);

    //! Read images into buffer
    //! \param images
    //! \return Float32 file data
    std::vector<float> preProcess(const std::vector<cv::Mat> &images) const;

    //! Post Process for Hourglass
    //! \param bufferManager It contains inference result
    //! \param postThres
    //! \return [x, y, cid, prob]
    std::vector<common::Keypoint> postProcess(common::BufferManager &bufferManager, float postThres=-1) const;

    //! Init Inference Session
    //! \param initOrder 0==========> init from SerializedPath. If failed, init from onnxPath.
    //!                             1 ==========> init from onnxPath and save the session into SerializedPath if it doesnt exist.
    //!                             2 ==========> init from onnxPath and force to save the session into SerializedPath.
    //! \return true if no errors happened.
    bool initSession(int initOrder) override ;

    //!
    //! \param image
    //! \param posThres Post process threshold.
    //! \return [x, y, cid, prob]
    std::vector<common::Keypoint> predOneImage(const cv::Mat &image, float posThres=-1);
};
#endif //TENSORRT_HOURGLASS_H
