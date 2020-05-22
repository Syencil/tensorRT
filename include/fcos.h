// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/9

#ifndef TENSORRT_CENTERNET_H
#define TENSORRT_CENTERNET_H

#include "tensorrt.h"
#include "utils.h"
#include <algorithm>


class FCOS : private TensorRT{
private:
    common::DetectParams mDetectParams;
private:

    //! If the image is padded, bboxes need to be restored.
    //! \param ih Input image height
    //! \param iw Input image width
    //! \param oh Output image height
    //! \param ow Output image width
    //! \param bboxes [xmin, ymin, xmax, ymax, cid, prob]
    void transformBbox(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes);

public:

    //! Initializing
    //! \param inputParams
    //! \param trtParams
    //! \param detectParams
    FCOS(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams detectParams);

    //! Read images into buffer
    //! \param images
    //! \return Float32 file data
    std::vector<float> preProcess(const std::vector<cv::Mat> &images) const;

    //! Post Process for Fcos
    //! \param bufferManager It contains inference result
    //! \param postThres
    //! \param nmsThres
    //! \return common::Bbox{xmin, ymin, xmax, ymax, score, cid}
    std::vector<common::Bbox> postProcess(common::BufferManager &bufferManager, float postThres=-1, float nmsThres=-1) const;

    //! Init Inference Session
    //! \param initOrder 0==========> init from SerializedPath. If failed, init from onnxPath.
    //!                             1 ==========> init from onnxPath and save the session into SerializedPath if it doesnt exist.
    //!                             2 ==========> init from onnxPath and force to save the session into SerializedPath.
    //! \return true if no errors happened.
    bool initSession(int initOrder) override;

    //!
    //! \param image
    //! \param posThres Post process threshold.
    //! \param nmsThres NMS Threshold
    //! \return common::Bbox{xmin, ymin, xmax, ymax, score, cid}
    std::vector<common::Bbox> predOneImage(const cv::Mat &image, float postThres=-1, float nmsThres=-1);
};

#endif //TENSORRT_CENTERNET_H
