// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/20

#ifndef TENSORRT_RETINAFACE_H
#define TENSORRT_RETINAFACE_H

#include <thread>
#include <mutex>

#import "tensorrt.h"

class Retinaface : private TensorRT{
private:
    common::DetectParams mDetectParams;
    std::mutex mMutex;

private:

    //! private function for postProcess
    //! \param start_h
    //! \param length
    //! \param width
    //! \param s
    //! \param pos
    //! \param loc
    //! \param conf
    //! \param land
    //! \param postThres
    //! \param bboxes
    void postProcessParall(unsigned long start_h, unsigned long length, unsigned long width,
                           unsigned long s, unsigned long pos, const float *loc, const float *conf, const float *land, float postThres,
                           std::vector<common::Bbox> *bboxes);

    //! Ensure thread safety
    //! \param bboxes
    //! \param bbox
    void safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox);

    //! If the image is padded, bboxes need to be restored.
    //! \param ih Input image height
    //! \param iw Input image width
    //! \param oh Output image height
    //! \param ow Output image width
    //! \param retinafaceParam
    void transformBbx(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes, bool is_padding=true);

public:

    //! Initializing
    //! \param inputParams
    //! \param trtParams
    //! \param detectParams
    Retinaface(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams detectParams);

    //! Read images into buffer
    //! \param images
    //! \return Float32 file data
    std::vector<float> preProcess(const std::vector<cv::Mat> &images) const;

    //! Post Process for retinaface
    //! \param bufferManager It contains inference result
    //! \param postThres
    //! \param nms
    //! \return
    std::vector<common::Bbox> postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres);

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
    //! \return
    std::vector<common::Bbox> predOneImage(const cv::Mat &image, float posThres=-1, float nmsThres=-1);


};

#endif //TENSORRT_RETINAFACE_H
