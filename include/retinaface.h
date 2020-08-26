// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/20

#ifndef TENSORRT_RETINAFACE_H
#define TENSORRT_RETINAFACE_H

#include <thread>
#include <mutex>

#import "tensorrt.h"

class Retinaface : private DetectionTRT{
private:

    //! private function for postProcess
    void postProcessParall(unsigned long start_h, unsigned long length, unsigned long width,
                           unsigned long s, unsigned long pos, const float *loc, const float *conf, const float *land, float postThres,
                           std::vector<common::Bbox> *bboxes);

    //! Ensure thread safety
    //! \param bboxes
    //! \param bbox
    void safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox);

//    //! If the image is padded, bboxes need to be restored.
//    //! \param ih Input image height
//    //! \param iw Input image width
//    //! \param oh Output image height
//    //! \param ow Output image width
//    //! \param retinafaceParam
//    void transformBbx(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes, bool is_padding=true);

public:

    //! Initializing
    Retinaface(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams);

    //! Read images into buffer
    std::vector<float> preProcess(const std::vector<cv::Mat> &images) const override;

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

#endif //TENSORRT_RETINAFACE_H
