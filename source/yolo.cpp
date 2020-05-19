// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/14

#include "../include/yolo.h"

Yolo::Yolo(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams) :
        TensorRT(std::move(inputParams), std::move(trtParams)), mYoloParams(std::move(yoloParams)) {

}

std::vector<float> Yolo::preProcess(const std::vector<cv::Mat> &images) const {
    std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, mInputParams.pFunction, true);
    return fileData;
}

std::vector<common::Bbox> Yolo::postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) const {
    if(postThres<0){
        postThres = mYoloParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mYoloParams.NMSThreshold;
    }
    std::vector<common::Bbox> bboxes;
    common::Bbox bbox;
    float conf, score, prob;
    int cid;
    for (int scale_idx=0; scale_idx<3; ++scale_idx){
        int length = mInputParams.ImgH / mYoloParams.Strides[scale_idx] * mInputParams.ImgW / mYoloParams.Strides[scale_idx] * mYoloParams.AnchorPerScale;
        auto *origin_output = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[scale_idx]));
        for(int i=0; i<length*(5+mYoloParams.NumClass); i+=(5+mYoloParams.NumClass)){
            prob=-1;
            cid=-1;
            bbox.xmin = clip<float>(origin_output[i] - origin_output[i+2] * 0.5, 0, static_cast<float>(mInputParams.ImgW-1));
            bbox.ymin = clip<float>(origin_output[i+1] - origin_output[i+3] * 0.5, 0, static_cast<float>(mInputParams.ImgH-1));
            bbox.xmax = clip<float>(origin_output[i] + origin_output[i+2] * 0.5, 0, static_cast<float>(mInputParams.ImgW-1));
            bbox.ymax = clip<float>(origin_output[i+1] + origin_output[i+3] * 0.5, 0, static_cast<float>(mInputParams.ImgH-1));
            conf = origin_output[i+4];
            for (int c=i+5; c<i+5+mYoloParams.NumClass; ++c){
                if (origin_output[c] > prob){
                    prob = origin_output[c];
                    cid = c - 5 - i;
                }
            }
            score = conf * prob;
            if (score>=postThres){
                bbox.score = score;
                bbox.cid = cid;
                bboxes.emplace_back(bbox);
            }
        }
    }
    return nms(bboxes, nmsThres);
}

bool Yolo::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}


std::vector<common::Bbox> Yolo::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
    float elapsedTime = infer(std::vector<std::vector<float>>{preProcess(std::vector<cv::Mat>{image})}, bufferManager);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    std::vector<common::Bbox> bboxes = postProcess(bufferManager, postThres, nmsThres);
    if(mInputParams.IsPadding){
        this->transformBbx(image.rows, image.cols, mInputParams.ImgH, mInputParams.ImgW, bboxes);
    }
    return bboxes;
}

void Yolo::transformBbx(const int &ih, const int &iw, const int &oh, const int &ow,
                        std::vector<common::Bbox> &bboxes) {
    float scale = std::min(static_cast<float>(ow) / static_cast<float>(iw), static_cast<float>(oh) / static_cast<float>(ih));
    int nh = static_cast<int>(scale * static_cast<float>(ih));
    int nw = static_cast<int>(scale * static_cast<float>(iw));
    int dh = (oh - nh) / 2;
    int dw = (ow - nw) / 2;
    for (auto &bbox : bboxes){
        bbox.xmin = (bbox.xmin - dw) / scale;
        bbox.ymin = (bbox.ymin - dh) / scale;
        bbox.xmax = (bbox.xmax - dw) / scale;
        bbox.ymax = (bbox.ymax - dh) / scale;
    }
}






