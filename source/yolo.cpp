// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/14

#include "../include/yolo.h"

Yolo::Yolo(common::InputParams inputParams, common::TrtParams trtParams, common::YoloParams yoloParams) : TensorRT(std::move(inputParams), std::move(trtParams)), mYoloParams(std::move(yoloParams)) {

}

std::vector<float> Yolo::preProcess(const std::vector<cv::Mat> &images) const {
    std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, mInputParams.pFunction, true);
    return fileData;
}

std::vector<std::vector<float>> Yolo::postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) const {
    if(postThres<0){
        postThres = mYoloParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mYoloParams.NMSThreshold;
    }
    bufferManager.copyOutputToHost();
    std::vector<std::vector<float>> bboxes;
    std::vector<float> bbox;
    float xmin, xmax, ymin, ymax, conf, score, prob, cid;
    for (int scale_idx=0; scale_idx<3; ++scale_idx){
        int length = mInputParams.ImgH / mYoloParams.Strides[scale_idx] * mInputParams.ImgW / mYoloParams.Strides[scale_idx] * mYoloParams.AnchorPerScale;
        auto *origin_output = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[scale_idx]));
        for(int i=0; i<length*(5+mYoloParams.NumClass); i+=(5+mYoloParams.NumClass)){
            bbox.clear();
            prob=-1;
            cid=-1;
            xmin = clip<float>(origin_output[i] - origin_output[i+2] * 0.5, 0, static_cast<float>(mInputParams.ImgW-1));
            ymin = clip<float>(origin_output[i+1] - origin_output[i+3] * 0.5, 0, static_cast<float>(mInputParams.ImgH-1));
            xmax = clip<float>(origin_output[i] + origin_output[i+2] * 0.5, 0, static_cast<float>(mInputParams.ImgW-1));
            ymax = clip<float>(origin_output[i+1] + origin_output[i+3] * 0.5, 0, static_cast<float>(mInputParams.ImgH-1));
            conf = origin_output[i+4];

            for (int c=i+5; c<i+5+mYoloParams.NumClass; ++c){
                if (origin_output[c] > prob){
                    prob = origin_output[c];
                    cid = static_cast<float>(c) - 5 - i;
                }
            }
            score = conf * prob;
            if (score>=postThres){
                bbox.emplace_back(xmin);
                bbox.emplace_back(ymin);
                bbox.emplace_back(xmax);
                bbox.emplace_back(ymax);
                bbox.emplace_back(score);
                bbox.emplace_back(cid);
                bboxes.emplace_back(bbox);
            }
        }
    }
    return nms(bboxes, nmsThres);
}

bool Yolo::initSession(int initOrder) {
    if(initOrder==0){
        if(!this->deseriazeEngine(mTrtParams.SerializedPath)){
            if(!this->constructNetwork(mTrtParams.OnnxPath)){
                gLogError << "Init Session Failed!" << std::endl;
            }
            std::ifstream f(mTrtParams.SerializedPath);
            if(!f.good()){
                if(!this->serializeEngine(mTrtParams.SerializedPath)){
                    gLogError << "Init Session Failed!" << std::endl;
                    return false;
                }
            }
        }
    } else if(initOrder==1){
        if(!this->constructNetwork(mTrtParams.OnnxPath)){
            gLogError << "Init Session Failed!" << std::endl;
            return false;
        }
    } else if(initOrder==2){
        if(!this->constructNetwork(mTrtParams.OnnxPath) || this->serializeEngine(mTrtParams.SerializedPath)){
            gLogError << "Init Session Failed!" << std::endl;
            return false;
        }
    }
    return true;
}


std::vector<std::vector<float>> Yolo::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
    float elapsedTime = infer(std::vector<std::vector<float>>{preProcess(std::vector<cv::Mat>{image})}, bufferManager);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    std::vector<std::vector<float>> bboxes = postProcess(bufferManager, postThres, nmsThres);
    if(mInputParams.IsPadding){
        this->transformBbx(image.rows, image.cols, mInputParams.ImgH, mInputParams.ImgW, bboxes);
    }
    return bboxes;
}

void Yolo::transformBbx(const int &ih, const int &iw, const int &oh, const int &ow,
                         std::vector<std::vector<float>> &bboxes) {
    float scale = std::min(static_cast<float>(ow) / static_cast<float>(iw), static_cast<float>(oh) / static_cast<float>(ih));
    int nh = static_cast<int>(scale * static_cast<float>(ih));
    int nw = static_cast<int>(scale * static_cast<float>(iw));
    int dh = (oh - nh) / 2;
    int dw = (ow - nw) / 2;
    for (auto &bbox : bboxes){
        bbox[0] = (bbox[0] - dw) / scale;
        bbox[1] = (bbox[1] - dh) / scale;
        bbox[2] = (bbox[2] - dw) / scale;
        bbox[3] = (bbox[3] - dh) / scale;
    }
}






