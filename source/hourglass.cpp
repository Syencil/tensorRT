// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/16

#include "hourglass.h"

Hourglass::Hourglass(common::InputParams inputParams, common::TrtParams trtParams, common::HourglassParams hourglassParams) : TensorRT(std::move(inputParams), std::move(trtParams)),mHourglassParams(std::move(hourglassParams)) {

}

std::vector<float> Hourglass::preProcess(const std::vector<cv::Mat> &images) const {
    std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, mInputParams.pFunction);
    return fileData;
}

std::vector<std::vector<float>> Hourglass::postProcess(common::BufferManager &bufferManager, float postThres) const {
    if(postThres<0){
        postThres = mHourglassParams.PostThreshold;
    }
    bufferManager.copyOutputToHost();
    assert(mInputParams.OutputTensorNames.size()==1);
    auto *origin_output = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[0]));
    //  Keypoint index transformation idx_x, idx_y, prob
    std::vector<std::vector<float>> keypoints;
    std::vector<float> keypoint;
    for (int c = 0; c < mHourglassParams.NumClass; ++c){
        keypoint.clear();
        int max_idx = -1;
        float max_prob = -1;
        // 输出是HWC
        for (int idx = c; idx < mHourglassParams.NumClass * mHourglassParams.HeatMapH * mHourglassParams.HeatMapW; idx+=mHourglassParams.NumClass){
            if (origin_output[idx] > max_prob){
                max_idx = idx;
                max_prob = origin_output[idx];
            }
        }
        if (max_prob>=postThres){
            keypoint.emplace_back(static_cast<float>(max_idx / mHourglassParams.NumClass % mHourglassParams.HeatMapW)  / mHourglassParams.HeatMapW);
            keypoint.emplace_back(static_cast<float>((max_idx / mHourglassParams.NumClass)  / mHourglassParams.HeatMapW) / mHourglassParams.HeatMapH);
            keypoint.emplace_back(c);
            keypoint.emplace_back(max_prob);
            keypoints.emplace_back(keypoint);
        }
    }
    return keypoints;
}

bool Hourglass::initSession(int initOrder) {
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

std::vector<std::vector<float>> Hourglass::predOneImage(const cv::Mat &image, float posThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
    float elapsedTime = infer(std::vector<std::vector<float>>{preProcess(std::vector<cv::Mat>{image})}, bufferManager);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    std::vector<std::vector<float>> keypoints = postProcess(bufferManager, posThres);
    this->transformPoint(image.rows, image.cols, keypoints);
    return keypoints;
}

void Hourglass::transformPoint(const int &h, const int &w, std::vector<std::vector<float>> &keypoints) {
    int pad = abs(h-w) >> 1;
    for(auto &keypoint : keypoints){
        keypoint[0] *= w;
        keypoint[1] *= h;
        if(mInputParams.IsPadding) {
            keypoint[0] = h > w ? keypoint[0] - pad : keypoint[0];
            keypoint[1] = h > w ? keypoint[1] : keypoint[1] - pad;
        }
    }
}
