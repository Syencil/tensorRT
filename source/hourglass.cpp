// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/16

#include "hourglass.h"

Hourglass::Hourglass(common::InputParams inputParams, common::TrtParams trtParams, common::KeypointParams hourglassParams) : TensorRT(std::move(inputParams), std::move(trtParams)), keypointParams(std::move(hourglassParams)) {

}

std::vector<float> Hourglass::preProcess(const std::vector<cv::Mat> &images) const {
    std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, mInputParams.pFunction);
    return fileData;
}

std::vector<common::Keypoint> Hourglass::postProcess(common::BufferManager &bufferManager, float postThres) const {
    if(postThres<0){
        postThres = keypointParams.PostThreshold;
    }
    assert(mInputParams.OutputTensorNames.size()==1);
    auto *origin_output = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[0]));
    //  Keypoint index transformation idx_x, idx_y, prob
    std::vector<common::Keypoint> keypoints;
    common::Keypoint keypoint;
    for (int c = 0; c < keypointParams.NumClass; ++c){
        int max_idx = -1;
        float max_prob = -1;
        // 输出是HWC
        for (int idx = c; idx < keypointParams.NumClass * keypointParams.HeatMapH * keypointParams.HeatMapW; idx+=keypointParams.NumClass){
            if (origin_output[idx] > max_prob){
                max_idx = idx;
                max_prob = origin_output[idx];
            }
        }
        if (max_prob>=postThres){
            keypoint.x = static_cast<float>(max_idx / keypointParams.NumClass % keypointParams.HeatMapW) / keypointParams.HeatMapW;
            keypoint.y = static_cast<float>((max_idx / keypointParams.NumClass) / keypointParams.HeatMapW) / keypointParams.HeatMapH;
            keypoint.score = max_prob;
            keypoint.cid = c;
            keypoints.emplace_back(keypoint);
        }
    }
    return keypoints;
}

bool Hourglass::initSession(int initOrder) {
   return TensorRT::initSession(initOrder);
}

std::vector<common::Keypoint> Hourglass::predOneImage(const cv::Mat &image, float postThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
    float elapsedTime = infer(std::vector<std::vector<float>>{preProcess(std::vector<cv::Mat>{image})}, bufferManager);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    std::vector<common::Keypoint> keypoints = postProcess(bufferManager, postThres);
    this->transformPoint(image.rows, image.cols, keypoints);
    return keypoints;
}

void Hourglass::transformPoint(const int &h, const int &w, std::vector<common::Keypoint> &keypoints) {
    int pad = abs(h-w) >> 1;
    for(auto &keypoint : keypoints){
        keypoint.x *= w;
        keypoint.y *= h;
        if(mInputParams.IsPadding) {
            keypoint.x = h > w ? keypoint.x - pad : keypoint.x;
            keypoint.y = h > w ? keypoint.y : keypoint.y - pad;
        }
    }
}
