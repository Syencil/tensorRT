// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/16

#include "hourglass.h"

Hourglass::Hourglass(common::InputParams inputParams, common::TrtParams trtParams, common::KeypointParams hourglassParams) :
                    Keypoints(std::move(inputParams), std::move(trtParams), std::move(hourglassParams)){

}

std::vector<common::Keypoint> Hourglass::postProcess(common::BufferManager &bufferManager, float postThres) {
    if(postThres<0){
        postThres = mKeypointParams.PostThreshold;
    }
    assert(mInputParams.OutputTensorNames.size()==1);
    auto *origin_output = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[0]));
    //  Keypoint index transformation idx_x, idx_y, prob
    std::vector<common::Keypoint> keypoints;
    common::Keypoint keypoint;
    for (int c = 0; c < mKeypointParams.NumClass; ++c){
        int max_idx = -1;
        float max_prob = -1;
        // 输出是HWC
        for (int idx = c; idx < mKeypointParams.NumClass * mKeypointParams.HeatMapH * mKeypointParams.HeatMapW; idx+=mKeypointParams.NumClass){
            if (origin_output[idx] > max_prob){
                max_idx = idx;
                max_prob = origin_output[idx];
            }
        }
        if (max_prob>=postThres){
            keypoint.x = static_cast<float>(max_idx / mKeypointParams.NumClass % mKeypointParams.HeatMapW);
            keypoint.y = static_cast<float>((max_idx / mKeypointParams.NumClass) / mKeypointParams.HeatMapW);
            keypoint.score = max_prob;
            keypoint.cid = c;
            keypoints.emplace_back(keypoint);
        }
    }
    return keypoints;
}
