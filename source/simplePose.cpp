// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/11/5

#include "simplePose.h"

SimplePose::SimplePose(common::InputParams inputParams, common::TrtParams trtParams,
                       common::KeypointParams hourglassParams) : Keypoints(inputParams, trtParams, hourglassParams) {

}

std::vector<common::Keypoint> SimplePose::postProcess(common::BufferManager &bufferManager, float postThres) {
    if(postThres<0){
        postThres = mKeypointParams.PostThreshold;
    }
    assert(mInputParams.OutputTensorNames.size()==1);
    auto *ptr = static_cast<float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[0]));
    std::vector<common::Keypoint> keypoints;
    common::Keypoint keypoint;
    int num_cls = mKeypointParams.NumClass;

    for(int c=0; c<num_cls; ++c){
        float *p = ptr + c * mKeypointParams.HeatMapH * mKeypointParams.HeatMapW;
        float score_m = -1;
        int x_m = 0;
        int y_m = 0;
        for(int h=0; h<mKeypointParams.HeatMapH; ++h) {
            for (int w = 0; w < mKeypointParams.HeatMapW; ++w) {
                if(*p > postThres && *p > score_m){
                    x_m = w;
                    y_m = h;
                    score_m = *p;
                }
                p++;
            }
        }
        if(score_m > -1){
            keypoint.cid = c;
            keypoint.score = score_m;
            keypoint.x = x_m;
            keypoint.y = y_m;
            std::cout << "x: "<< x_m << "  y: "<< y_m << "  cid: "<< c << "  score: " << score_m << std::endl;
            keypoints.emplace_back(keypoint);
        }
    }

    return keypoints;
}
