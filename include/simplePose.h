// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/11/5

#ifndef TENSORRT_SIMPLEPOSE_H
#define TENSORRT_SIMPLEPOSE_H

#include "tensorrt.h"

class SimplePose : public Keypoints{
public:
    SimplePose(common::InputParams inputParams, common::TrtParams trtParams, common::KeypointParams hourglassParams);

    std::vector<common::Keypoint> postProcess(common::BufferManager &bufferManager, float postThres) override;
};

#endif //TENSORRT_SIMPLEPOSE_H
