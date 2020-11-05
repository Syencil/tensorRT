// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/16

#ifndef TENSORRT_HOURGLASS_H
#define TENSORRT_HOURGLASS_H

#include "tensorrt.h"

class Hourglass : public Keypoints{
public:
        Hourglass(common::InputParams inputParams, common::TrtParams trtParams, common::KeypointParams hourglassParams);

        std::vector<common::Keypoint> postProcess(common::BufferManager &bufferManager, float postThres) override;
};
#endif //TENSORRT_HOURGLASS_H
