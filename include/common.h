// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/13

#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

#include <vector>
#include <string>
#include <iostream>
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#define CHECK( err ) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

namespace common{
    // <============== Params =============>
    struct InputParams {
        // General
        int ImgH;
        int ImgW;
        int ImgC;
        int BatchSize;
        bool IsPadding;
        bool HWC;
        // Tensor
        std::vector<std::string> InputTensorNames;
        std::vector<std::string> OutputTensorNames;
        // Image pre-process function
        float(*pFunction)(unsigned char&);
    };

    struct TrtParams{
        std::size_t ExtraWorkSpace;
        bool FP32;
        bool FP16;
        bool Int32;
        bool Int8;
        int worker;
        int MaxBatch;
        int MinTimingIteration;
        int AvgTimingIteration;
        std::string CalibrationTablePath;
        std::string CalibrationImageDir;
        std::string OnnxPath;
        std::string SerializedPath;
    };

    struct Anchor{
        float width;
        float height;
    };

    struct DetectParams{
        // Detection/Segmentation
        std::vector<int> Strides;
        std::vector<common::Anchor> Anchors;
        int AnchorPerScale;
        int NumClass;
        float NMSThreshold;
        float PostThreshold;
    };

    struct KeypointParams{
        // Hourglass
        int HeatMapH;
        int HeatMapW;
        int NumClass;
        float PostThreshold;
    };

    struct ClassificationParams{
        int NumClass;
    };

    // <============== Outputs =============>
    struct Bbox{
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float score;
        int cid;
    };
    
    struct Keypoint{
        float x;
        float y;
        float score;
        int cid;
    };


    // <============== Operator =============>
    struct InferDeleter{
        template <typename T>
        void operator()(T* obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };
}

#endif //TENSORRT_COMMON_H
