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
        int ImgC;
        int ImgH;
        int ImgW;
        int BatchSize;
        bool IsPadding;
        bool HWC;
        // Tensor
        std::vector<std::string> InputTensorNames;
        std::vector<std::string> OutputTensorNames;
        // Image pre-process function
        float(*pFunction)(const unsigned char&);
        InputParams() : ImgC(0), ImgH(0), ImgW(0), BatchSize(0), IsPadding(true), HWC(false), InputTensorNames(),
            OutputTensorNames(), pFunction(nullptr){
        };
    };

    struct TrtParams{
        std::size_t ExtraWorkSpace;
        bool FP32;
        bool FP16;
        bool Int32;
        bool Int8;
        int useDLA;
        int worker;
        int MaxBatch;
        int MinTimingIteration;
        int AvgTimingIteration;
        std::string CalibrationTablePath;
        std::string CalibrationImageDir;
        std::string OnnxPath;
        std::string SerializedPath;
        TrtParams() : ExtraWorkSpace(0), FP32(true), FP16(false), Int32(false), Int8(false), useDLA(-1), worker(0),
            MaxBatch(100), MinTimingIteration(1), AvgTimingIteration(2), CalibrationImageDir(), CalibrationTablePath(),
            OnnxPath(), SerializedPath(){
        };
    };

    struct Anchor{
        float width;
        float height;
        Anchor() : width(0), height(0){
        };
    };

    struct DetectParams{
        // Detection/SegmentationTRT
        std::vector<int> Strides;
        std::vector<common::Anchor> Anchors;
        int AnchorPerScale;
        int NumClass;
        float NMSThreshold;
        float PostThreshold;
        DetectParams() : Strides(), Anchors(), AnchorPerScale(0), NumClass(0), NMSThreshold(0), PostThreshold(0) {
        };
    };

    struct KeypointParams{
        // Hourglass
        int HeatMapH;
        int HeatMapW;
        int NumClass;
        float PostThreshold;
        KeypointParams() : HeatMapH(0), HeatMapW(0), NumClass(0), PostThreshold(0) {
        };
    };

    struct ClassificationParams{
        int NumClass;
        ClassificationParams() : NumClass(0){
        };
    };

    // <============== Outputs =============>
    struct Bbox{
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float score;
        int cid;
        Bbox() : xmin(0), ymin(0), xmax(0), ymax(0), score(0), cid(0) {
        };
    };
    
    struct Keypoint{
        float x;
        float y;
        float score;
        int cid;
        Keypoint() : x(0), y(0), score(0), cid(0) {
        }
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
