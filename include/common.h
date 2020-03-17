// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/13

#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

#include <vector>
#include <string>
#include <cuda.h>

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

namespace common{
// <============== Params =============>
    struct InputParams {
        // General
        int ImgH;
        int ImgW;
        int ImgC;
        int BatchSize;
        bool IsPadding;
        // Tensor
        std::vector<std::string> InputTensorNames;
        std::vector<std::string> OutputTensorNames;
        // Image pre-process function
        float(*pFunction)(unsigned char);
    };

    struct TrtParams{
        std::size_t ExtraWorkSpace;
        bool FP32;
        bool FP16;
        bool Int32;
        bool Int8;
        int MaxBatch;
        int MinTimingIteration;
        int AvgTimingIteration;
        std::string CalibrationTablePath;
        std::string CalibrationImageDir;
        std::string OnnxPath;
        std::string SerializedPath;
    };

    struct YoloParams{
        // YOLO
        std::vector<int> Strides;
        int AnchorPerScale;
        int NumClass;
        float NMSThreshold;
        float PostThreshold;
    };

    struct HourglassParams{
        // Hourglass
        int HeatMapH;
        int HeatMapW;
        int NumClass;
        float PostThreshold;
    };


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
