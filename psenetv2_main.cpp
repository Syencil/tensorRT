// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/7/13

#include "psenetv2.h"

void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 640;
    inputParams.ImgW = 640;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.IsPadding = false;
    inputParams.InputTensorNames = std::vector<std::string>{"input.1"};
    inputParams.OutputTensorNames = std::vector<std::string>{"690"};
    inputParams.pFunction = [](unsigned char &x){return static_cast<float>(x) / 255;};
}

void initTrtParams(common::TrtParams &trtParams){
    trtParams.ExtraWorkSpace = 0;
    trtParams.FP32 = true;
    trtParams.FP16 = false;
    trtParams.Int32 = false;
    trtParams.Int8 = false;
    trtParams.worker = 4;
    trtParams.MaxBatch = 100;
    trtParams.MinTimingIteration = 1;
    trtParams.AvgTimingIteration = 2;
    trtParams.CalibrationTablePath = "/work/tensorRT-7/data/psenetv2Int8.calibration";
    trtParams.CalibrationImageDir = "/data/dataset/ocr/mlt2019/train/images";
    trtParams.OnnxPath = "/work/tensorRT-7/data/onnx/psenetv2.onnx";
    trtParams.SerializedPath = "/work/tensorRT-7/data/onnx/psenetv2.serialized";
}

void initDetectParams(common::DetectParams &detectParams){
    detectParams.NumClass = 6;
    detectParams.Strides = std::vector<int>{1};
    detectParams.PostThreshold = 0.7311;
}

int main(int args, char **argv){

    common::InputParams inputParams;
    common::TrtParams trtParams;
    common::DetectParams detectParams;
    initInputParams(inputParams);
    initTrtParams(trtParams);
    initDetectParams(detectParams);

    Psenetv2 psenetv2(inputParams, trtParams, detectParams);
    psenetv2.initSession(0);

    cv::Mat image = cv::imread("/data/dataset/ocr/icdar/test/images/img_99.jpg");
    const auto start_t = std::chrono::high_resolution_clock::now();
    cv::Mat mask = psenetv2.predOneImage(image);
    const auto end_t = std::chrono::high_resolution_clock::now();
    std::cout
            << "Wall clock time passed: "
            << std::chrono::duration<double, std::milli>(end_t-start_t).count()<<"ms"
            <<std::endl;
    cv::imwrite("/work/tensorRT-7/data/image/render.jpg", renderSegment(image, mask));
    return 0;
}