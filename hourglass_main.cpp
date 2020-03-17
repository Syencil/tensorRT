// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/15

#include "hourglass.h"

void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 512;
    inputParams.ImgW = 512;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.IsPadding = false;
    inputParams.InputTensorNames = std::vector<std::string>{"Placeholder/inputs_x:0"};
    inputParams.OutputTensorNames = std::vector<std::string>{"Keypoints/keypoint_1/conv/Sigmoid:0"};
    inputParams.pFunction = [](unsigned char x){return static_cast<float>(x) /128-1;};
}

void initTrtParams(common::TrtParams &trtParams){
    trtParams.ExtraWorkSpace = 0;
    trtParams.FP32 = true;
    trtParams.FP16 = false;
    trtParams.Int32 = false;
    trtParams.Int8 = false;
    trtParams.MaxBatch = 100;
    trtParams.MinTimingIteration = 1;
    trtParams.AvgTimingIteration = 2;
    trtParams.CalibrationTablePath = "/work/tensorRT-7/data/houglassInt8.calibration";
    trtParams.CalibrationImageDir = "";
    trtParams.OnnxPath = "/work/tensorRT-7/data/onnx/hourglass.onnx";
    trtParams.SerializedPath = "/work/tensorRT-7/data/onnx/hourglass.serialized";
}

void initHourglassParams(common::HourglassParams &hourglassParams){
    hourglassParams.HeatMapH = 128;
    hourglassParams.HeatMapW = 128;
    hourglassParams.NumClass = 17;
    hourglassParams.PostThreshold = 0.0;
}

int main(int args, char **argv){
    common::InputParams inputParams;
    common::TrtParams trtParams;
    common::HourglassParams hourglassParams;
    initInputParams(inputParams);
    initTrtParams(trtParams);
    initHourglassParams(hourglassParams);

    Hourglass hourglass(inputParams, trtParams, hourglassParams);
    hourglass.initSession(0);

    cv::Mat image = cv::imread("/work/tensorRT-7/data/image/coco_3.jpg");
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    std::vector<std::vector<float>> keypoints = hourglass.predOneImage(image);

    image = renderKeypoint(image, keypoints);
    cv::imwrite("/work/tensorRT-7/data/image/render.jpg", image);
    return 0;
}