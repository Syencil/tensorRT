// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/11/05

#include "simplePose.h"

void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 256;
    inputParams.ImgW = 192;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.HWC = false;
    inputParams.IsPadding = false;
    inputParams.InputTensorNames = std::vector<std::string>{"images"};
    inputParams.OutputTensorNames = std::vector<std::string>{"output"};
    inputParams.pFunction = [](const unsigned char &x){return (static_cast<float>(x)-118) /  58;};
}

void initTrtParams(common::TrtParams &trtParams){
    trtParams.ExtraWorkSpace = 1 << 30;
    trtParams.FP32 = true;
    trtParams.FP16 = false;
    trtParams.Int32 = false;
    trtParams.Int8 = false;
    trtParams.MaxBatch = 100;
    trtParams.MinTimingIteration = 1;
    trtParams.AvgTimingIteration = 2;
    trtParams.CalibrationTablePath = "./data/int8.calibration";
    trtParams.CalibrationImageDir = "";
    trtParams.OnnxPath = "./data/onnx/pose_resnet_50_256x192.onnx";
    trtParams.SerializedPath = "./data/onnx/pose_resnet_50_256x192.serialized";
}

void initKeypointParam(common::KeypointParams &keypointParams){
    keypointParams.HeatMapH = 64;
    keypointParams.HeatMapW = 48;
    keypointParams.NumClass = 17;
    keypointParams.PostThreshold = 0.0;
}

int main(int args, char **argv){
    common::InputParams inputParams;
    common::TrtParams trtParams;
    common::KeypointParams hourglassParams;
    initInputParams(inputParams);
    initTrtParams(trtParams);
    initKeypointParam(hourglassParams);

    SimplePose model(inputParams, trtParams, hourglassParams);
    model.initSession(0);

    cv::Mat image = cv::imread("/work/tensorRT-7/data/image/coco_2.png");
    std::vector<common::Keypoint> keypoints;
    for (int i=0; i<10; ++i){
        const auto start_t = std::chrono::high_resolution_clock::now();
        keypoints = model.predOneImage(image);
        const auto end_t = std::chrono::high_resolution_clock::now();
        std::cout
                << "Wall clock time passed: "
                << std::chrono::duration<double, std::milli>(end_t-start_t).count()<<"ms"
                <<std::endl;
    }
    image = renderKeypoint(image, keypoints);
    cv::imwrite("/work/tensorRT-7/data/image/render.jpg", image);
    return 0;
}