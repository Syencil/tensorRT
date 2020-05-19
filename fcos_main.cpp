// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/13

#include "fcos.h"

void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 1280;
    inputParams.ImgW = 800;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.IsPadding = true;
    inputParams.InputTensorNames = std::vector<std::string>{"input.1"};
    inputParams.OutputTensorNames = std::vector<std::string>{"1077", "1094", "1078", "1107", "1124", "1108", "1137", "1154", "1138", "1167", "1184", "1168", "1197", "1214", "1198"};
    inputParams.pFunction = [](unsigned char x){return static_cast<float>(x) /255;};
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
    trtParams.CalibrationTablePath = "/work/tensorRT-7/data/fcosInt8.calibration";
    trtParams.CalibrationImageDir = "/data/dataset/coco/images/train2017";
    trtParams.OnnxPath = "/work/tensorRT-7/data/onnx/fcos.onnx";
    trtParams.SerializedPath = "/work/tensorRT-7/data/onnx/fcos.serialized";
}

void initDetectParams(common::DetectParams &fcosParams){
    fcosParams.Strides = std::vector<int> {8, 16, 32, 64, 128};
    fcosParams.AnchorPerScale = -1;
    fcosParams.NumClass = 80;
    fcosParams.NMSThreshold = 0.45;
    fcosParams.PostThreshold = 0.3;
}

int main(int args, char **argv){

    common::InputParams inputParams;
    common::TrtParams trtParams;
    common::DetectParams fcosParams;
    initInputParams(inputParams);
    initTrtParams(trtParams);
    initDetectParams(fcosParams);

    FCOS fcos(inputParams, trtParams, fcosParams);
    fcos.initSession(0);

    cv::Mat image = cv::imread("/work/tensorRT-7/data/image/coco_1.jpg");


    std::vector<std::vector<float>> bboxes = fcos.predOneImage(image);

    image = renderBoundingBox(image, bboxes);
    cv::imwrite("/work/tensorRT-7/data/image/render.jpg", image);
    return 0;
}