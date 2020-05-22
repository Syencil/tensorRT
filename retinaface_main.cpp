// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/20

#include "retinaface.h"

void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 640;
    inputParams.ImgW = 640;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.IsPadding = true;
    inputParams.InputTensorNames = std::vector<std::string>{"input0"};
    inputParams.OutputTensorNames = std::vector<std::string>{"output0", "588", "587"};
    inputParams.pFunction = [](unsigned char x){return static_cast<float>(x) - 118;};
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
    trtParams.CalibrationTablePath = "/work/tensorRT-7/data/retinafaceInt8.calibration";
    trtParams.CalibrationImageDir = "/data/dataset/coco/images/train2017";
    trtParams.OnnxPath = "/work/tensorRT-7/data/onnx/retinaface.onnx";
    trtParams.SerializedPath = "/work/tensorRT-7/data/onnx/retinaface.serialized";
}

std::vector<common::Anchor> initAnchors(){
    std::vector<common::Anchor> anchors;
    common::Anchor anchor;
//    [16, 32, 64, 128, 256, 512]
    anchor.width = anchor.height = 16;
    anchors.emplace_back(anchor);
    anchor.width = anchor.height = 32;
    anchors.emplace_back(anchor);
    anchor.width = anchor.height = 64;
    anchors.emplace_back(anchor);
    anchor.width = anchor.height = 128;
    anchors.emplace_back(anchor);
    anchor.width = anchor.height = 256;
    anchors.emplace_back(anchor);
    anchor.width = anchor.height = 512;
    anchors.emplace_back(anchor);
    return anchors;
}

void initDetectParams(common::DetectParams &detectParams){
    detectParams.Strides = std::vector<int> {8, 16, 32};
    detectParams.AnchorPerScale = 2;
    detectParams.NumClass = 2;
    detectParams.NMSThreshold = 0.5;
    detectParams.PostThreshold = 0.6;
    detectParams.Anchors = initAnchors();
}

int main(int args, char **argv){
    common::InputParams inputParams;
    common::TrtParams trtParams;
    common::DetectParams yoloParams;
    initInputParams(inputParams);
    initTrtParams(trtParams);
    initDetectParams(yoloParams);

    Retinaface retinaface(inputParams, trtParams, yoloParams);
    retinaface.initSession(0);

    cv::Mat image = cv::imread("/work/tensorRT-7/data/image/coco_1.jpg");

    std::vector<common::Bbox> bboxes = retinaface.predOneImage(image);

    image = renderBoundingBox(image, bboxes);
    cv::imwrite("/work/tensorRT-7/data/image/render.jpg", image);
    return 0;
}