// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/15

#include "yolo.h"

void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 512;
    inputParams.ImgW = 512;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.IsPadding = true;
    inputParams.InputTensorNames = std::vector<std::string>{"Placeholder/inputs_x:0"};
    inputParams.OutputTensorNames = std::vector<std::string>{"pred_sbbox/decode:0", "pred_mbbox/decode:0", "pred_lbbox/decode:0"};
    inputParams.pFunction = [](unsigned char x){return static_cast<float>(x) /255;};
}

void initTrtParams(common::TrtParams &trtParams){
    trtParams.ExtraWorkSpace = 0;
    trtParams.FP32 = false;
    trtParams.FP16 = false;
    trtParams.Int32 = false;
    trtParams.Int8 = true;
    trtParams.MaxBatch = 100;
    trtParams.MinTimingIteration = 1;
    trtParams.AvgTimingIteration = 2;
    trtParams.CalibrationTablePath = "/work/tensorRT-7/data/yoloInt8.calibration";
    trtParams.CalibrationImageDir = "/data/dataset/coco/images/train2017";
    trtParams.OnnxPath = "/work/tensorRT-7/data/onnx/yolo.onnx";
    trtParams.SerializedPath = "/work/tensorRT-7/data/onnx/yolo.serialized";
}

void initDetectParams(common::DetectParams &yoloParams){
    yoloParams.Strides = std::vector<int> {8, 16, 32};
    yoloParams.AnchorPerScale = 3;
    yoloParams.NumClass = 80;
    yoloParams.NMSThreshold = 0.45;
    yoloParams.PostThreshold = 0.3;
}

int main(int args, char **argv){
    common::InputParams inputParams;
    common::TrtParams trtParams;
    common::DetectParams yoloParams;
    initInputParams(inputParams);
    initTrtParams(trtParams);
    initDetectParams(yoloParams);

    Yolo yolo(inputParams, trtParams, yoloParams);
    yolo.initSession(0);

    cv::Mat image = cv::imread("/work/tensorRT-7/data/image/coco_1.jpg");

    std::vector<common::Bbox> bboxes = yolo.predOneImage(image);

    image = renderBoundingBox(image, bboxes);
    cv::imwrite("/work/tensorRT-7/data/image/render.jpg", image);
    return 0;
}