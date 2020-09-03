// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/6/17

#include "yolov5.h"

void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 640;
    inputParams.ImgW = 640;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.HWC = false;
    inputParams.IsPadding = true;
    inputParams.InputTensorNames = std::vector<std::string>{"images"};
    inputParams.OutputTensorNames = std::vector<std::string>{"output", "692", "693"};
    inputParams.pFunction = [](const unsigned char &x){return static_cast<float>(x) /255;};
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
    trtParams.CalibrationTablePath = "/work/tensorRT-7/data/yolo5Int8.calibration";
    trtParams.CalibrationImageDir = "/data/dataset/coco/images/train2017";
    trtParams.OnnxPath = "/work/tensorRT-7/data/onnx/yolov5x.onnx";
    trtParams.SerializedPath = "/work/tensorRT-7/data/onnx/yolov5x.serialized";
}

std::vector<common::Anchor> initAnchors(){
    std::vector<common::Anchor> anchors;
    common::Anchor anchor;
    // 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90,  156,198,  373,326
    anchor.width = 10;
    anchor.height = 13;
    anchors.emplace_back(anchor);
    anchor.width = 16;
    anchor.height = 30;
    anchors.emplace_back(anchor);
    anchor.width = 32;
    anchor.height = 23;
    anchors.emplace_back(anchor);
    anchor.width = 30;
    anchor.height = 61;
    anchors.emplace_back(anchor);
    anchor.width = 62;
    anchor.height = 45;
    anchors.emplace_back(anchor);
    anchor.width = 59;
    anchor.height = 119;
    anchors.emplace_back(anchor);
    anchor.width = 116;
    anchor.height = 90;
    anchors.emplace_back(anchor);
    anchor.width = 156;
    anchor.height = 198;
    anchors.emplace_back(anchor);
    anchor.width = 373;
    anchor.height = 326;
    anchors.emplace_back(anchor);
    return anchors;
}

void initDetectParams(common::DetectParams &yoloParams){
    yoloParams.Strides = std::vector<int> {8, 16, 32};
    yoloParams.Anchors = initAnchors();
    yoloParams.AnchorPerScale = 3;
    yoloParams.NumClass = 80;
    yoloParams.NMSThreshold = 0.5;
    yoloParams.PostThreshold = 0.6;
}

int main(int args, char **argv){
    // 46ms ===> 35ms worker=4
    // 21.7fps ===> 28.5fps
    common::InputParams inputParams;
    common::TrtParams trtParams;
    common::DetectParams yoloParams;
    initInputParams(inputParams);
    initTrtParams(trtParams);
    initDetectParams(yoloParams);

    Yolov5 yolo(inputParams, trtParams, yoloParams);
    yolo.initSession(0);

    cv::Mat image = cv::imread("/work/tensorRT-7/data/image/coco_1.jpg");
    std::vector<common::Bbox> bboxes;
    for(int i=0; i<20; ++i){
        const auto start_t = std::chrono::high_resolution_clock::now();
        bboxes = yolo.predOneImage(image);
        const auto end_t = std::chrono::high_resolution_clock::now();
        std::cout
                << "Wall clock time passed: "
                << std::chrono::duration<double, std::milli>(end_t-start_t).count()<<"ms"
                << std::endl;
    }
    image = renderBoundingBox(image, bboxes);
    cv::imwrite("/work/tensorRT-7/data/image/render.jpg", image);
    return 0;
}