// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/15

#include "retinanet.h"

void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 640;
    inputParams.ImgW = 640;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.IsPadding = true;
    inputParams.HWC = false;
    inputParams.InputTensorNames = std::vector<std::string>{"input.1"};
    inputParams.OutputTensorNames = std::vector<std::string>{"815", "816", "841", "842", "867", "868", "893", "894", "919", "920"};
    inputParams.pFunction = [](const unsigned char &x){return (static_cast<float>(x) -118) / 58;};
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
    trtParams.CalibrationTablePath = "/work/tensorRT-7/data/retinanetInt8.calibration";
    trtParams.CalibrationImageDir = "/data/dataset/coco/images/train2017";
    trtParams.OnnxPath = "/work/tensorRT-7/data/onnx/retinanet.onnx";
    trtParams.SerializedPath = "/work/tensorRT-7/data/onnx/retinanet.serialized";
}

std::vector<common::Anchor> initAnchors(float *ratios, int ratio_num, float *scales, int scales_num, float base_size, bool scale_first=true){
    std::vector<common::Anchor> anchors;
    common::Anchor anchor;
    if(scale_first){
        for(int i=0; i<ratio_num; ++i){
            float h_ratio = sqrtf(ratios[i]);
            float w_ratio = 1 / h_ratio;
            for(int j=0; j<scales_num; ++j){
                anchor.width = base_size * w_ratio * scales[j];
                anchor.height = base_size * h_ratio * scales[j];
                anchors.emplace_back(anchor);
            }
        }
    }else{
        for(int i=0; i<scales_num; ++i){
            for(int j=0; j<ratio_num; ++j){
                float h_ratio = sqrtf(ratios[j]);
                float w_ratio = 1 / h_ratio;
                anchor.width = base_size * w_ratio * scales[i];
                anchor.height = base_size * h_ratio * scales[i];
                anchors.emplace_back(anchor);
            }
        }
    }
    return anchors;
}

void initDetectParams(common::DetectParams &detectParams){
    detectParams.Strides = std::vector<int> {8, 16, 32, 64, 128};
    float ratios[3] = {0.5, 1, 2};
    float scales[3] = {4, 5.0397, 6.3496};

    detectParams.Anchors = initAnchors(ratios, 3, scales, 3, 1);
    detectParams.AnchorPerScale = 9;
    detectParams.NumClass = 80;
    detectParams.NMSThreshold = 0.5;
    detectParams.PostThreshold = 0.6;
}

int main(int args, char **argv){

    common::InputParams inputParams;
    common::TrtParams trtParams;
    common::DetectParams detectParams;
    initInputParams(inputParams);
    initTrtParams(trtParams);
    initDetectParams(detectParams);

    RetinaNet retinaNet(inputParams, trtParams, detectParams);
    retinaNet.initSession(0);

    cv::Mat image = cv::imread("/work/tensorRT-7/data/image/coco_1.jpg");
    for(int i=0; i<10; ++i){
        const auto start_t = std::chrono::high_resolution_clock::now();
        std::vector<common::Bbox> bboxes = retinaNet.predOneImage(image);
        const auto end_t = std::chrono::high_resolution_clock::now();
        std::cout
                << "Wall clock time passed: "
                << std::chrono::duration<double, std::milli>(end_t-start_t).count()<<"ms"
                <<std::endl;
        image = renderBoundingBox(image, bboxes);
        cv::imwrite("/work/tensorRT-7/data/image/render.jpg", image);
    }
    return 0;
}