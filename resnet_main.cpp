// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/6

#include "resnet.h"

void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 224;
    inputParams.ImgW = 224;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.IsPadding = true;
    inputParams.InputTensorNames = std::vector<std::string>{"0"};
    inputParams.OutputTensorNames = std::vector<std::string>{"466"};
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
    trtParams.CalibrationTablePath = "/work/tensorRT-7/data/resnetInt8.calibration";
    trtParams.CalibrationImageDir = "";
    trtParams.OnnxPath = "/work/tensorRT-7/data/onnx/resnet.onnx";
    trtParams.SerializedPath = "/work/tensorRT-7/data/onnx/resnet.serialized";
}

void initClassificationParams(common::ClassificationParams &classifactionParams){
    classifactionParams.NumClass = 4;
}

int getMaxProb(const std::vector<float> &prob){
    int cid = 0;
    float max_prob = 0;
    for(auto i=0; i<prob.size(); ++i){
        printf("cid ===> %d   prob ===> %f\n", i, prob[i]);
        if(max_prob < prob[i]){
            max_prob = prob[i];
            cid = i;
        }
    }
    printf("Cid is %d, Prob is %f\n", cid, max_prob);
}

int main(int args, char **argv){
    common::InputParams inputParams;
    common::TrtParams trtParams;
    common::ClassificationParams classifactionParams;
    initInputParams(inputParams);
    initTrtParams(trtParams);
    initClassificationParams(classifactionParams);

    Resnet resnet(inputParams, trtParams, classifactionParams);
    resnet.initSession(0);

    cv::Mat image = cv::imread("/work/tensorRT-7/data/image/blue.jpg");
    std::vector<float> prob = resnet.predOneImage(image);
    getMaxProb(prob);

    return 0;
}
