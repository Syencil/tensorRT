// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/15

#include "retinanet.h"

RetinaNet::RetinaNet(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams) :
        DetectionTRT(std::move(inputParams), std::move(trtParams), std::move(yoloParams)) {

}


std::vector<float> RetinaNet::preProcess(const std::vector<cv::Mat> &images) const {
    return DetectionTRT::preProcess(images);
}


float RetinaNet::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                    cudaStream_t stream) const {
    return DetectionTRT::infer(InputDatas, bufferManager, stream);
}


std::vector<common::Bbox>
RetinaNet::postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) {
    assert(mInputParams.BatchSize==1);
    if(postThres<0){
        postThres = mDetectParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mDetectParams.NMSThreshold;
    }
    std::vector<common::Bbox> bboxes_total;

    for(int stride_idx=0; stride_idx<mDetectParams.Strides.size(); ++stride_idx){
        std::vector<common::Bbox> bboxes_t;
        // 配置参数
        const int stride = mDetectParams.Strides[stride_idx];
        const int width = (mInputParams.ImgW +stride-1)/ stride;
        const int height = (mInputParams.ImgH +stride-1) / stride;
        const int length = height * width;

        // CHW
        auto *cls_f = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[stride_idx*2+0]));
        auto *reg_f = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[stride_idx*2+1]));

        // Decode
        for(int h=0; h<height; ++h){
            for(int w=0; w<width; ++w){
                int pos = h * width + w;
                float cx_a = w * stride;
                float cy_a = h * stride;
                for(int a=0; a<mDetectParams.AnchorPerScale; ++a){
                    int cid = -1;
                    float score = -1;
                    for(int c=0; c<mDetectParams.NumClass; ++c){
                        float score_t = sigmoid(cls_f[(a * mDetectParams.NumClass  + c) * length + pos]);
                        if(score_t > score){
                            score = score_t;
                            cid = c;
                        }
                    }
                    if(score >= postThres){
                        float cx_b = reg_f[(a * 4 + 0) * length + pos] * (mDetectParams.Anchors[a].width) * stride + cx_a;
                        float cy_b = reg_f[(a * 4 + 1) * length + h * width + w] * (mDetectParams.Anchors[a].height) * stride + cy_a;
                        float w_b = expf(reg_f[(a * 4 + 2) * length + h * width + w]) * (mDetectParams.Anchors[a].width) * stride;
                        float h_b = expf(reg_f[(a * 4 + 3) * length + h * width + w]) * (mDetectParams.Anchors[a].height) * stride;
                        float xmin = cx_b - w_b / 2;
                        float ymin = cy_b - h_b / 2;
                        float xmax = cx_b + w_b / 2;
                        float ymax = cy_b + h_b / 2;
                        if (xmin > -5 && ymin > -5 && xmax <mInputParams.ImgW + 4 && ymax < mInputParams.ImgH + 4){
                            common::Bbox bbox;
                            bbox.xmin = clip(xmin, 0.f, static_cast<float>(mInputParams.ImgW));
                            bbox.ymin = clip(ymin, 0.f, static_cast<float>(mInputParams.ImgH));;
                            bbox.xmax = clip(xmax, 0.f, static_cast<float>(mInputParams.ImgW));;
                            bbox.ymax = clip(ymax, 0.f, static_cast<float>(mInputParams.ImgH));;
                            bbox.score = score;
                            bbox.cid = cid;
                            bboxes_total.emplace_back(bbox);
                        }
                    }
                }
            }
        }
    }
    std::sort(bboxes_total.begin(), bboxes_total.end(), [&](common::Bbox b1, common::Bbox b2){return b1.score > b2.score;});
    std::vector<int> nms_idx = nms(bboxes_total, nmsThres);
    std::vector<common::Bbox> bboxes_nms(nms_idx.size());
    for (int i=0; i<nms_idx.size(); ++i){
        bboxes_nms[i] = bboxes_total[nms_idx[i]];
    }
    return bboxes_nms;
}


void RetinaNet::transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes,
                       bool is_padding) {
    DetectionTRT::transform(ih, iw, oh, ow, bboxes, is_padding);
}


bool RetinaNet::initSession(int initOrder) {
    return DetectionTRT::initSession(initOrder);
}


std::vector<common::Bbox> RetinaNet::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    return DetectionTRT::predOneImage(image, postThres, nmsThres);
}