// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/13

#include "fcos.h"


FCOS::FCOS(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams detectParams) :
        TensorRT(std::move(inputParams), std::move(trtParams)), mDetectParams(std::move(detectParams)){

}

std::vector<float> FCOS::preProcess(const std::vector<cv::Mat> &images) const {
    std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW,
            mInputParams.IsPadding, mInputParams.pFunction, false);
    return fileData;
}

std::vector<common::Bbox>
FCOS::postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) const {
    assert(mInputParams.BatchSize==1);
    if(postThres<0){
        postThres = mDetectParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mDetectParams.NMSThreshold;
    }
    // 将所有features转换为bboxes [xmin, ymin, xmax, ymax, score, cid]
    std::vector<common::Bbox> bboxes;
    common::Bbox bbox;
    for(int i=0; i<mDetectParams.Strides.size(); ++i){
        std::string cls_name = mInputParams.OutputTensorNames[i*3 + 0];
        std::string reg_name = mInputParams.OutputTensorNames[i*3 + 1];
        std::string cen_name = mInputParams.OutputTensorNames[i*3 + 2];
        int H = mInputParams.ImgH / mDetectParams.Strides[i];
        int W = mInputParams.ImgW / mDetectParams.Strides[i];
        int length = H * W;
        auto *cls_f = static_cast<const float*>(bufferManager.getHostBuffer(cls_name));
        auto *reg_f = static_cast<const float*>(bufferManager.getHostBuffer(reg_name));
        auto *cen_f = static_cast<const float*>(bufferManager.getHostBuffer(cen_name));
        // CHW
        for(int pos=0; pos<length; ++pos){
            int cid = 0;
            float score = 0;
            for(int c=0; c<mDetectParams.NumClass; ++c){
                float tmp = sigmoid(cls_f[pos*length+c]) * cen_f[pos];
                if(tmp > score){
                    cid = c;
                    score = tmp;
                }
            }
            if(score>=postThres){
                int w = pos % W;
                int h = pos / W;
                bbox.xmin = clip(int(w - sigmoid(reg_f[pos])), 0, W-1);
                bbox.ymin= clip(int(h - sigmoid(reg_f[pos+length])), 0, H-1);
                bbox.xmax = clip(int(w + sigmoid(reg_f[pos+length*2])), 0, W-1);
                bbox.ymax = clip(int(h + sigmoid(reg_f[pos+length*3])), 0, H-1);
                bbox.score = score;
                bbox.cid = cid;
                bboxes.emplace_back(bbox);
            }
        }
        // 取前topK个
    }
    // TODO 按类别做nms
    std::sort(bboxes.begin(), bboxes.end(), [&](common::Bbox b1, common::Bbox b2){return b1.score > b2.score;});
    std::vector<int> nms_idx = nms(bboxes, nmsThres);
    std::vector<common::Bbox> bboxes_nms(nms_idx.size());
    for (int i=0; i<nms_idx.size(); ++i){
        bboxes_nms[i] = bboxes[nms_idx[i]];
    }
    return bboxes_nms;
}

bool FCOS::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}

std::vector<common::Bbox> FCOS::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
    float elapsedTime = infer(std::vector<std::vector<float>>{preProcess(std::vector<cv::Mat>{image})}, bufferManager);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    std::vector<common::Bbox> bboxes = postProcess(bufferManager, postThres, nmsThres);
    if(mInputParams.IsPadding){
        this->transformBbox(image.rows, image.cols, mInputParams.ImgH, mInputParams.ImgW, bboxes);
    }
    return bboxes;
}

void FCOS::transformBbox(const int &ih, const int &iw, const int &oh, const int &ow,
                        std::vector<common::Bbox> &bboxes) {
    float scale = std::min(static_cast<float>(ow) / static_cast<float>(iw), static_cast<float>(oh) / static_cast<float>(ih));
    int nh = static_cast<int>(scale * static_cast<float>(ih));
    int nw = static_cast<int>(scale * static_cast<float>(iw));
    int dh = (oh - nh) / 2;
    int dw = (ow - nw) / 2;
    for (auto &bbox : bboxes){
        bbox.xmin = (bbox.xmin - dw) / scale;
        bbox.ymin= (bbox.ymin - dh) / scale;
        bbox.xmax = (bbox.xmax - dw) / scale;
        bbox.ymax = (bbox.ymax - dh) / scale;
    }
}

