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

std::vector<std::vector<float>>
FCOS::postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) const {
    assert(mInputParams.BatchSize==1);
    if(postThres<0){
        postThres = mDetectParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mDetectParams.NMSThreshold;
    }
    // 将所有features转换为bboxes [xmin, ymin, xmax, ymax, score, cid]
    std::vector<std::vector<float>> bboxes;
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
        std::vector<std::vector<float>> bboxes_t{static_cast<size_t>(length), std::vector<float>{0,0,0,0}};
        for(int pos=0; pos<length; ++pos){
            int w = pos % W;
            int h = pos / W;
            int cid = 0;
            float score = 0;
            bboxes_t[i][0] = clip(int(w - sigmoid(reg_f[pos])), 0, W-1);
            bboxes_t[i][1] = clip(int(h - sigmoid(reg_f[pos+length])), 0, H-1);
            bboxes_t[i][2] = clip(int(w + sigmoid(reg_f[pos+length*2])), 0, W-1);
            bboxes_t[i][3] = clip(int(h + sigmoid(reg_f[pos+length*3])), 0, H-1);
            for(int c=0; c<mDetectParams.NumClass; ++c){
                float tmp = sigmoid(cls_f[pos*length+c]) * cen_f[pos];
                if(tmp > score){
                    cid = c;
                    score = tmp;
                }
            }
            bboxes_t[i][4] = score;
            bboxes_t[i][5] = cid;
        }
        // 取前topK个
        std::sort(bboxes_t.begin(), bboxes_t.end(), [&](std::vector<float> x, std::vector<float> y){return x[4]>y[4];});
        bboxes.insert(bboxes.end(), bboxes_t.begin(), bboxes_t.begin() + std::min(bboxes_t.size(), 100UL));
    }
    // TODO 按类别做nms
    return nms(bboxes, nmsThres);
}

bool FCOS::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}

std::vector<std::vector<float>> FCOS::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
    float elapsedTime = infer(std::vector<std::vector<float>>{preProcess(std::vector<cv::Mat>{image})}, bufferManager);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    std::vector<std::vector<float>> bboxes = postProcess(bufferManager, postThres, nmsThres);
    if(mInputParams.IsPadding){
        this->transformBbx(image.rows, image.cols, mInputParams.ImgH, mInputParams.ImgW, bboxes);
    }
    return bboxes;
}

void FCOS::transformBbx(const int &ih, const int &iw, const int &oh, const int &ow,
                        std::vector<std::vector<float>> &bboxes) {
    float scale = std::min(static_cast<float>(ow) / static_cast<float>(iw), static_cast<float>(oh) / static_cast<float>(ih));
    int nh = static_cast<int>(scale * static_cast<float>(ih));
    int nw = static_cast<int>(scale * static_cast<float>(iw));
    int dh = (oh - nh) / 2;
    int dw = (ow - nw) / 2;
    for (auto &bbox : bboxes){
        bbox[0] = (bbox[0] - dw) / scale;
        bbox[1] = (bbox[1] - dh) / scale;
        bbox[2] = (bbox[2] - dw) / scale;
        bbox[3] = (bbox[3] - dh) / scale;
    }

}

