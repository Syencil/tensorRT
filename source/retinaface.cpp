// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/20

#import "retinaface.h"

Retinaface::Retinaface(common::InputParams inputParams, common::TrtParams trtParams,
                       common::DetectParams detectParams) : TensorRT(std::move(inputParams), std::move(trtParams)),
                       mDetectParams(std::move(detectParams)){

}

std::vector<float> Retinaface::preProcess(const std::vector<cv::Mat> &images) const {
    std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, mInputParams.pFunction, false);
    return fileData;
}

std::vector<common::Bbox>
Retinaface::postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) const {
    if(postThres<0){
        postThres = mDetectParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mDetectParams.NMSThreshold;
    }

    std::vector<std::vector<float>> anchors;
    std::vector<float> anchor(4, 0);

    assert(mInputParams.BatchSize==1);
    std::vector<common::Bbox> bboxes;
    common::Bbox bbox;
    auto *loc = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[0]));
    auto *conf = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[1]));
    auto *land = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[2]));

    int pos = 0;
    for(int s=0; s<mDetectParams.Strides.size(); ++s){
        int stride = mDetectParams.Strides[s];
        int height = (mInputParams.ImgH - 1) / stride + 1;
        int width = (mInputParams.ImgW -1) / stride + 1;
        for(int h=0; h<height; ++h){
            for(int w=0; w<width; ++w){
                for(int a=0; a<mDetectParams.AnchorPerScale; ++a){
                    float score = conf[pos*2+1];
                    if(score>=postThres){
                        // bbox
                        float cx_a = (w + 0.5) * stride;
                        float cy_a = (h + 0.5) * stride;
                        float w_a = mDetectParams.Anchors[mDetectParams.AnchorPerScale * s + a].width;
                        float h_a = mDetectParams.Anchors[mDetectParams.AnchorPerScale * s + a].height;
                        float cx_b = cx_a + loc[pos * 4 + 0] * 0.1 * w_a;
                        float cy_b = cy_a + loc[pos * 4 + 1] * 0.1 * h_a;
                        float w_b = w_a * expf(loc[pos * 4 + 2] * 0.2);
                        float h_b = h_a * expf(loc[pos * 4 + 3] * 0.2);
                        bbox.xmin = (cx_b - w_b / 2);
                        bbox.ymin = (cy_b - h_b / 2);
                        bbox.xmax = (cx_b + w_b / 2);
                        bbox.ymax = (cy_b + h_b / 2);
                        bbox.score = score;
                        bbox.cid = 0;
                        bboxes.emplace_back(bbox);
                    }
                    ++pos;
                }
            }
        }
    }
    std::sort(bboxes.begin(), bboxes.end(), [&](common::Bbox &b1, common::Bbox &b2){return b1.score > b2.score;});
    std::vector<int> nms_idx = nms(bboxes, nmsThres);
    std::vector<common::Bbox>  bboxes_nms(nms_idx.size());
    for (int i=0; i<nms_idx.size(); ++i){
        bboxes_nms[i] = bboxes[nms_idx[i]];
    }
    return bboxes_nms;
}

bool Retinaface::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}

std::vector<common::Bbox> Retinaface::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
    float elapsedTime = infer(std::vector<std::vector<float>>{preProcess(std::vector<cv::Mat>{image})}, bufferManager);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    std::vector<common::Bbox> bboxes = postProcess(bufferManager, postThres, nmsThres);
    if(mInputParams.IsPadding){
        this->transformBbx(image.rows, image.cols, mInputParams.ImgH, mInputParams.ImgW, bboxes);
    }
    return bboxes;
}

void Retinaface::transformBbx(const int &ih, const int &iw, const int &oh, const int &ow,
                        std::vector<common::Bbox> &bboxes, bool is_padding) {
    if(is_padding){
        float scale = std::min(static_cast<float>(ow) / static_cast<float>(iw), static_cast<float>(oh) / static_cast<float>(ih));
        int nh = static_cast<int>(scale * static_cast<float>(ih));
        int nw = static_cast<int>(scale * static_cast<float>(iw));
        int dh = (oh - nh) / 2;
        int dw = (ow - nw) / 2;
        for (auto &bbox : bboxes){
            bbox.xmin = (bbox.xmin - dw) / scale;
            bbox.ymin = (bbox.ymin - dh) / scale;
            bbox.xmax = (bbox.xmax - dw) / scale;
            bbox.ymax = (bbox.ymax - dh) / scale;
        }
    }else{
        for (auto &bbox : bboxes){
            bbox.xmin = bbox.xmin * iw / ow;
            bbox.ymin = bbox.ymin * ih / oh;
            bbox.xmax = bbox.xmax * iw / ow;
            bbox.ymax = bbox.ymax * ih / oh;
        }
    }
}




