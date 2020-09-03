// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/6/17

#include "yolov5.h"

Yolov5::Yolov5(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams) :
        DetectionTRT(std::move(inputParams), std::move(trtParams), std::move(yoloParams)) {

}


std::vector<float> Yolov5::preProcess(const std::vector<cv::Mat> &images) {
    return DetectionTRT::preProcess(images);
}


float Yolov5::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                    cudaStream_t stream) const {
    return DetectionTRT::infer(InputDatas, bufferManager, stream);
}


void Yolov5::safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox) {
    std::lock_guard<std::mutex> guard(mMutex);
    (*bboxes).emplace_back((*bbox));
}


void Yolov5::postProcessParall(unsigned long start, unsigned long length, unsigned long height, unsigned long width, int scale_idx, float postThres, const float *origin_output, std::vector<common::Bbox> *bboxes) {
    common::Bbox bbox;
    float cx, cy, w_b, h_b, score;
    int cid;
    unsigned long pos = start * width * mDetectParams.AnchorPerScale * (5+mDetectParams.NumClass);
    const float *ptr = origin_output + pos;
    for(unsigned long h=start; h<start+length; ++h){
        for(unsigned long w=0; w<width; ++w){
            for(unsigned long a=0; a<mDetectParams.AnchorPerScale; ++a){
                const float *cls_ptr =  ptr + 5;
                cid = argmax(cls_ptr, cls_ptr+mDetectParams.NumClass);
                score = sigmoid(ptr[4]) * sigmoid(cls_ptr[cid]);
                if(score>=postThres){
                    cx = (sigmoid(ptr[0]) * 2.f - 0.5f + static_cast<float>(w)) * static_cast<float>(mDetectParams.Strides[scale_idx]);
                    cy = (sigmoid(ptr[1]) * 2.f - 0.5f + static_cast<float>(h)) * static_cast<float>(mDetectParams.Strides[scale_idx]);
                    w_b = powf(sigmoid(ptr[2]) * 2.f, 2) * mDetectParams.Anchors[scale_idx * mDetectParams.AnchorPerScale + a].width;
                    h_b = powf(sigmoid(ptr[3]) * 2.f, 2) * mDetectParams.Anchors[scale_idx * mDetectParams.AnchorPerScale + a].height;
                    bbox.xmin = clip(cx - w_b / 2, 0.f, static_cast<float>(mInputParams.ImgW - 1));
                    bbox.ymin = clip(cy - h_b / 2, 0.f, static_cast<float>(mInputParams.ImgH - 1));
                    bbox.xmax = clip(cx + w_b / 2, 0.f, static_cast<float>(mInputParams.ImgW - 1));
                    bbox.ymax = clip(cy + h_b / 2, 0.f, static_cast<float>(mInputParams.ImgH - 1));
                    bbox.score = score;
                    bbox.cid = cid;
                    this->safePushBack(bboxes, &bbox);
                }
                ptr += 5 + mDetectParams.NumClass;
            }
        }
    }
}


std::vector<common::Bbox> Yolov5::postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) {
    if(postThres<0){
        postThres = mDetectParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mDetectParams.NMSThreshold;
    }
    assert(mInputParams.BatchSize==1);
    std::vector<common::Bbox> bboxes;
    // 并发执行
    for (int scale_idx=0; scale_idx<mInputParams.OutputTensorNames.size(); ++scale_idx) {
        const int stride = mDetectParams.Strides[scale_idx];
        const int width = (mInputParams.ImgW +stride-1)/ stride;
        const int height = (mInputParams.ImgH +stride-1) / stride;
        auto *origin_output = static_cast<const float *>(bufferManager.getHostBuffer(
                mInputParams.OutputTensorNames[scale_idx]));

        // BHWC decode
        unsigned long min_threads;
        if (mTrtParams.worker < 0) {
            const unsigned long min_length = 64;
            min_threads = (height - 1) / min_length + 1;
        } else if (mTrtParams.worker == 0) {
            min_threads = 1;
        } else {
            min_threads = mTrtParams.worker;
        }
        const unsigned long cpu_max_threads = std::thread::hardware_concurrency();
        const unsigned long num_threads = std::min(cpu_max_threads != 0 ? cpu_max_threads : 1, min_threads);
        const unsigned long block_size = height / num_threads;
        std::vector<std::future<void>> futures (num_threads - 1);
        unsigned long block_start = 0;
        for (auto &future : futures) {
            future = mThreadPool.submit(&Yolov5::postProcessParall, this, block_start, block_size, height, width, scale_idx, postThres, origin_output, &bboxes);
            block_start += block_size;
        }
        this->postProcessParall(block_start, height-block_start, height, width, scale_idx, postThres, origin_output, &bboxes);
        for (auto &future : futures){
            future.get();
        }
    }
    std::sort(bboxes.begin(), bboxes.end(), [&](common::Bbox b1, common::Bbox b2){return b1.score > b2.score;});
    std::vector<int> nms_idx = nms(bboxes, nmsThres);
    std::vector<common::Bbox> bboxes_nms(nms_idx.size());
    for (int i=0; i<nms_idx.size(); ++i){
        bboxes_nms[i] = bboxes[nms_idx[i]];
    }
    return bboxes_nms;
}


void Yolov5::transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes,
                       bool is_padding) {
    DetectionTRT::transform(ih, iw, oh, ow, bboxes, is_padding);
}


bool Yolov5::initSession(int initOrder) {
    return DetectionTRT::initSession(initOrder);
}


std::vector<common::Bbox> Yolov5::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    return DetectionTRT::predOneImage(image, postThres, nmsThres);
}
