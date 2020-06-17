// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/14

#include "yolov3.h"

Yolov3::Yolov3(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams) :
        TensorRT(std::move(inputParams), std::move(trtParams)), mDetectParams(std::move(yoloParams)) {

}

std::vector<float> Yolov3::preProcess(const std::vector<cv::Mat> &images) const {
    const auto start_t = std::chrono::high_resolution_clock::now();
    std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, mInputParams.pFunction, true, mTrtParams.worker);
    return fileData;
}

std::vector<common::Bbox> Yolov3::postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) {
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
        const int length = height * width * mDetectParams.AnchorPerScale;
        auto *origin_output = static_cast<const float *>(bufferManager.getHostBuffer(
                mInputParams.OutputTensorNames[scale_idx]));

        unsigned long min_threads;
        if (mTrtParams.worker < 0) {
            const unsigned long min_length = 64;
            min_threads = (length - 1) / min_length + 1;
        } else if (mTrtParams.worker == 0) {
            min_threads = 1;
        } else {
            min_threads = mTrtParams.worker;
        }
        const unsigned long cpu_max_threads = std::thread::hardware_concurrency();
        const unsigned long num_threads = std::min(cpu_max_threads != 0 ? cpu_max_threads : 1, min_threads);
        const unsigned long block_size = length / num_threads;
        std::vector<std::thread> threads(num_threads - 1);
        unsigned long block_start = 0;
        for (auto &thread : threads) {
            thread = std::thread(&Yolov3::postProcessParall, this, block_start, block_size, postThres, origin_output, &bboxes);
            block_start += block_size;
        }
        this->postProcessParall(block_start, length-block_start, postThres, origin_output, &bboxes);
        for (auto &thread : threads){
            thread.join();
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

bool Yolov3::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}


std::vector<common::Bbox> Yolov3::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
    float elapsedTime = infer(std::vector<std::vector<float>>{preProcess(std::vector<cv::Mat>{image})}, bufferManager);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    std::vector<common::Bbox> bboxes = postProcess(bufferManager, postThres, nmsThres);
    this->transformBbx(image.rows, image.cols, mInputParams.ImgH, mInputParams.ImgW, bboxes, mInputParams.IsPadding);
    return bboxes;
}

void Yolov3::transformBbx(const int &ih, const int &iw, const int &oh, const int &ow,
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


void Yolov3::safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox) {
    std::lock_guard<std::mutex> guard(mMutex);
    (*bboxes).emplace_back((*bbox));
}


void Yolov3::postProcessParall(unsigned long start, unsigned long length, float postThres, const float *origin_output, std::vector<common::Bbox> *bboxes) {
    common::Bbox bbox;
    for(unsigned long i=start*(5 + mDetectParams.NumClass); i < (start + length) * (5 + mDetectParams.NumClass); i+=(5 + mDetectParams.NumClass)){
        float prob=-1;
        unsigned long cid=-1;
        float conf = origin_output[i+4];
        for (unsigned long c=i+5; c< i + 5 + mDetectParams.NumClass; ++c){
            if (origin_output[c] > prob){
                prob = origin_output[c];
                cid = c - 5 - i;
            }
        }
        float score = conf * prob;
        if (score>=postThres){
            bbox.xmin = clip<float>(origin_output[i] - origin_output[i+2] * 0.5, 0, static_cast<float>(mInputParams.ImgW-1));
            bbox.ymin = clip<float>(origin_output[i+1] - origin_output[i+3] * 0.5, 0, static_cast<float>(mInputParams.ImgH-1));
            bbox.xmax = clip<float>(origin_output[i] + origin_output[i+2] * 0.5, 0, static_cast<float>(mInputParams.ImgW-1));
            bbox.ymax = clip<float>(origin_output[i+1] + origin_output[i+3] * 0.5, 0, static_cast<float>(mInputParams.ImgH-1));
            bbox.score = score;
            bbox.cid = static_cast<int>(cid);
            this->safePushBack(bboxes, &bbox);
        }
    }

}





