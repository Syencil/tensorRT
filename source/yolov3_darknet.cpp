// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2021/1/26

#include <yolov3_darknet.h>

Darknet::Darknet(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams)
        : DetectionTRT(inputParams, trtParams, yoloParams) {

}

std::vector<float> Darknet::preProcess(const std::vector<cv::Mat> &images) {
    return DetectionTRT::preProcess(images);
}

float Darknet::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                     cudaStream_t stream) const {
    return DetectionTRT::infer(InputDatas, bufferManager, stream);
}

std::vector<common::Bbox> Darknet::postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) {
    if(postThres<0){
        postThres = mDetectParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mDetectParams.NMSThreshold;
    }
    assert(mInputParams.BatchSize==1);
    std::vector<common::Bbox> bboxes;
    auto *bbox_ptr = static_cast<const float*> (bufferManager.getHostBuffer(mInputParams.OutputTensorNames[0]));
    auto *conf_ptr = static_cast<const float*> (bufferManager.getHostBuffer(mInputParams.OutputTensorNames[1]));

    unsigned long total_num = 0;
    for (int scale_idx=0; scale_idx<mInputParams.OutputTensorNames.size(); ++scale_idx) {
        const int stride = mDetectParams.Strides[scale_idx];
        const int width = (mInputParams.ImgW + stride - 1) / stride;
        const int height = (mInputParams.ImgH + stride - 1) / stride;
        total_num += height * width * mDetectParams.AnchorPerScale;
    }

    // bbox -> BxNx1x4   conf -> BxNxCls   B=1   N=total_num=416/8x416/8x3+416/16x416/16x3+416/32x416/32x3
    unsigned long min_threads;
    if (mTrtParams.worker < 0) {
        const unsigned long min_length = 64;
        min_threads = (total_num - 1) / min_length + 1;
    } else if (mTrtParams.worker == 0) {
        min_threads = 1;
    } else {
        min_threads = mTrtParams.worker;
    }
    const unsigned long cpu_max_threads = std::thread::hardware_concurrency();
    const unsigned long num_threads = std::min(cpu_max_threads != 0 ? cpu_max_threads : 1, min_threads);
    const unsigned long block_size = total_num / num_threads;
    std::vector<std::future<void>> futures (num_threads - 1);
    unsigned long block_start = 0;
    for (auto &future : futures) {
        future = mThreadPool->submit(&Darknet::postProcessParall, this, block_start, block_size, postThres, bbox_ptr, conf_ptr, &bboxes);
        block_start += block_size;
    }
    this->postProcessParall(block_start, total_num-block_start, postThres, bbox_ptr, conf_ptr, &bboxes);
    for (auto &future : futures){
        future.get();
    }
    nms_cpu(bboxes, nmsThres);
    return bboxes;
}

void Darknet::transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes,
                        bool is_padding) {
    DetectionTRT::transform(ih, iw, oh, ow, bboxes, is_padding);
}

bool Darknet::initSession(int initOrder) {
    return DetectionTRT::initSession(initOrder);
}

std::vector<common::Bbox> Darknet::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    return DetectionTRT::predOneImage(image, postThres, nmsThres);
}

void Darknet::postProcessParall(unsigned long start, unsigned long length, float postThres, const float *bbox_ptr, const float *conf_ptr, std::vector<common::Bbox> *bboxes) {
    int cid;
    float score;
    common::Bbox bbox;
    for(unsigned long pos=start; pos<start+length; ++pos){
        const auto* cptr = conf_ptr + pos * mDetectParams.NumClass;
        const auto* bptr = bbox_ptr + pos * 4;
        cid = argmax(cptr, cptr + mDetectParams.NumClass);
        score = cptr[cid];
        if(score>=postThres){
            bbox.xmin = bptr[0] * mInputParams.ImgW;
            bbox.ymin = bptr[1] * mInputParams.ImgH;
            bbox.xmax = bptr[2] * mInputParams.ImgW;
            bbox.ymax = bptr[3] * mInputParams.ImgH;
            bbox.cid = cid;
            bbox.score = score;
            this->safePushBack(bboxes, &bbox);
        }
    }
}

void Darknet::safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox) {
    std::lock_guard<std::mutex> guard(mMutex);
    (*bboxes).emplace_back((*bbox));
}

