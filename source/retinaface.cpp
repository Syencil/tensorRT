// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/20

#import "retinaface.h"


Retinaface::Retinaface(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams) :
        DetectionTRT(std::move(inputParams), std::move(trtParams), std::move(yoloParams)) {

}


std::vector<float> Retinaface::preProcess(const std::vector<cv::Mat> &images) {
    return DetectionTRT::preProcess(images);
}


float Retinaface::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                    cudaStream_t stream) const {
    return DetectionTRT::infer(InputDatas, bufferManager, stream);
}


void Retinaface::safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox) {
    std::lock_guard<std::mutex> guard(mMutex);
    (*bboxes).emplace_back((*bbox));
}


void Retinaface::postProcessParall(unsigned long start_h, unsigned long length, unsigned long width,
                                   unsigned long s, unsigned long pos, const float *loc, const float *conf, const float *land, float postThres,
                                   std::vector<common::Bbox> *bboxes) {
    // CHW
    int stride = mDetectParams.Strides[s];
    common::Bbox bbox;
    pos += start_h * width * mDetectParams.AnchorPerScale;
    for(unsigned long h=start_h; h<start_h+length; ++h){
        for(unsigned long w = 0; w<width; ++w){
            for(unsigned long a=0; a<mDetectParams.AnchorPerScale; ++a){
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
                    this->safePushBack(bboxes, &bbox);
                }
                ++pos;
            }
        }
    }
}


std::vector<common::Bbox>
Retinaface::postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) {
    if(postThres<0){
        postThres = mDetectParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mDetectParams.NMSThreshold;
    }
    assert(mInputParams.BatchSize==1);
    std::vector<common::Bbox> bboxes;
    auto *loc = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[0]));
    auto *conf = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[1]));
    auto *land = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[2]));

    unsigned long pos = 0;
    for(unsigned long s=0; s<mDetectParams.Strides.size(); ++s){
        unsigned long stride = mDetectParams.Strides[s];
        unsigned long height = (mInputParams.ImgH - 1) / stride + 1;
        unsigned long width = (mInputParams.ImgW -1) / stride + 1;
        // 并发
        unsigned long min_threads;
        if (mTrtParams.worker<0){
            const unsigned long min_length = 64;
            min_threads = (height - 1) / min_length + 1;
        }else if(mTrtParams.worker==0){
            min_threads = 1;
        }else{
            min_threads = mTrtParams.worker;
        }
        const unsigned long cpu_max_threads = std::thread::hardware_concurrency();
        const unsigned long num_threads = std::min(cpu_max_threads !=0 ? cpu_max_threads : 1, min_threads);
        const unsigned long block_size = height / num_threads;
        std::vector<std::future<void>> futures(num_threads-1);
        unsigned long block_start = 0;
        for(auto & future : futures){
            future = mThreadPool->submit(&Retinaface::postProcessParall, this,  block_start, block_size, width, s, pos, loc, conf, land, postThres, &bboxes);
            block_start += block_size;
        }
        this->postProcessParall(block_start, height - block_start, width, s, pos, loc, conf, land, postThres, &bboxes);
        for(auto & future : futures){
            future.get();
        }
        pos += height * width * mDetectParams.AnchorPerScale;
    }
    std::sort(bboxes.begin(), bboxes.end(), [&](common::Bbox &b1, common::Bbox &b2){return b1.score > b2.score;});
    std::vector<int> nms_idx = nms(bboxes, nmsThres);
    std::vector<common::Bbox>  bboxes_nms(nms_idx.size());
    for (int i=0; i<nms_idx.size(); ++i){
        bboxes_nms[i] = bboxes[nms_idx[i]];
    }
    return bboxes_nms;
}


void Retinaface::transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes,
                       bool is_padding) {
    DetectionTRT::transform(ih, iw, oh, ow, bboxes, is_padding);
}


bool Retinaface::initSession(int initOrder) {
    return DetectionTRT::initSession(initOrder);
}


std::vector<common::Bbox> Retinaface::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    return DetectionTRT::predOneImage(image, postThres, nmsThres);
}









