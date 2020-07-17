// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/7/1

#include <queue>

#include "psenet.h"

Psenet::Psenet(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams):
            TensorRT(std::move(inputParams), std::move(trtParams)), mDetectParams(std::move(yoloParams)){

}

std::vector<float> Psenet::preProcess(const std::vector<cv::Mat> &images) const {
    std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, mInputParams.pFunction, false, mTrtParams.worker);
    return fileData;
}

bool Psenet::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}

std::tuple<cv::Mat, std::map<int, std::vector<cv::Point>>> Psenet::postProcess(common::BufferManager &bufferManager, float postThres) {
    if (postThres==-1){
        postThres = mDetectParams.PostThreshold;
    }
    assert(mInputParams.BatchSize==1);
    assert(mInputParams.OutputTensorNames.size()==1);
    assert(mDetectParams.Strides.size()==1);
    assert(postThres<=1 && postThres>=0);

    // BxCxHxW         S0 ===> S5  small ===> large       已经进过sigmoid了
    auto *origin_output = static_cast<const float *>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[0]));
    const int stride = mDetectParams.Strides[0];
    const int num_kernels = mDetectParams.NumClass;
    const int height = (mInputParams.ImgH +stride-1) / stride;
    const int width = (mInputParams.ImgW +stride-1)/ stride;
    const int length = height * width;

    // Mat存储kernel 小的kernel和最大的kernel做logical_and运算
    std::vector<cv::Mat> kernels(num_kernels);
    cv::Mat max_kernel(height, width, CV_32F, (void*)(origin_output+(num_kernels-1)*length), 0);
    cv::threshold(max_kernel, max_kernel, postThres, 255, cv::THRESH_BINARY);
    max_kernel.convertTo(max_kernel, CV_8U);
    assert(max_kernel.rows==height && max_kernel.cols==width);
    for(int i=0; i<num_kernels-1; ++i){
        cv::Mat kernel = cv::Mat(height, width, CV_32F, (void*)(origin_output+i*length), 0);
        cv::threshold(kernel, kernel, postThres, 255, cv::THRESH_BINARY);
        kernel.convertTo(kernel, CV_8U);
        cv::bitwise_and(kernel, max_kernel, kernel);
        assert(kernel.rows==height && kernel.cols==width);
        kernels[i] = kernel;
    }
    kernels[num_kernels-1] = max_kernel;

//    // 渲染每一个kernel
//    cv::imwrite("/work/tensorRT-7/data/image/mask0.jpg", kernels[0]);
//    cv::imwrite("/work/tensorRT-7/data/image/mask1.jpg", kernels[1]);
//    cv::imwrite("/work/tensorRT-7/data/image/mask2.jpg", kernels[2]);
//    cv::imwrite("/work/tensorRT-7/data/image/mask3.jpg", kernels[3]);
//    cv::imwrite("/work/tensorRT-7/data/image/mask4.jpg", kernels[4]);
//    cv::imwrite("/work/tensorRT-7/data/image/mask5.jpg", kernels[5]);

    // 求出所有可能的连通域
    cv::Mat label_image;
    cv::Mat stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(kernels[0], label_image, stats, centroids,4);
    label_image.convertTo(label_image, CV_8U);
    assert(label_image.rows==max_kernel.rows && label_image.cols==max_kernel.cols);
    // 存储结果
    std::map<int, std::vector<cv::Point>> contourMaps;

    // 渐进式扩张算法
    std::queue<std::tuple<int, int, int>> q;
    std::queue<std::tuple<int, int, int>> q_next;
    for(int h=0; h<height; ++h){
        for(int w=0; w<width; ++w){
            auto label = *label_image.ptr(h, w);
            if(label>0){
                q.emplace(std::make_tuple(w, h, label));
                contourMaps[label].emplace_back(cv::Point(w, h));
            }
        }
    }
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};
    for(int idx=1; idx<num_kernels; ++idx){
        auto *ptr_kernel = kernels[idx].data;
        while(!q.empty()){
            auto q_n = q.front();
            q.pop();
            int x = std::get<0>(q_n);
            int y = std::get<1>(q_n);
            int l = std::get<2>(q_n);
            bool is_edge = true;
            for(int j=0; j<4; ++j){
                int tmpx = x + dx[j];
                int tmpy = y + dy[j];
                int offset = tmpy * width + tmpx;
                if (tmpx<0 || tmpx>=width || tmpy<0 || tmpy>=height){
                    continue;
                }
                if (!(int)ptr_kernel[offset] || (int)*label_image.ptr(tmpy, tmpx)>0){
                    continue;
                }
                q.emplace(std::make_tuple(tmpx, tmpy, l));
                *label_image.ptr(tmpy, tmpx) = l;
                contourMaps[l].emplace_back(cv::Point(tmpx, tmpy));
                is_edge = false;
            }
            if(is_edge){
                q_next.emplace(std::make_tuple(x, y, l));
            }
        }
        std::swap(q, q_next);
    }
    // cv::Mat ===> Bbox
    return std::make_tuple(label_image, contourMaps);
}

std::tuple<cv::Mat, std::vector<cv::RotatedRect>> Psenet::predOneImage(const cv::Mat &image, float postThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
    float elapsedTime = infer(std::vector<std::vector<float>>{preProcess(std::vector<cv::Mat>{image})}, bufferManager);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    const auto start_t = std::chrono::high_resolution_clock::now();
    auto result = postProcess(bufferManager, postThres);
    const auto end_t = std::chrono::high_resolution_clock::now();
    cv::Mat mask = std::get<0>(result);
    auto points = std::get<1>(result);
    gLogInfo << "Post time is "<< std::chrono::duration<double, std::milli>(end_t-start_t).count()<<"ms" << std::endl;
    auto RBox = this->point2RBox(points);
    mask = this->transformBbx(image.rows, image.cols, mInputParams.ImgH, mInputParams.ImgW, mask, RBox, mInputParams.IsPadding);
    return std::make_tuple(mask, RBox);
}

cv::Mat Psenet::transformBbx(const int &ih, const int &iw, const int &oh, const int &ow, cv::Mat &mask, std::vector<cv::RotatedRect> &RBox,
                          bool is_padding) {
    cv::Mat out(ih, iw, CV_8U);
    if(is_padding) {
        float scale = std::min(static_cast<float>(ow) / static_cast<float>(iw),
                               static_cast<float>(oh) / static_cast<float>(ih));
        int nh = static_cast<int>(scale * static_cast<float>(ih));
        int nw = static_cast<int>(scale * static_cast<float>(iw));
        int dh = (oh - nh) / 2;
        int dw = (ow - nw) / 2;
        cv::Mat crop_mask = mask(cv::Range(dh, dh + nh), cv::Range(dw, dw + nw));
        cv::resize(crop_mask, out, out.size());

        for (auto & rec : RBox){
            rec.size.width /= scale;
            rec.size.height /= scale;
            rec.center.x = (rec.center.x - dw) / scale;
            rec.center.y = (rec.center.y - dh) / scale;
        }
    }else{
        const cv::Mat& crop_mask (mask);
        cv::resize(crop_mask, out, out.size());
        for (auto & rec : RBox){
            rec.size.width = rec.size.width * iw / ow;
            rec.size.height = rec.size.height * ih / oh;
            rec.center.x = rec.center.x * iw / ow;
            rec.center.y = rec.center.y * ih / oh;
        }
    }
    return out;
}

std::vector<cv::RotatedRect> Psenet::point2RBox(std::map<int, std::vector<cv::Point>> contoursMaps) {
    std::vector<cv::RotatedRect>RBox;
    for (const auto & cnt : contoursMaps){
        cv::Mat bbox;
        cv::RotatedRect rect = cv::minAreaRect(cnt.second);
        RBox.emplace_back(rect);
    }
    return RBox;
}




