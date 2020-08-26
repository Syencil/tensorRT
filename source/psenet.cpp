// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/7/1

#include <queue>
#include "psenet.h"

Psenet::Psenet(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams)
        : Segmentation(std::move(inputParams), std::move(trtParams), std::move(yoloParams)) {

}


std::vector<float> Psenet::preProcess(const std::vector<cv::Mat> &images) const {
    return Segmentation::preProcess(images);
}

float Psenet::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                      cudaStream_t stream) const {
    return Segmentation::infer(InputDatas, bufferManager, stream);
}

cv::Mat Psenet::postProcess(common::BufferManager &bufferManager, float postThres){
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
    return label_image;
}

void Psenet::transform(const int &ih, const int &iw, const int &oh, const int &ow, cv::Mat &mask, bool is_padding) {
    Segmentation::transform(ih, iw, oh, ow, mask, is_padding);
}

bool Psenet::initSession(int initOrder) {
    return Segmentation::initSession(initOrder);
}

cv::Mat Psenet::predOneImage(const cv::Mat &image, float postThres) {
    return Segmentation::predOneImage(image, postThres);
}


