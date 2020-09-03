// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/7/13

#include <queue>
#include <utility>

#include "psenetv2.h"


Psenetv2::Psenetv2(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams yoloParams)
        : Segmentation(std::move(inputParams), std::move(trtParams), std::move(yoloParams)) {

}


std::vector<float> Psenetv2::preProcess(const std::vector<cv::Mat> &images) {
    return Segmentation::preProcess(images);
}

float Psenetv2::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                      cudaStream_t stream) const {
    return Segmentation::infer(InputDatas, bufferManager, stream);
}

cv::Mat Psenetv2::postProcess(common::BufferManager &bufferManager, float postThres) {
    if (postThres==-1){
        postThres = mDetectParams.PostThreshold;
    }
    assert(mInputParams.BatchSize==1);
    assert(mInputParams.OutputTensorNames.size()==1);
    assert(mDetectParams.Strides.size()==1);
    assert(postThres<=1 && postThres>=0);

    // BxCxHxW        C = 6    text(C=0)    kernel(C=1)    s_vector(C=2,3,4,5)
    // 没有经过sigmoid这里和PSENet的处理略微不同，不想改了
    auto *origin_output = static_cast<const float *>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[0]));
    const int stride = mDetectParams.Strides[0];
    const int height = (mInputParams.ImgH +stride-1) / stride;
    const int width = (mInputParams.ImgW +stride-1)/ stride;
    const int length = height * width;

    auto tmp_ptr = std::shared_ptr<float>(new float[length]);
    // 1. 求解text部分，sigmoid ===> 二值化
    sigmoid(origin_output, tmp_ptr.get(), length);
    cv::Mat text(height, width, CV_32F, (void*)tmp_ptr.get(), 0);
    cv::threshold(text, text, postThres, 255, cv::THRESH_BINARY);
    text.convertTo(text, CV_8U);
    assert(text.rows==height && text.cols==width);

    // 2. 求解kernel部分，sigmoid ===> 二值化 ===> 和text做logical and运算
    sigmoid(origin_output+length, tmp_ptr.get(), length);
    cv::Mat kernel = cv::Mat(height, width, CV_32F, (void*)tmp_ptr.get(), 0);
    kernel.convertTo(kernel, CV_8U);
    cv::bitwise_and(kernel, text, kernel);
    assert(kernel.rows==height && kernel.cols==width);

    // 3. 取出similarity vectors
    std::vector<cv::Mat> s_vector(4);
    for(int i=0; i<4; ++i){
        s_vector[i] = cv::Mat(height, width, CV_32F, (void*)(origin_output+(i+2)*length), 0);
    }

    // 求出所有可能的连通域
    cv::Mat label_image;
    cv::Mat stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(kernel, label_image, stats, centroids,4);
    assert(num_labels <= 100);
    label_image.convertTo(label_image, CV_8U);
    assert(label_image.rows==text.rows && label_image.cols==text.cols);

    // PAN的基于相似向量的扩张算法
    std::queue<std::tuple<int, int, int>> q;
    std::queue<std::tuple<int, int, int>> q_next;
    float kernel_vector[100][5] = {0};
    for(int h=0; h<height; ++h){
        for(int w=0; w<width; ++w){
            auto label = *label_image.ptr(h, w);
            if(label>0){
                kernel_vector[label][0] += static_cast<float>(*s_vector[0].ptr(h, w));
                kernel_vector[label][1] += static_cast<float>(*s_vector[1].ptr(h, w));
                kernel_vector[label][2] += static_cast<float>(*s_vector[2].ptr(h, w));
                kernel_vector[label][3] += static_cast<float>(*s_vector[3].ptr(h, w));
                kernel_vector[label][4] += 1;
                q.emplace(std::make_tuple(w, h, label));
            }
        }
    }

    for(int i=0; i<num_labels; ++i){
        for (int j=0; j<4; ++j){
            kernel_vector[i][j] /= kernel_vector[i][4];
        }
    }

    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};
    auto *ptr_kernel = kernel.data;
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
            // 计算距离
            float dis = 0;
            for(int i=0; i<4; ++i){
                dis += powf((float)*s_vector[i].ptr(tmpy, tmpx) - kernel_vector[l][i], 2);
            }
            dis = sqrtf(dis);
            if (dis >= 6){
                continue;
            }
            q.emplace(std::make_tuple(tmpx, tmpy, l));
            *label_image.ptr(tmpy, tmpx) = l;
        }
    }
    // cv::Mat ===> Bbox
    return label_image;
}

void Psenetv2::transform(const int &ih, const int &iw, const int &oh, const int &ow, cv::Mat &mask, bool is_padding) {
    Segmentation::transform(ih, iw, oh, ow, mask, is_padding);
}

bool Psenetv2::initSession(int initOrder) {
    return Segmentation::initSession(initOrder);
}

cv::Mat Psenetv2::predOneImage(const cv::Mat &image, float postThres) {
    return Segmentation::predOneImage(image, postThres);
}









