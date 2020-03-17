// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/16
#include "yolo.h"

#include <cuda.h>
#include <cublas_v2.h>

//static void HandleError(cudaError_t err, const char *file, int line ) {
//    if (err != cudaSuccess) {
//        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
//                file, line );
//        exit( EXIT_FAILURE );
//    }
//}
//#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask, int bd=5) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;
    const int box_dim = bd;

    const int row_size =
            min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
            min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    extern __shared__ float block_boxes[];

    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * box_dim + 0] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * box_dim + 0];
        block_boxes[threadIdx.x * box_dim + 1] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * box_dim + 1];
        block_boxes[threadIdx.x * box_dim + 2] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * box_dim + 2];
        block_boxes[threadIdx.x * box_dim + 3] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * box_dim + 3];
        block_boxes[threadIdx.x * box_dim + 4] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * box_dim + 4];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float *cur_box = dev_boxes + cur_box_idx * box_dim;
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
            start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (devIoU(cur_box, block_boxes + i * box_dim) > nms_overlap_thresh) {
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }

}

void _set_device(int device_id) {
    int current_device;
    CHECK(cudaGetDevice(&current_device));
    if (current_device == device_id) {
        return;
    }
    // The call to cudaSetDevice must come before any calls to Get, which
    // may perform initialization using the GPU.
    CHECK(cudaSetDevice(device_id));
}

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id) {
    _set_device(device_id);

    float* boxes_dev = NULL;
    unsigned long long* mask_dev = NULL;

    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
    CHECK(cudaMalloc(&boxes_dev,
                            boxes_num * boxes_dim * sizeof(float)));
    CHECK(cudaMemcpy(boxes_dev,
                            boxes_host,
                            boxes_num * boxes_dim * sizeof(float),
                            cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&mask_dev,
                            boxes_num * col_blocks * sizeof(unsigned long long)));

    dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
                DIVUP(boxes_num, threadsPerBlock));
    dim3 threads(threadsPerBlock);
    nms_kernel<<<blocks, threads, threadsPerBlock * boxes_dim * sizeof(float)>>>(boxes_num,
            nms_overlap_thresh,
            boxes_dev,
            mask_dev,
            boxes_dim);
    std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
    CHECK(cudaMemcpy(&mask_host[0],
                            mask_dev,
                            sizeof(unsigned long long) * boxes_num * col_blocks,
                            cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

    int num_to_keep = 0;
    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
            keep_out[num_to_keep++] = i;
            unsigned long long *p = &mask_host[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++) {
                remv[j] |= p[j];
            }
        }
    }
    *num_out = num_to_keep;

    CHECK(cudaFree(boxes_dev));
    CHECK(cudaFree(mask_dev));
}




bool sort_score(std::vector<float> a, std::vector<float> b){
    return a[4] > b[4];
}

std::vector<std::vector<float>> Yolo::nms(std::vector<std::vector<float>> bboxes, float threshold) const {
    std::vector<std::vector<float>> bboxes_nms;
    if (bboxes.empty()) {
        return bboxes_nms;
    }
    // 1.按照score排序
    std::sort(bboxes.begin(), bboxes.end(), sort_score);

    auto *bboxes_1d = new float[bboxes.size() * bboxes[0].size()];
    for (int i = 0; i < bboxes.size(); ++i) {
        for (int j = 0; j < bboxes[i].size(); ++j) {
            bboxes_1d[i * bboxes[i].size() + j] = bboxes[i][j];
        }
    }


    // 2.device malloc cpy
    int *keep_output = new int[bboxes.size()];
    int *num_out = new int;
    _nms(keep_output, num_out, bboxes_1d, bboxes.size(), bboxes[0].size(), threshold, 0);
    for (int i = 0; i < *num_out; ++i) {
        bboxes_nms.push_back(bboxes[keep_output[i]]);

    }
    delete[]bboxes_1d;
    delete[]keep_output;
    delete num_out;
    return bboxes_nms;
}
