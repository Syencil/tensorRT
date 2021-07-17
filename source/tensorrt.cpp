// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/10

#include "tensorrt.h"
#include <climits>
#include <memory>


TensorRT::TensorRT(common::InputParams inputParams, common::TrtParams trtParams)
                : mInputParams(std::move(inputParams)), mTrtParams(std::move(trtParams)) , mThreadPool(new tss::thread_pool(trtParams.worker)){
    CHECK(cudaEventCreate(&this->start_t));
    CHECK(cudaEventCreate(&this->stop_t));
}


bool TensorRT::constructNetwork(const std::string &onnxPath) {
    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder){
        gLogError << "Create Builder Failed" << std::endl;
        return false;
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network){
        gLogError << "Create Network Failed" << std::endl;
        return false;
    }
    auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config){
        gLogError << "Create Config Failed" << std::endl;
        return false;
    }
    auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser){
        gLogError << "Create Parser Failed" << std::endl;
        return false;
    }
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(gLogger.getReportableSeverity()))){
        gLogError << "Parsing File Failed" << std::endl;
        return false;
    }
    builder->setMaxBatchSize(mInputParams.BatchSize);
    config->setMaxWorkspaceSize(mTrtParams.ExtraWorkSpace);

    if (mTrtParams.FP16){
        if(!builder->platformHasFastFp16()){
            gLogWarning << "Platform has not fast Fp16! It will still using Fp32!"<< std::endl;
        }else{
            config -> setFlag(nvinfer1::BuilderFlag::kFP16);
        }
    }

    if (mTrtParams.Int8){
        if(!builder->platformHasFastInt8()){
            gLogWarning << "Platform has not fast Fp16! It will still using Fp32!"<< std::endl;
        }else{
            config->setAvgTimingIterations(mTrtParams.AvgTimingIteration);
            config->setMinTimingIterations(mTrtParams.MinTimingIteration);
            config -> setFlag(nvinfer1::BuilderFlag ::kINT8);
            std::shared_ptr<EntropyCalibratorV2*> calibrator = std::make_shared<EntropyCalibratorV2*>(new EntropyCalibratorV2(mInputParams, mTrtParams)) ;
            config->setInt8Calibrator(*calibrator.get());
        }
    }

    if (mTrtParams.useDLA >= 0){
        if (builder->getNbDLACores() == 0){
            gLogWarning << "Platform has no DLA Core. It will fall back to gpu" << std::endl;
        }
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8)){
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(mTrtParams.useDLA);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }

    mCudaEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config),common::InferDeleter());
    if (!mCudaEngine){
        gLogError << "Create Engine Failed" << std::endl;
        return false;
    }
    mContext = UniquePtr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());
    if(!mContext){
        gLogError << "Create Context Failed" << std::endl;
        return false;
    }
    assert(network->getNbInputs()==mInputParams.InputTensorNames.size());
    assert(network->getNbOutputs()==mInputParams.OutputTensorNames.size());
    return true;
}


bool TensorRT::serializeEngine(const std::string &save_path) {
    nvinfer1::IHostMemory *gie_model_stream = mCudaEngine -> serialize();
    std::ofstream serialize_output_stream;
    std::string serialize_str;
    serialize_str.resize(gie_model_stream->size());
    memcpy((void*)serialize_str.data(),gie_model_stream->data(), gie_model_stream->size());
    serialize_output_stream.open(save_path);
    if(!serialize_output_stream.good()){
        gLogError << "Serializing Engine Failed" << std::endl;
        return false;
    }
    serialize_output_stream<<serialize_str;
    serialize_output_stream.close();
    return true;
}


bool TensorRT::deseriazeEngine(const std::string &load_path) {
    std::ifstream fin(load_path);
    if (!fin.good()){
        return false;
    }
    std::string deserialize_str;
    while (fin.peek() != EOF){ // 使用fin.peek()防止文件读取时无限循环
        std::stringstream buffer;
        buffer << fin.rdbuf();
        deserialize_str.append(buffer.str());
    }
    fin.close();
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    mCudaEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(deserialize_str.data(), deserialize_str.size()), common::InferDeleter());
    mContext = UniquePtr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());
    if(!mContext){
        gLogError << "Create Context Failed" << std::endl;
        return false;
    }
    return true;
}


void TensorRT::resize_bilinear_c3_parrall(unsigned char *dst, int w, int block_start, int block_size, int stride, const unsigned char *src, int srcstride, const int *xofs, const int *yofs, const short *ialpha, const short *ibeta){
    // loop body 参考boxfilter
    // 1. 设置缓存区域 记录row上和row下
    auto rowsbuf0 = std::shared_ptr<short>(new short[w * 3 + 1]);
    auto rowsbuf1 = std::shared_ptr<short>(new short[w * 3 + 1]);
    short *rows0 = rowsbuf0.get();
    short *rows1 = rowsbuf1.get();
    int prev_sy1 = -2;  // 复用row向量
    for(int dy=block_start; dy<block_start + block_size; ++dy){
        // 2. 先计算行向量 resize w
        int sy = yofs[dy];  // 取出对应位置的sy
        if (sy==prev_sy1){  // 判断rows是否可以复用
            // 复用所有行向量 rowsbuf0 和 rowsbuf1
        }else if (sy==prev_sy1+1){  // sy = -1
            // rowsbuf0 = rowsbuf1 只用单独计算rowsbuf1即可
            short *rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = src + srcstride * (sy + 1);
            const short *ialphap = ialpha;
            short *rows1p = rows1;
            for(int dx=0; dx<w; ++dx){
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];
                const unsigned char *S1p = S1 + sx;
                rows1p[0] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[4]  * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[5]  * a1) >> 4;
                ialphap += 2;
                rows1p += 3;
            }
        }else{
            const unsigned char *S0 = src + srcstride * sy;  // 上一行像素点 sy = yofs[dy] = floor(fy)
            const unsigned char *S1 = src + srcstride * (sy + 1);  // 下一行像素点 sy + 1 = yofs[dy] + 1 = ceil(fy)
            const short* ialphap = ialpha;
            short *rows0p = rows0;
            short *rows1p = rows1;
            for(int dx=0; dx<w; ++dx){
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];
                const unsigned char *S0p = S0 + sx;
                const unsigned char *S1p = S1 + sx;
                rows0p[0] = (S0p[0] * a0 + S0p[3] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[4] * a1) >> 4;
                rows0p[2] = (S0p[2] * a0 + S0p[5] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;
                ialphap += 2;
                rows0p += 3;
                rows1p += 3;
            }
        }
        prev_sy1 = sy;

        // 3. 计算列向量 resize h
        short b0 = ibeta[0];
        short b1 = ibeta[1];
        short *rows0p = rows0;
        short *rows1p = rows1;
        unsigned char* Dp = dst + stride * dy;

        for (int i=0; i<w*3; ++i){
            *Dp++ = static_cast<unsigned char>(((b0 * (*rows0p++) >> 16) + (b1 * (*rows1p++) >> 16) + 2) >> 2);
        }
        ibeta += 2;
    }
}

/* 考虑一个1-d的tensor进行bilinear resize，操作总共可以分为4个大步骤 half_pixel采样方式
 * 设输入tensor长为length_in，输出tensor长为length_out，scale = length_in / length_out
 * for dx in [0, length_out):
 *     1. 得到第dx个点在输出tensor上的采样坐标w(dx) = dx + 0.5
 *     2. 计算输出采样坐标对应输入的采样坐标x = w(dx) * scale，对应输入tensor上的坐标fx = x - 0.5  = (dx + 0.5) * scale - 0.5
 *     3. 找到fx左边距离最近的点sx = int(floor(fx))，将fx = fx - sx作为采样点的权重
 *     4. 计算系列fx和sx的加权平均，更新输出点i的值
 * */
void TensorRT::resize_bilinear_c3(const unsigned char *src, int srcw, int srch, unsigned char *dst, int w, int h) {
    // 参考ncnn和opencv
    int srcstride = srcw * 3;
    int stride = w * 3;
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;  // 放大系数 将double转换为short进行计算

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    auto buf = std::shared_ptr<int>(new int[w + h + w + h]);
    int *xofs = buf.get();  // new int[w]
    int *yofs = buf.get() + w;  // new int[h]
    auto ialpha = (short*)(buf.get() + w + h);  // new short[w*2]
    auto ibeta = (short*)(buf.get() + w + h + w);  // new short[h*2]
#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);
    for(int dx=0; dx<w; ++dx){
        auto fx = (float)((dx + 0.5) * scale_x - 0.5);
        int sx = static_cast<int>(floor(fx));
        fx -= sx;
        // 对sx进行clip 舍弃
        if(sx < 0){  // 输出的第0个采样点落在输入第0个采样点左侧 矫正后选取0和1号点 fx=0则只用0号点计算
            sx = 0;
            fx = 0.f;
        }
        if(sx >= srcw - 1){
            sx = srcw - 2;  // 选右边点
            fx = 1.f;
        }
        xofs[dx] = sx * 3;  // channel = 3 hwc的存储格式
        float a0 = (1 - fx) * INTER_RESIZE_COEF_SCALE;  // 提升精度用的
        float a1 = fx * INTER_RESIZE_COEF_SCALE;
        ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);  // 将权重值存入ialpha里面
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }
    for(int dy=0; dy<h; ++dy){
        auto fy = (float)((dy + 0.5) * scale_y - 0.5);
        int sy = static_cast<int>(floor(fy));
        fy -= sy;
        if (sy < 0){
            sy = 0;
            fy = 0.f;
        }
        if (sy >= srch - 1){
            sy = srch - 2;
            fy = 1.f;
        }
        yofs[dy] = sy;
        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;
        ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }
#undef SATURATE_CAST_SHORT
    int min_threads;
    if (mTrtParams.worker < 0) {
        const unsigned long min_length = 64;
        min_threads = (h - 1) / min_length + 1;
    } else if (mTrtParams.worker == 0) {
        min_threads = 1;
    } else {
        min_threads = mTrtParams.worker;
    }
    const int cpu_max_threads = std::thread::hardware_concurrency();
    const int num_threads = std::min(cpu_max_threads != 0 ? cpu_max_threads : 1, min_threads);
    const int block_size = h / num_threads;
    std::vector<std::future<void>> futures (num_threads - 1);
    int block_start = 0;
    for (auto &future : futures) {
        future = mThreadPool->submit(&TensorRT::resize_bilinear_c3_parrall, this, dst, w, block_start, block_size, stride, src, srcstride, xofs, yofs, ialpha, ibeta);
        block_start += block_size;
    }
    this->resize_bilinear_c3_parrall(dst, w, block_start, h - block_start, stride, src, srcstride, xofs, yofs, ialpha, ibeta);
    for (auto &future : futures){
        future.get();
    }
}


void TensorRT::pixel_convert_parrall(const unsigned char *src, int h_start, int len, float *dst) {
    auto *pFunc = mInputParams.pFunction;
    const unsigned char *Sp = src + h_start * mInputParams.ImgW * 3;
    if(mInputParams.HWC){
        auto *Dp = dst + h_start * mInputParams.ImgW * 3;
        for(int h=h_start; h<h_start+len; ++h){
            for(int w=0; w<mInputParams.ImgW; ++w){
                Dp[2] = (*pFunc)(Sp[0]);
                Dp[1] = (*pFunc)(Sp[1]);
                Dp[0] = (*pFunc)(Sp[2]);
                Dp += 3;
                Sp += 3;
            }
        }
    }else{
        int img_len = mInputParams.ImgH * mInputParams.ImgW;
        int ofs = h_start * mInputParams.ImgW;
        float *Dp0 = dst + 0 * img_len + ofs;
        float *Dp1 = dst + 1 * img_len + ofs;
        float *Dp2 = dst + 2 * img_len + ofs;
        for(int h=h_start; h<h_start+len; ++h){
            for(int w=0; w<mInputParams.ImgW; ++w){
                *Dp2++ = (*pFunc)(Sp[0]);
                *Dp1++ = (*pFunc)(Sp[1]);
                *Dp0++ = (*pFunc)(Sp[2]);
                Sp += 3;
            }
        }
    }
}


void TensorRT::pixel_convert(const unsigned char *src, float *dst){
    int min_threads;
    if (mTrtParams.worker < 0) {
        const unsigned long min_length = 64;
        min_threads = (mInputParams.ImgH - 1) / min_length + 1;
    } else if (mTrtParams.worker == 0) {
        min_threads = 1;
    } else {
        min_threads = mTrtParams.worker;
    }
    const int cpu_max_threads = std::thread::hardware_concurrency();
    const int num_threads = std::min(cpu_max_threads != 0 ? cpu_max_threads : 1, min_threads);
    const int block_size = mInputParams.ImgH / num_threads;
    std::vector<std::future<void>> futures (num_threads - 1);
    int block_start = 0;
    for (auto &future : futures) {
        future = mThreadPool->submit(&TensorRT::pixel_convert_parrall, this, src, block_start, block_size, dst);
        block_start += block_size;
    }
    this->pixel_convert_parrall(src, block_start, mInputParams.ImgH-block_start, dst);
    for (auto &future : futures){
        future.get();
    }
}


float TensorRT::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager, cudaStream_t stream) const {
    assert(InputDatas.size()==mInputParams.InputTensorNames.size());
    for(int i=0; i<InputDatas.size(); ++i){
        std::memcpy((void*)bufferManager.getHostBuffer(mInputParams.InputTensorNames[i]), (void*)InputDatas[i].data(), InputDatas[i].size() * sizeof(float));
    }
    bufferManager.copyInputToDeviceAsync();
    CHECK(cudaEventRecord(this->start_t, stream));
    if (!mContext->enqueueV2(bufferManager.getDeviceBindings().data(), stream, nullptr)) {
        gLogError << "Execute Failed!" << std::endl;
        return false;
    }
    CHECK(cudaEventRecord(this->stop_t, stream));
    bufferManager.copyOutputToHostAsync();
    float elapsed_time;
    CHECK(cudaEventSynchronize(this->stop_t));
    CHECK(cudaEventElapsedTime(&elapsed_time, this->start_t, this->stop_t));
    return elapsed_time;
}


bool TensorRT::initSession(int initOrder) {
    if(initOrder==0){
        if(!this->deseriazeEngine(mTrtParams.SerializedPath)){
            if(!this->constructNetwork(mTrtParams.OnnxPath)){
                gLogError << "Init Session Failed!" << std::endl;
            }
            std::ifstream f(mTrtParams.SerializedPath);
            if(!f.good()){
                if(!this->serializeEngine(mTrtParams.SerializedPath)){
                    gLogError << "Init Session Failed!" << std::endl;
                    return false;
                }
            }
        }
    } else if(initOrder==1){
        if(!this->constructNetwork(mTrtParams.OnnxPath)){
            gLogError << "Init Session Failed!" << std::endl;
            return false;
        }
    } else if(initOrder==2){
        if(!this->constructNetwork(mTrtParams.OnnxPath) || this->serializeEngine(mTrtParams.SerializedPath)){
            gLogError << "Init Session Failed!" << std::endl;
            return false;
        }
    }
    return true;
}


TensorRT::~TensorRT() {
    CHECK(cudaEventDestroy(this->start_t));
    CHECK(cudaEventDestroy(this->stop_t));
}


std::vector<float> TensorRT::preProcess(const std::vector<cv::Mat> &images) {
    const unsigned long len_img = mInputParams.ImgH * mInputParams.ImgW * 3;
    std::vector<float> files_data(len_img * images.size());
    for(auto img_count=0; img_count<images.size(); ++img_count){
        auto image = images[img_count];
        cv::Mat image_processed(mInputParams.ImgH, mInputParams.ImgW, CV_8UC3);
        int dh = 0;
        int dw = 0;
        if(mInputParams.IsPadding){
            int ih = image.rows;
            int iw = image.cols;
            float scale = std::min(static_cast<float>(mInputParams.ImgW) / static_cast<float>(iw), static_cast<float>(mInputParams.ImgH) / static_cast<float>(ih));
            int nh = static_cast<int>(scale * static_cast<float>(ih));
            int nw = static_cast<int>(scale * static_cast<float>(iw));
            dh = (mInputParams.ImgH - nh) / 2;
            dw = (mInputParams.ImgW - nw) / 2;
            cv::Mat image_resized = cv::Mat(nh, nw, CV_8UC3);
            this->resize_bilinear_c3(image.data, image.cols, image.rows, image_resized.data, nw, nh);
            cv::copyMakeBorder(image_resized, image_processed, dh, mInputParams.ImgH-nh-dh, dw, mInputParams.ImgW-nw-dw, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
        }else{
            this->resize_bilinear_c3(image.data, image.cols, image.rows, image_processed.data, mInputParams.ImgW, mInputParams.ImgH);
        }
        std::vector<unsigned char> file_data = image_processed.reshape(1, 1);
        this->pixel_convert(file_data.data(), files_data.data() + img_count * len_img);
    }

    return files_data;
}

// ============== DetectionTRT =====================>

DetectionTRT::DetectionTRT(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams detectParams) :
                TensorRT(std::move(inputParams), std::move(trtParams)), mDetectParams(std::move(detectParams)) {

}

std::vector<float> DetectionTRT::preProcess(const std::vector<cv::Mat> &images) {
    return TensorRT::preProcess(images);
}

float DetectionTRT::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                          cudaStream_t stream) const {
    return TensorRT::infer(InputDatas, bufferManager, stream);
}

bool DetectionTRT::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}

std::vector<common::Bbox> DetectionTRT::predOneImage(const cv::Mat &image, float postThres, float nmsThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);

    Clock<std::chrono::high_resolution_clock > clock_t;

    clock_t.tick();
    auto preImg = preProcess(std::vector<cv::Mat>{image});
    clock_t.tock();
    gLogInfo << "Pre Process time is " << clock_t.duration<double>() << "ms"<< std::endl;

    float elapsedTime = this->infer(std::vector<std::vector<float>>{preImg}, bufferManager, nullptr);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;

    clock_t.tick();
    std::vector<common::Bbox> bboxes = postProcess(bufferManager, postThres, nmsThres);
    clock_t.tock();
    gLogInfo << "Post Process time is " << clock_t.duration<double>() << "ms"<< std::endl;

    this->transform(image.rows, image.cols, mInputParams.ImgH, mInputParams.ImgW, bboxes, mInputParams.IsPadding);
    return bboxes;
}

void DetectionTRT::transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes,
                          bool is_padding) {
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

// ============== SegmentationTRT =====================>

SegmentationTRT::SegmentationTRT(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams detectParams) :
        TensorRT(std::move(inputParams), std::move(trtParams)), mDetectParams(std::move(detectParams)) {

}

std::vector<float> SegmentationTRT::preProcess(const std::vector<cv::Mat> &images) {
    return TensorRT::preProcess(images);
}

float SegmentationTRT::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                             cudaStream_t stream) const {
    return TensorRT::infer(InputDatas, bufferManager, stream);
}

void SegmentationTRT::transform(const int &ih, const int &iw, const int &oh, const int &ow, cv::Mat &mask, bool is_padding) {
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
    }else{
        const cv::Mat& crop_mask (mask);
        cv::resize(crop_mask, out, out.size());
    }
    mask = out;
}

bool SegmentationTRT::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}

cv::Mat SegmentationTRT::predOneImage(const cv::Mat &image, float postThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
    Clock<std::chrono::high_resolution_clock > clock_t;
    clock_t.tick();
    auto preImg = preProcess(std::vector<cv::Mat>{image});
    clock_t.tock();
    gLogInfo << "Pre Process time is " << clock_t.duration<double>() << "ms"<< std::endl;
    float elapsedTime = this->infer(std::vector<std::vector<float>>{preImg}, bufferManager, nullptr);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    clock_t.tick();
    cv::Mat mask = postProcess(bufferManager, postThres);
    clock_t.tock();
    gLogInfo << "Post Process time is " << clock_t.duration<double>() << "ms"<< std::endl;
    this->transform(image.rows, image.cols, mInputParams.ImgH, mInputParams.ImgW, mask, mInputParams.IsPadding);
    return mask;
}

// ============== KeypointsTRT =====================>

Keypoints::Keypoints(common::InputParams inputParams, common::TrtParams trtParams, common::KeypointParams keypointParams) :
        TensorRT(std::move(inputParams), std::move(trtParams)), mKeypointParams(std::move(keypointParams)) {

}

std::vector<float> Keypoints::preProcess(const std::vector<cv::Mat> &images) {
    return TensorRT::preProcess(images);
}

float Keypoints::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                       cudaStream_t stream) const {
    return TensorRT::infer(InputDatas, bufferManager, stream);
}

void Keypoints::transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Keypoint> &keypoints,
                          bool is_padding) {
    assert(!is_padding);
    for(auto &keypoint : keypoints){
        keypoint.x *= iw / ow;
        keypoint.y *= ih / oh;
    }
}

bool Keypoints::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}

std::vector<common::Keypoint> Keypoints::predOneImage(const cv::Mat &image, float postThres) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);

    Clock<std::chrono::high_resolution_clock > clock_t;

    clock_t.tick();
    auto preImg = preProcess(std::vector<cv::Mat>{image});
    clock_t.tock();
    gLogInfo << "Pre Process time is " << clock_t.duration<double>() << "ms"<< std::endl;

    float elapsedTime = this->infer(std::vector<std::vector<float>>{preImg}, bufferManager, nullptr);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;

    clock_t.tick();
    std::vector<common::Keypoint> keypoints = postProcess(bufferManager, postThres);
    clock_t.tock();
    gLogInfo << "Post Process time is " << clock_t.duration<double>() << "ms"<< std::endl;

    this->transform(image.rows, image.cols, mKeypointParams.HeatMapH, mKeypointParams.HeatMapW, keypoints, mInputParams.IsPadding);
    return keypoints;
}

// ============== StreamProcess =====================>

StreamProcess::StreamProcess(DetectionTRT *trt, u_int32_t len) : mTrt(trt), mQ1(len), mQ2(len), mQ3(len), mQ4(len), flag_done(false) {

}

void StreamProcess::preFunc(const char *video_path) {
    try{
        cv::VideoCapture capture;
        cv::Mat frame;
        frame = capture.open(video_path);

        if(!capture.isOpened()){
            std::cout << "Video can not be opened! \n";
            flag_done = true;
            return;
        }else{
            std::cout << "Ori Video path is "<< video_path << std::endl;
        }
        while(!flag_done){
            if(!capture.read(frame)){
                flag_done = true;
            }else{
                auto pre_f = mTrt->preProcess(std::vector<cv::Mat>{frame});
                while( !mQ1.push(pre_f) && !flag_done){
                    std::this_thread::yield();
                }
                while(!mQ4.push(frame.clone()) && !flag_done){
                    std::this_thread::yield();
                }
            }
        }
        capture.release();
    }catch (std::exception &e){
        std::cout << "preFunc Exception  ===>\n" << e.what();
        throw e;
    }

}

void StreamProcess::inferFunc() {
    try{
        while(!flag_done){
            auto pre_f_p = mQ1.try_pop();
            if(pre_f_p== nullptr){
                std::this_thread::yield();
            }else{
                std::shared_ptr<common::BufferManager> bufferManager;
                bufferManager = std::make_shared<common::BufferManager>(mTrt->mCudaEngine, 1);
                std::vector<std::vector<float>> inputdata {*pre_f_p};
                mTrt->infer(inputdata, *bufferManager, nullptr);
                while(!mQ2.push(bufferManager) && !flag_done){
                    std::this_thread::yield();
                }
            }
        }
    }catch (std::exception &e){
        std::cout << "inferFunc Exception  ===> " << e.what();
        throw e;
    }
}

void StreamProcess::postFunc(const char *video_path, int sample_n) {
    try{
        int fps = 15;
        cv::VideoWriter writer;
        writer.open(video_path, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(1920, 1080));
        if(!writer.isOpened()){
            std::cout << "Writer can not be opened! \n";
            flag_done = true;
            return;
        }
        std::vector<common::Bbox> bboxes;
        while(!flag_done){
            auto post_f_p = mQ2.try_pop();
            auto img_p = mQ4.try_pop();
            if(post_f_p == nullptr && img_p == nullptr){
                std::this_thread::yield();
            }else{
                while(post_f_p == nullptr){
                    post_f_p = mQ2.try_pop();
                }
                while(img_p == nullptr){
                    img_p = mQ4.try_pop();
                }
                bboxes = mTrt->postProcess(**post_f_p, mTrt->mDetectParams.PostThreshold,
                                           mTrt->mDetectParams.NMSThreshold);
                auto ori_img = *img_p;
                mTrt->transform(ori_img.rows, ori_img.cols, mTrt->mInputParams.ImgH, mTrt->mInputParams.ImgW, bboxes, mTrt->mInputParams.IsPadding);
                auto render = renderBoundingBox(ori_img, bboxes);
                writer.write(render);
                std::cout << "frame = "<< count++ << std::endl;
            }
        }
        writer.release();
    }catch (std::exception &e){
        std::cout << "postFunc Exception  ===>\n" << e.what();
        throw e;
    }
}

void StreamProcess::schedule(int64_t s) {
    std::this_thread::sleep_for(std::chrono::seconds(s));
    flag_done = true;
}

void StreamProcess::print(){
    while(!flag_done){
        auto bboxes_p = mQ3.try_pop();
        if(bboxes_p == nullptr){
            std::this_thread::yield();
        }else{
            ++count;
            for(const auto & b : **bboxes_p){
                std::cout << "xmin = " << b.xmin << "  ymin = " << b.ymin << "  xmax = " << b.xmax << "  ymax = " << b.ymax << " score ="<< b.score <<"  cid=" << b.cid << std::endl;
            }
        }
    }
}

void StreamProcess::run(const char* video_path, const char* render_path) {
    count = 0;
    std::cout << "Run the Test"<< std::endl;
    std::vector<std::thread> threads(3);
    Clock<std::chrono::high_resolution_clock > tiktock;
    tiktock.tick();
    threads[0] = std::thread(&StreamProcess::preFunc, this, video_path);
    threads[1] = std::thread(&StreamProcess::inferFunc, this);
    threads[2] = std::thread(&StreamProcess::postFunc, this, render_path, 1);
//    threads[3] = std::thread(&StreamProcess::schedule, this, 1200);
//    threads[4] = std::thread(&StreamProcess::print, this);

    for(auto & t : threads){
        t.join();
    }
    tiktock.tock();
    auto total_time =  tiktock.duration<double>();
    std::cout << "Stream Process time is " << total_time << "ms   count is " <<count << "  Ave time is "<< total_time / count << std::endl;

}
