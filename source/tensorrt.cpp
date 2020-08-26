// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/10

#include "tensorrt.h"

#include <utility>


TensorRT::TensorRT(common::InputParams inputParams, common::TrtParams trtParams) : mInputParams(std::move(inputParams)), mTrtParams(std::move(trtParams)) {
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

std::vector<float> TensorRT::preProcess(const std::vector<cv::Mat> &images) const {
    std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, mInputParams.pFunction, mInputParams.HWC, mTrtParams.worker);
    return fileData;
}

// ============== DetectionTRT =====================>

DetectionTRT::DetectionTRT(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams detectParams) :
                TensorRT(std::move(inputParams), std::move(trtParams)), mDetectParams(std::move(detectParams)) {

}

std::vector<float> DetectionTRT::preProcess(const std::vector<cv::Mat> &images) const {
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



Segmentation::Segmentation(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams detectParams) :
        TensorRT(std::move(inputParams), std::move(trtParams)), mDetectParams(std::move(detectParams)) {

}

std::vector<float> Segmentation::preProcess(const std::vector<cv::Mat> &images) const {
    return TensorRT::preProcess(images);
}

float Segmentation::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager,
                          cudaStream_t stream) const {
    return TensorRT::infer(InputDatas, bufferManager, stream);
}

void Segmentation::transform(const int &ih, const int &iw, const int &oh, const int &ow, cv::Mat &mask, bool is_padding) {
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

bool Segmentation::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}

cv::Mat Segmentation::predOneImage(const cv::Mat &image, float postThres) {
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


