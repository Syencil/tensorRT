// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/10

#include "tensorrt.h"

TensorRT::TensorRT(common::InputParams inputParams, common::TrtParams trtParams) : mInputParams(std::move(inputParams)), mTrtParams(std::move(trtParams)) {

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

float TensorRT::infer(const std::vector<std::vector<float>> &InputDatas, common::BufferManager &bufferManager) const {
    assert(InputDatas.size()==mInputParams.InputTensorNames.size());
    for(int i=0; i<InputDatas.size(); ++i){
        std::memcpy((void*)bufferManager.getHostBuffer(mInputParams.InputTensorNames[i]), (void*)InputDatas[i].data(), InputDatas[i].size() * sizeof(float));
    }
    bufferManager.copyInputToDevice();
    const auto t_start = std::chrono::high_resolution_clock::now();
    if (!mContext->execute(mInputParams.BatchSize, bufferManager.getDeviceBindings().data())) {
        gLogError << "Execute Failed!" << std::endl;
        return false;
    }
    const auto t_end = std::chrono::high_resolution_clock::now();
    const float elapsed_time = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    return elapsed_time;
}




