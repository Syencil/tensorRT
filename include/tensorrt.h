// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/10

#ifndef TENSORRT_7_TENSORRT_H
#define TENSORRT_7_TENSORRT_H

#include "common.h"
#include "buffers.h"
#include "logger.h"
#include "Int8Calibrator.h"
#include "utils.h"

#include <fstream>
#include <opencv2/core.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <map>
#include <mutex>
#include <thread>

//! TensorRT Base Class
class TensorRT{
protected:
    template <typename T>
    using UniquePtr = std::unique_ptr<T, common::InferDeleter>;
    std::shared_ptr<nvinfer1::ICudaEngine> mCudaEngine;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext;
    common::InputParams mInputParams;
    common::TrtParams mTrtParams;
    cudaEvent_t start_t, stop_t;

protected:
    //! Initialize mInputParams, mTrtParms
    //! \param inputParams Input images params
    //! \param trtParams TensorRT definition configs
    TensorRT(common::InputParams inputParams, common::TrtParams trtParams);

    ~TensorRT();

    //! Build the network and construct CudaEngine and Context. It should be called before serializeEngine()
    //! \param onnxPath Onnx file path.
    //! \return Return true if no errors happened.
    virtual bool constructNetwork(const std::string &onnxPath);

    //! Serialize the CudaEngine. It should be called after constructNetwork();
    //! \param save_path Saved file path
    //! \return Return true if no errors happened.
    virtual bool serializeEngine(const std::string &save_path);

    //! Deserialize the CudaEngine.
    //! \param load_path Saved file path
    //! \return Return true if no errors happened.
    virtual bool deseriazeEngine(const std::string &load_path);

    //! PreProcess
    virtual std::vector<float> preProcess(const std::vector<cv::Mat> &images) const;

    //! (ASynchronously) Execute the inference on a batch.
    //! \param InputDatas Float arrays which must corresponded with InputTensorNames.
    //! \param bufferManager An instance of BufferManager. It holds the raw inference results.
    //! \param cudaStream_t Execute in stream
    //! \return Return the inference time in ms. If failed, return 0.
    virtual float infer(const std::vector<std::vector<float>>&InputDatas, common::BufferManager &bufferManager, cudaStream_t stream) const;

protected:
    //! Init Inference Session
    //! \param initOrder 0==========> init from SerializedPath. If failed, init from onnxPath.
    //!                             1 ==========> init from onnxPath and save the session into SerializedPath if it doesnt exist.
    //!                             2 ==========> init from onnxPath and force to save the session into SerializedPath.
    //! \return true if no errors happened.
    virtual bool initSession(int initOrder);

};


//! TensorRT Detection Base Class
class DetectionTRT : protected TensorRT{
protected:
    common::DetectParams mDetectParams;
    std::mutex mMutex;

protected:
    DetectionTRT(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams detectParams);

    std::vector<float> preProcess(const std::vector<cv::Mat> &images) const override ;

    float infer(const std::vector<std::vector<float>>&InputDatas, common::BufferManager &bufferManager, cudaStream_t stream) const override ;

    virtual std::vector<common::Bbox> postProcess(common::BufferManager &bufferManager, float postThres, float nmsThres) = 0;

    virtual void transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes, bool is_padding) ;

    bool initSession(int initOrder) override ;

    virtual std::vector<common::Bbox> predOneImage(const cv::Mat &image, float postThres, float nmsThres) ;

};


//! TensorRT Segment Base Class
class Segmentation : protected TensorRT{
protected:
    common::DetectParams mDetectParams;
    std::mutex mMutex;

protected:
    Segmentation(common::InputParams inputParams, common::TrtParams trtParams, common::DetectParams detectParams);

    std::vector<float> preProcess(const std::vector<cv::Mat> &images) const override ;

    float infer(const std::vector<std::vector<float>>&InputDatas, common::BufferManager &bufferManager, cudaStream_t stream) const override ;

    virtual cv::Mat postProcess(common::BufferManager &bufferManager, float postThres) = 0;

    virtual void transform(const int &ih, const int &iw, const int &oh, const int &ow, cv::Mat &mask, bool is_padding);

    bool initSession(int initOrder) override;

    virtual cv::Mat predOneImage(const cv::Mat &image, float postThres);

};

#endif //TENSORRT_7_TENSORRT_H
