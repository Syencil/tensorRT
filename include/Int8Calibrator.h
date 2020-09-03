// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/13

#ifndef TENSORRT_INT8CALIBRATOR_H
#define TENSORRT_INT8CALIBRATOR_H

#include "common.h"
#include "utils.h"
#include "logger.h"
#include <fstream>

#include <driver_types.h>
#include <iterator>
#include <NvInfer.h>
#include <cuda.h>


class EntropyCalibratorV2: public nvinfer1::IInt8Calibrator{
private:
    std::vector<char> mCache;

    std::vector<std::pair<std::string, std::string>> mFileList;
    common::TrtParams mTrtParams;
    common::InputParams mInputParams;
    std::vector<nvinfer1::Dims3> mInputsDim;

    int mImageSize;
    int mInputCount;

    int mStartPos;
    int mCurPos;
    int mEndPos;

    float* mDevice_ptr;
    float* mHost_ptr;
    // 预处理函数
    float (*pFunction)(const unsigned char&);
private:
    bool update();
    bool readIntoBuffer();
public:
    EntropyCalibratorV2(common::InputParams inputParams, common::TrtParams trtParams);
    ~EntropyCalibratorV2() override ;
    int getBatchSize() const override;
    bool getBatch (void *bindings[], const char *names[], int nbBindings) override;
    const void *readCalibrationCache (std::size_t &length) override;
    void writeCalibrationCache (const void *ptr, std::size_t length) override;
    nvinfer1::CalibrationAlgoType getAlgorithm() TRTNOEXCEPT override { return nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION_2; }

    //
    void reset();

};

#endif //TENSORRT_INT8CALIBRATOR_H
