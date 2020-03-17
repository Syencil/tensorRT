// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/13
#include "Int8Calibrator.h"



EntropyCalibratorV2::EntropyCalibratorV2(common::InputParams inputParams, common::TrtParams trtParams): mInputParams(std::move(inputParams)), mTrtParams(std::move(trtParams)){
    pFunction = mInputParams.pFunction;
    mImageSize = mInputParams.ImgC * mInputParams.ImgH * mInputParams.ImgW;
    mInputCount = mInputParams.BatchSize * mImageSize;
    CHECK(cudaMalloc((void**)&mDevice_ptr, sizeof(float) * mInputCount));
    mHost_ptr = new float[mInputCount];

    // search file
    std::vector<std::pair<std::string, std::string>> file_path;
    mFileList = searchDirectory(std::vector<std::string>{mTrtParams.CalibrationImageDir}, std::vector<std::string>{".jpg", ".png"});

    mStartPos = 0;
    mCurPos = mStartPos + mInputParams.BatchSize;
    mEndPos = mFileList.size();
}


bool EntropyCalibratorV2::getBatch(void **bindings, const char **names, int nbBindings) {
    if(readIntoBuffer()&&update()){
        CHECK(cudaMemcpy(mDevice_ptr, mHost_ptr, sizeof(float) * mInputCount, cudaMemcpyHostToDevice));
        bindings[0] = mDevice_ptr;
        return true;
    }
    return false;
}

EntropyCalibratorV2::~EntropyCalibratorV2() {
    CHECK(cudaFree(mDevice_ptr));
    delete []mHost_ptr;
}

