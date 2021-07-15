// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/13

#include "Int8Calibrator.h"

int EntropyCalibratorV2::getBatchSize() const {
    return mInputParams.BatchSize;
}

const void *EntropyCalibratorV2::readCalibrationCache(std::size_t &length){
    mCache.clear();
    std::ifstream input(mTrtParams.CalibrationTablePath, std::ios::binary);
    input >> std::noskipws;
    if (input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCache));
    }
    length = mCache.size();
    return length ? mCache.data() : nullptr;
}

void EntropyCalibratorV2::writeCalibrationCache(const void *ptr, std::size_t length) {
    std::ofstream output(mTrtParams.CalibrationTablePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(ptr), length);
}

void EntropyCalibratorV2::reset() {
    mStartPos = 0;
    mCurPos = 0;
}

bool EntropyCalibratorV2::update() {
    mStartPos = mCurPos;
    mCurPos += mInputParams.BatchSize;
    return (mCurPos <= mEndPos && mCurPos/mInputParams.BatchSize <= mTrtParams.MaxBatch);
}

bool EntropyCalibratorV2::readIntoBuffer() {
    if(mCurPos>mEndPos){
        return false;
    }
    std::vector<cv::Mat> images;
    for(int i=0; i<mInputParams.BatchSize;++i){
        std::string image_path = mFileList[i+mStartPos].first.append("/").append(mFileList[i+mStartPos].second);
        cv::Mat img = cv::imread(image_path);
        if (!img.data){
            gLogError << "Invalid image path " << image_path << std::endl;
        }
        images.emplace_back(cv::imread(image_path, cv::IMREAD_COLOR));
    }
    std::vector<float>raw_file = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, pFunction);
    if (raw_file.size() != mInputCount){
        gLogError << "FileData size is "<<raw_file.size()<<" but needed "<<  mInputCount <<std::endl;
        return false;
    }
    std::memcpy(mHost_ptr, raw_file.data(), mInputCount);
    return true;
}



