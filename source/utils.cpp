// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/10

#include "utils.h"

std::vector<float> imagePreprocess(const std::vector<cv::Mat> &images, const int &image_h, const int &image_w, bool is_padding, float(*pFun)(unsigned char), bool HWC){
    // image_path ===> BGR/HWC ===> RGB/HWC
    std::vector<float> fileData;
    for(const auto &image : images){
        cv::Mat rgb_image = image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        cv::Mat prodessed_image(image_h, image_w, CV_8UC3);

        if(is_padding){
            int ih = image.rows;
            int iw = image.cols;
            float scale = std::min(static_cast<float>(image_w) / static_cast<float>(iw), static_cast<float>(image_h) / static_cast<float>(ih));
            int nh = static_cast<int>(scale * static_cast<float>(ih));
            int nw = static_cast<int>(scale * static_cast<float>(iw));
            int dh = (image_h - nh) / 2;
            int dw = (image_w - nw) / 2;
            cv::Mat resized_image(nh, nw, CV_8UC3);
            cv::resize(rgb_image, resized_image, cv::Size(nw, nh));
            cv::copyMakeBorder(resized_image, prodessed_image, dh, image_h-nh-dh, dw, image_w-nw-dw, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
        }else{
            cv::Mat resized_image(image_h, image_w, CV_8UC3);
            cv::resize(rgb_image, prodessed_image, cv::Size(image_w, image_h));
        }
        std::vector<unsigned char> file_data = prodessed_image.reshape(1, 1);
        if(HWC){
            // HWC
            for (int i=0;i<file_data.size();++i){
                fileData.push_back((*pFun)(file_data[i]));
            }
        }else{
            // CHW
            int c, h, w, idx;
            for (int i=0;i<file_data.size();++i){
                w = i % image_w;
                h = i / image_w % image_h;
                c = i / image_w / image_h;
                idx = h * image_w * 3 + w * 3 + c;
                fileData.push_back((*pFun)(file_data[idx]));
            }
        }
    }
    return fileData;
}

std::vector<std::pair<std::string, std::string>> searchDirectory(const std::vector<std::string> &directory, const std::vector<std::string> &suffix){
    std::vector<std::pair<std::string, std::string>> file_path;
    for(const auto &sdir : directory){
        DIR *dir = opendir(sdir.c_str());
        dirent *p = nullptr;
        while((p = readdir(dir)) != nullptr){
            if ('.' != p->d_name[0]){
                for(const auto &suf : suffix){
                    if (strstr(p -> d_name, suf.c_str())){
                        file_path.emplace_back(sdir, p->d_name);
                    }
                }
            }
        }
    }
    return file_path;
}

cv::Mat renderBoundingBox(cv::Mat image, const std::vector<std::vector<float>> &bboxes){
    for (auto it: bboxes){
        float score = it[4];
        cv::rectangle(image, cv::Point(static_cast<int>(it[0]), static_cast<int>(it[1])), cv::Point(static_cast<int>(it[2]),
                                                                                                    static_cast<int>(it[3])), cv::Scalar(255, 204,0), 3);
        cv::putText(image, std::to_string(score), cv::Point(static_cast<int>(it[0]), static_cast<int>(it[1])), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
    }
    return image;
}

cv::Mat renderKeypoint(cv::Mat image, const std::vector<std::vector<float>> &keypoints){
    int image_h = image.rows;
    int image_w = image.cols;
    int point_x, point_y;
    for (int i=0; i<keypoints.size(); ++i){
        point_x = keypoints[i][0];
        point_y = keypoints[i][1];
        cv::circle(image, cv::Point(point_x, point_y), 5, cv::Scalar(255, 204,0), 3);
    }
    return image;
}