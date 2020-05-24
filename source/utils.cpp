// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/10

#include "utils.h"

// ==============Pre Process=============>
std::vector<float> imagePreprocess(const std::vector<cv::Mat> &images, const int &image_h, const int &image_w, bool is_padding, float(*pFun)(unsigned char&), bool HWC){
    // image_path ===> cv::Mat ===> resize(padding) ===> CHW/HWC (BRG=>RGB)
    // 测试发现RGB转BGR的cv::cvtColor 和 HWC 转 CHW非常耗时，故将其合并为一次操作
    const int image_length = image_h*image_w*3;
    std::vector<float> fileData(images.size()*image_length);
    for(int img_count=0; img_count<images.size(); ++img_count){
        cv::Mat image = images[img_count].clone();
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
            cv::resize(image, resized_image, cv::Size(nw, nh));
            cv::copyMakeBorder(resized_image, prodessed_image, dh, image_h-nh-dh, dw, image_w-nw-dw, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
        }else{
            cv::Mat resized_image(image_h, image_w, CV_8UC3);
            cv::resize(image, prodessed_image, cv::Size(image_w, image_h));
        }
        std::vector<unsigned char> file_data = prodessed_image.reshape(1, 1);
        if(HWC){
            // HWC and BRG=>RGB
            for (int h=0; h<image_h; ++h){
                for (int w=0; w<image_w; ++w){
                    fileData[img_count * image_length + h * image_w * 3 + w * 3 + 0] =
                            (*pFun)(file_data[h * image_w * 3 + w * 3 + 2]);
                    fileData[img_count * image_length + h * image_w * 3 + w * 3 + 1] =
                            (*pFun)(file_data[h * image_w * 3 + w * 3 + 1]);
                    fileData[img_count * image_length + h * image_w * 3 + w * 3 + 2] =
                            (*pFun)(file_data[h * image_w * 3 + w * 3 + 0]);
                }
            }
        }else{
            // CHW and BRG=>RGB
            for (int h=0; h<image_h; ++h){
                for (int w=0; w<image_w; ++w) {
                    fileData[img_count * image_length + 0 * image_h * image_w + h * image_w + w] =
                            (*pFun)(file_data[h * image_w * 3 + w * 3 + 2]);
                    fileData[img_count * image_length + 1 * image_h * image_w + h * image_w + w] =
                            (*pFun)(file_data[h * image_w * 3 + w * 3 + 1]);
                    fileData[img_count * image_length + 2 * image_h * image_w + h * image_w + w] =
                            (*pFun)(file_data[h * image_w * 3 + w * 3 + 0]);
                }
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


// =============Post Process=============>


// ===============Rendering =============>

cv::Mat renderBoundingBox(cv::Mat image, const std::vector<common::Bbox> &bboxes){
    for (auto it: bboxes){
        float score = it.score;
        cv::rectangle(image, cv::Point(it.xmin, it.ymin), cv::Point(it.xmax, it.ymax), cv::Scalar(255, 204,0), 3);
        cv::putText(image, std::to_string(score), cv::Point(it.xmin, it.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
    }
    return image;
}

cv::Mat renderKeypoint(cv::Mat image, const std::vector<common::Keypoint> &keypoints){
    int image_h = image.rows;
    int image_w = image.cols;
    int point_x, point_y;
    for (const auto &keypoint : keypoints){
        point_x = keypoint.x;
        point_y = keypoint.y;
        cv::circle(image, cv::Point(point_x, point_y), 5, cv::Scalar(255, 204,0), 3);
    }
    return image;
}


