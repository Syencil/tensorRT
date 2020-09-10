// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/10

#include "utils.h"

void idxTransformParall(std::vector<unsigned char> *in_file, std::vector<float> *out_file,
                        unsigned long start_h, unsigned long length, unsigned long image_h, unsigned long image_w,
                        unsigned long start, float (*pFun)(const unsigned char&), bool HWC){
    if(HWC){
        // HWC and BRG=>RGB
        for(unsigned long h=start_h; h<start_h+length; ++h){
            for(unsigned long w=0; w<image_w; ++w){
                (*out_file)[start + h * image_w * 3 + w * 3 + 0] =
                        (*pFun)((*in_file)[h * image_w * 3 + w * 3 + 2]);
                (*out_file)[start + h * image_w * 3 + w * 3 + 1] =
                        (*pFun)((*in_file)[h * image_w * 3 + w * 3 + 1]);
                (*out_file)[start + h * image_w * 3 + w * 3 + 2] =
                        (*pFun)((*in_file)[h * image_w * 3 + w * 3 + 0]);
            }
        }
    }else{
        // CHW and BRG=>RGB
        for(unsigned long h=start_h; h<start_h+length; ++h){
            for(unsigned long w=0; w<image_w; ++w){
                (*out_file)[start + 0 * image_h * image_w + h * image_w + w] =
                        (*pFun)((*in_file)[h * image_w * 3 + w * 3 + 2]);
                (*out_file)[start + 1 * image_h * image_w + h * image_w + w] =
                        (*pFun)((*in_file)[h * image_w * 3 + w * 3 + 1]);
                (*out_file)[start + 2 * image_h * image_w + h * image_w + w] =
                        (*pFun)((*in_file)[h * image_w * 3 + w * 3 + 0]);
            }
        }
    }
}

// ==============Pre Process=============>
std::vector<float> imagePreprocess(const std::vector<cv::Mat> &images, const int &image_h, const int &image_w, bool is_padding, float(*pFun)(const unsigned char&), bool HWC, int worker){
    // image_path ===> cv::Mat ===> resize(padding) ===> CHW/HWC (BRG=>RGB)
    // 测试发现RGB转BGR的cv::cvtColor 和 HWC 转 CHW非常耗时，故将其合并为一次操作
    const unsigned long image_length = image_h*image_w*3;
    std::vector<float> fileData(images.size()*image_length);
    for(unsigned long img_count=0; img_count<images.size(); ++img_count){
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
        // 并发
        unsigned long min_threads;
        if (worker<0){
            const unsigned long min_length = 64;
            min_threads = (image_h - 1) / min_length + 1;
        }else if(worker==0){
            min_threads = 1;
        }else{
            min_threads = worker;
        }
        const unsigned long cpu_max_threads = std::thread::hardware_concurrency();
        const unsigned long num_threads = std::min(cpu_max_threads !=0 ? cpu_max_threads : 1, min_threads);
        const unsigned long block_size = image_h / num_threads;
        std::vector<std::thread> threads(num_threads-1);
        unsigned long block_start = 0;
        for (auto &t : threads){
            t = std::thread(idxTransformParall, &file_data, &fileData, block_start, block_size, image_h, image_w, img_count * image_length, pFun, HWC);
            block_start += block_size;
        }
        idxTransformParall(&file_data, &fileData, block_start, image_h - block_start, image_h, image_w, img_count * image_length, pFun, HWC);
        for(auto &t : threads){
            t.join();
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
void nms_cpu(std::vector<common::Bbox> &bboxes, float threshold) {
    if (bboxes.empty()){
        return ;
    }
    // 1.之前需要按照score排序
    std::sort(bboxes.begin(), bboxes.end(), [&](common::Bbox b1, common::Bbox b2){return b1.score>b2.score;});
    // 2.先求出所有bbox自己的大小
    std::vector<float> area(bboxes.size());
    for (int i=0; i<bboxes.size(); ++i){
        area[i] = (bboxes[i].xmax - bboxes[i].xmin + 1) * (bboxes[i].ymax - bboxes[i].ymin + 1);
    }
    // 3.循环
    for (int i=0; i<bboxes.size(); ++i){
        for (int j=i+1; j<bboxes.size(); ){
            float left = std::max(bboxes[i].xmin, bboxes[j].xmin);
            float right = std::min(bboxes[i].xmax, bboxes[j].xmax);
            float top = std::max(bboxes[i].ymin, bboxes[j].ymin);
            float bottom = std::min(bboxes[i].ymax, bboxes[j].ymax);
            float width = std::max(right - left + 1, 0.f);
            float height = std::max(bottom - top + 1, 0.f);
            float u_area = height * width;
            float iou = (u_area) / (area[i] + area[j] - u_area);
            if (iou>=threshold){
                bboxes.erase(bboxes.begin()+j);
                area.erase(area.begin()+j);
            }else{
                ++j;
            }
        }
    }
}

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

cv::Mat renderPoly(cv::Mat image, const std::vector<std::vector<cv::Point>> &polygons){
    for(const auto & poly : polygons){
        for(int i=0; i<poly.size()-1; ++i){
            cv::line(image, poly[i], poly[i+1], cv::Scalar(0,204,255));
        }
    }
    return image;
}

cv::Mat renderSegment(cv::Mat image, const cv::Mat &mask){
    double min_val, max_val;
    int min_idx[2] = {};
    int max_idx[2] = {};
    cv::minMaxIdx(mask, &min_val, &max_val, min_idx, max_idx);
    cv::Mat color;
    cv::merge(std::vector<cv::Mat>{mask * (rand()%255), mask * (rand()%(int)(255/max_val)), mask * (rand()%(int)(255/max_val))}, color);
    cv::addWeighted(image, 1, color, 1, 0, image);
    return image;
}

cv::Mat renderRBox(cv::Mat image, const std::vector<cv::RotatedRect> &RBox) {
    for(const auto & rb : RBox){
        std::vector<cv::Point> points;
        cv::Mat bbox;
        cv::boxPoints(rb, bbox);
        cv::Scalar color(0, 204, 255);
        for (int i = 1; i < bbox.rows; ++i) {
            cv::line(image, cv::Point(int(bbox.at<float>(i-1, 0) ), int(bbox.at<float>(i-1, 1))),  cv::Point(int(bbox.at<float>(i, 0) ), int(bbox.at<float>(i, 1))), color, 3);
        }
        cv::line(image, cv::Point(int(bbox.at<float>(bbox.rows-1, 0) ), int(bbox.at<float>(bbox.rows-1, 1))),  cv::Point(int(bbox.at<float>(0, 0) ), int(bbox.at<float>(0, 1))), color, 3);
    }
    return image;
}