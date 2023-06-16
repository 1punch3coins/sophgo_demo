#ifndef YOLOV5_
#define YOLOV5_
#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../bmrun_helper.h"
#include "../det_structs.h"

class Yolov5 {
public:
    struct Result {
        std::vector<Bbox2D> bbox_list;
        float process_time;
        Result() {bbox_list.reserve(10);}
    };

public:
    Yolov5(float obj_threshold = 0.4, float cls_threshold = 0.2, float iou_threshold = 0.5):
        obj_confidence_threshold_(obj_threshold), cls_confidence_threshold_(cls_threshold), nms_iou_threshold_(iou_threshold)
    {}

public:
    int32_t Initialize(const std::string& model);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& original_mat, Result& result);
    int32_t Process(cv::Mat& original_mat, Result& result);
    
private:
    void GetBoxPerLevel(float* data_ptr, const int32_t grid_h, const int32_t grid_w, const int32_t delta_x, const int32_t delta_y , const float scale_h, const float scale_w, std::vector<Bbox2D>& bbox_list);
    int32_t ReadClaNames(const std::string& filename);

private:
    std::unique_ptr<BmrunHelper> bmrun_helper_;
    float obj_confidence_threshold_;
    float cls_confidence_threshold_;
    float nms_iou_threshold_;
    std::vector<std::string> cls_names_;
};

#endif