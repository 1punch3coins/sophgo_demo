#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../bmrun_helper.h"
#include "../det_structs.h"

class Yolov8Seg {
public:
    struct Result {
        // std::vector<std::pair<Bbox2D, std::vector<Point2D>>> bbox_mask_vec;
        std::vector<Bbox2D> bbox_list;
        std::vector<cv::Mat> mask_list;
        float process_time;
        // Result() {bbox_mask_vec.reserve(10);}
        Result() {bbox_list.reserve(10);}
    };

public:
    Yolov8Seg(float obj_threshold = 0.4, float cls_threshold = 0.2, float iou_threshold = 0.5):
        obj_confidence_threshold_(obj_threshold), cls_confidence_threshold_(cls_threshold), nms_iou_threshold_(iou_threshold)
    {}

public:
    int32_t Initialize(const std::string& model);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& original_mat, Result& result);
    int32_t Process(cv::Mat& original_mat, Result& result);
    int32_t ReadClsNames(const std::string& filename);
    
private:
    void GetBoxPerLevel(const float* data_ptr, int32_t& index, const int32_t grid_h, const int32_t grid_w, const int32_t delta_x, const int32_t delta_y, const float scale_h, const float scale_w, std::vector<Bbox2D>& bbox_list, std::vector<std::pair<float, const float*>>& seg_channel_weights_list);
    
private:
    std::unique_ptr<BmrunHelper> bm_helper_;
    float obj_confidence_threshold_;
    float cls_confidence_threshold_;
    float nms_iou_threshold_;
    std::vector<std::string> cls_names_;
};