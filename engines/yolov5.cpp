#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "yolov5.h"

#define INPUT_NCHW true
#define INPUT_RGB true
#define OUTPUT_NLC true
#define MODEL_POST_ORIGIN true
#define IDENTIFIER "obj_det"

static constexpr float qMeanList[] = {0.0, 0.0, 0.0};
static constexpr float qNormList[] = {1.0, 1.0, 1.0};
static constexpr int32_t kGridScaleList[] = {8, 16, 32};
static constexpr int32_t kElmentAnchorNum = 3;
static constexpr int32_t kClassNum = 80;
static constexpr int32_t kOutputChannels = kClassNum + 5;
static int32_t OutputSpatialSize = 0;
#if (kElmentAnchorNum > 1)
static float constexpr AnchorWidthList[] = {0.5, 1.0, 2.0};
static float constexpr AnchorHeightList[] = {2.0, 1.0, 0.5};
#else
static float constexpr AnchorWidthList[] = {1.0};
static float constexpr AnchorHeightList[] = {1.0};
#endif

int32_t Yolov5::Initialize(const std::string& model) {
    NetworkMeta* p_meta = new NetworkMeta(NetworkMeta::kTensorTypeFloat32, INPUT_NCHW, INPUT_RGB, OUTPUT_NLC);
    p_meta->normalize.mean[0] = qMeanList[0];
    p_meta->normalize.mean[1] = qMeanList[1];
    p_meta->normalize.mean[2] = qMeanList[2];
    p_meta->normalize.norm[0] = qNormList[0];
    p_meta->normalize.norm[1] = qNormList[1];
    p_meta->normalize.norm[2] = qNormList[2];
    bmrun_helper_.reset(BmrunHelper::Create(model, p_meta));

    if (!bmrun_helper_) {
        return 0;
    }
    if (bmrun_helper_->Initialize() != 1) {
        std::cout << "bmrun_helper initialization failed" << std::endl;
        bmrun_helper_.reset();
        return 0;
    }

    // Check output tensor meta
    if (bmrun_helper_->GetOutputChannelNum() != kOutputChannels) {
        std::cout << "output channel size mismatched" << std::endl;
        return 0;
    }
    for (const auto& grid_scale : kGridScaleList) {
        OutputSpatialSize += (bmrun_helper_->GetInputWidth() / grid_scale) * (bmrun_helper_->GetInputHeight() / grid_scale);
    }
    OutputSpatialSize *= kElmentAnchorNum;
    if (bmrun_helper_->GetOutputLength() != OutputSpatialSize) {
        std::cout << "output spatial size mismatched" << std::endl;
        return 0;
    }
    // static constexpr int32_t KOutputSpatialSize = bmrun_helper_->GetOutputLength();
    ReadClsNames("./resource/inputs/label_coco_80.txt");

    return 1;
}

int32_t Yolov5::ReadClsNames(const std::string& filename) {
    std::ifstream ifs(filename);
    if (ifs.fail()) {
        std::cout << "failed to read " << filename << std::endl;
        return 0;
    }
    cls_names_.clear();
    std::string str;
    while (getline(ifs, str)) {
        cls_names_.push_back(str);
    }
    return 1;
}

void Yolov5::GetBoxPerLevel(const float* data_ptr, const int32_t grid_h, const int32_t grid_w, const int32_t delta_x, const int32_t delta_y , const float scale_h, const float scale_w, std::vector<Bbox2D>& bbox_list) {
    int32_t index = 0;
    // loop for every anchor size
    for (int32_t anchor = 0; anchor < kElmentAnchorNum; anchor++) {
        // loop for every output point
        for (int32_t grid_y = 0; grid_y < grid_h; grid_y++) {
            for (int32_t grid_x = 0; grid_x < grid_w; grid_x++) {
                float obj_confidence = data_ptr[index + 4];
                if (obj_confidence > obj_confidence_threshold_) {
                    int32_t cls_id = 0;
                    float cls_confidence = 0;
                    // loop for every channel to do argmax
                    for (int32_t cls_index = 0; cls_index < kClassNum; cls_index++) {
                        if (data_ptr[index + 5 + cls_index] > cls_confidence) {
                            cls_id = cls_index;
                            cls_confidence = data_ptr[index + 5 + cls_index];
                        }
                    }

                    if (cls_confidence > cls_confidence_threshold_) {
#if MODEL_POST_ORIGIN
                        int32_t cx = static_cast<int32_t>(((data_ptr[index + 0]) - delta_x) * scale_w);
                        int32_t cy = static_cast<int32_t>(((data_ptr[index + 1]) - delta_y ) * scale_h);
                        int32_t w = static_cast<int32_t>(data_ptr[index + 2] * scale_w);                 
                        int32_t h = static_cast<int32_t>(data_ptr[index + 3] * scale_h);
#else
                        int32_t cx = static_cast<int32_t>((data_ptr[index + 0] + grid_x) * scale_w);
                        int32_t cy = static_cast<int32_t>((data_ptr[index + 1] + grid_y) * scale_h);
                        int32_t w = static_cast<int32_t>(std::exp(data_ptr[index + 2]) * AnchorWidthList[anchor] * scale_w);
                        int32_t h = static_cast<int32_t>(std::exp(data_ptr[index + 3]) * AnchorHeightList[anchor] * scale_h);
#endif
                        int32_t x = cx - w / 2;
                        int32_t y = cy - h / 2;
                        bbox_list.push_back(Bbox2D(cls_id, cls_names_[cls_id], cls_confidence, x, y, w, h));
                    }
                }
                index += kOutputChannels;
            }
        }
    }
}

int32_t Yolov5::Process(cv::Mat& original_mat, Result& result) {
    // 1. prep-rocess
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    int32_t original_w = original_mat.cols;
    int32_t original_h = original_mat.rows;
    bmrun_helper_->PreProcess(original_mat,  kCropStyle::CropAll_Embedd);
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    // 2. inference
    bmrun_helper_->Inference();

    // 3.1 post-process, retrive output and scale bboxes
    const auto& t_post_process0 = std::chrono::steady_clock::now();
    const float* output = bmrun_helper_->GetInferenceOutput();
    cv::Rect dst_crop = bmrun_helper_->GetCropInfo().second;
    std::vector<Bbox2D> bbox_list;
    for (const auto& grid_scale : kGridScaleList) {
        int32_t grid_w = bmrun_helper_->GetInputWidth() / grid_scale;
        int32_t grid_h = bmrun_helper_->GetInputHeight() / grid_scale;
        float scale_w = static_cast<float>(original_w) / dst_crop.width;
        float scale_h = static_cast<float>(original_h) / dst_crop.height;
        GetBoxPerLevel(output, grid_h, grid_w, dst_crop.x, dst_crop.y, scale_h, scale_w, bbox_list);
        output += grid_h * grid_w * kElmentAnchorNum * kOutputChannels;
    }

    // 3.2 post-process, do nms
    std::vector<Bbox2D> bbox_nms_list;
    std::sort(bbox_list.begin(), bbox_list.end(), [](const Bbox2D& lhs, const Bbox2D& rhs) {
        if (lhs.cls_confidence > rhs.cls_confidence) return true;
        return false;
        });
    std::vector<bool> is_merged(bbox_list.size());
    for (size_t i = 0; i < is_merged.size(); i++) is_merged[i] = false;
    for (size_t i = 0; i < bbox_list.size(); i++) {
        if (is_merged[i]) continue;
        bbox_nms_list.push_back(bbox_list[i]);
        for (size_t j = i + 1; j < bbox_list.size(); j++) {
            if (bbox_list[i].cls_id != bbox_list[j].cls_id) continue;
            int32_t inter_left   = std::max(bbox_list[i].x, bbox_list[j].x);
            int32_t inter_right  = std::min(bbox_list[i].x + bbox_list[i].w, bbox_list[j].x + bbox_list[j].w);
            int32_t inter_top    = std::max(bbox_list[i].y, bbox_list[j].y);
            int32_t inter_bottom = std::min(bbox_list[i].y + bbox_list[i].h, bbox_list[j].y + bbox_list[j].h);
            if (inter_left > inter_right || inter_top > inter_bottom) continue;
            int32_t area_inter = (inter_right - inter_left) * (inter_bottom - inter_top);
            int32_t area_i = bbox_list[i].h * bbox_list[i].w;
            int32_t area_j = bbox_list[j].h * bbox_list[j].w;
            float iou = static_cast<float>(area_inter) / (area_i + area_j - area_inter);
            if (iou > nms_iou_threshold_) is_merged[j] = true;
        }
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();
    
    result.bbox_list = bbox_nms_list;
    result.process_time = 1.0 * (t_post_process1 - t_pre_process0).count() * 1e-6;
    std::cout << "pre-process: " << std::setw(8) << 1.0 * (t_pre_process1 - t_pre_process0).count() * 1e-6   << " ms" << std::endl;
    std::cout << "inference:   " << std::setw(8) << 1.0 * (t_post_process0 - t_pre_process1).count() * 1e-6  << " ms" << std::endl;
    std::cout << "post-process:" << std::setw(8) << 1.0 * (t_post_process1 - t_post_process0).count() * 1e-6 << " ms" << std::endl;;

    return 1;
}