#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <array>
#include <algorithm>

#include "yolov8_seg.h"

#define INPUT_NCHW true
#define INPUT_RGB true
#define MODEL_POST_ORIGIN false
#define IDENTIFIER "obj_det"

static constexpr int32_t kInputTensorNum = 1;
static constexpr int32_t kOutputTensorNum = 2;
static constexpr std::array<const char*, kInputTensorNum> sInputNameList = {"images"};
static constexpr std::array<const char*, kOutputTensorNum> sOutputNameList = {"boxes_Concat_f32", "segments_Transpose_f32"};

static constexpr float qMeanList[] = {0.0, 0.0, 0.0};
static constexpr float qNormList[] = {1.0, 1.0, 1.0};
static constexpr int32_t kGridScaleList[] = {8, 16, 32};
static constexpr int32_t kElmentAnchorNum = 1;
static constexpr int32_t kBoxClassNum = 80;
static constexpr int32_t kSegChannelNum = 32;
static constexpr int32_t kSegGridScale = 4;
#if MODEL_POST_ORIGIN
static constexpr int32_t kBoxOutputChannelNum = kBoxClassNum + 4;
#define OUTPUT_NLC false
#else
static constexpr int32_t kBoxOutputChannelNum = 2 + 4 + kSegChannelNum;
#define OUTPUT_NLC true
#endif
static int32_t OutputSpatialSize = 0;
static int32_t SegOutputHeight = 0;
static int32_t SegOutputWidth = 0;

int32_t Yolov8Seg::Initialize(const std::string& model) {
    NetworkMeta* p_meta = new NetworkMeta(NetworkMeta::kTensorTypeFloat32, INPUT_NCHW, INPUT_RGB, OUTPUT_NLC, kInputTensorNum, kOutputTensorNum);
    p_meta->normalize.mean[0] = qMeanList[0];
    p_meta->normalize.mean[1] = qMeanList[1];
    p_meta->normalize.mean[2] = qMeanList[2];
    p_meta->normalize.norm[0] = qNormList[0];
    p_meta->normalize.norm[1] = qNormList[1];
    p_meta->normalize.norm[2] = qNormList[2];
    for (const auto input_name : sInputNameList) {
        p_meta->AddInputTensorMeta(input_name);
    }
    for (int32_t i = 0; i < kOutputTensorNum; i++) {
        p_meta->AddOutputTensorMeta(sOutputNameList[i]);
    }
    bm_helper_.reset(BmrunHelper::Create(model, p_meta));

    if (!bm_helper_) {
        return 0;
    }
    if (bm_helper_->Initialize() != 1) {
        std::cout << "bmrun_helper initialization failed" << std::endl;
        bm_helper_.reset();
        return 0;
    }

    // Check output tensor "boxes" meta
    if (bm_helper_->GetOutputChannelNum(sOutputNameList[0]) != kBoxOutputChannelNum) {
        std::cout << "output channel size mismatched" << std::endl;
        return 0;
    }
    for (const auto& grid_scale : kGridScaleList) {
        OutputSpatialSize += (bm_helper_->GetInputWidth() / grid_scale) * (bm_helper_->GetInputHeight() / grid_scale);
    }
    OutputSpatialSize *= kElmentAnchorNum;
    if (bm_helper_->GetOutputLength(sOutputNameList[0]) != OutputSpatialSize) {
        std::cout << "output spatial size mismatched" << std::endl;
        return 0;
    }
    // Check output tensor "segments" meta
    if (bm_helper_->GetOutputChannelNum(sOutputNameList[1]) != kSegChannelNum) {
        std::cout << "output channel size mismatched" << std::endl;
        return 0;
    }
    SegOutputHeight = bm_helper_->GetInputHeight() / kSegGridScale;
    SegOutputWidth = bm_helper_->GetInputWidth() / kSegGridScale;
    if (bm_helper_->GetOutputLength(sOutputNameList[1]) != SegOutputHeight * SegOutputWidth) {
        std::cout << "output spatial size mismatched" << std::endl;
        return 0;
    }

    ReadClsNames("./resource/inputs/label_coco_80.txt");

    return 1;
}

int32_t Yolov8Seg::ReadClsNames(const std::string& filename) {
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

void Yolov8Seg::GetBoxPerLevel(const float* data_ptr, int32_t& index, const int32_t grid_h, const int32_t grid_w, const int32_t delta_x, const int32_t delta_y, const float scale_h, const float scale_w, std::vector<Bbox2D>& bbox_list, std::vector<std::pair<float, const float*>>& seg_channel_weights_list) {
    // loop for every output point
    for (int32_t grid_y = 0; grid_y < grid_h; grid_y++) {
        for (int32_t grid_x = 0; grid_x < grid_w; grid_x++) {
#if MODEL_POST_ORIGIN
            int32_t cls_id = 0;
            float cls_confidence = 0;
            // loop for every channel for argmax
            for (int32_t cls_index = 0; cls_index < kBoxClassNum; cls_index++) {
                float confidence = data_ptr[index + OutputSpatialSize*(cls_index+4)];
                if (confidence > cls_confidence) {
                    cls_confidence = confidence;
                    cls_id = cls_index;
                }
            }
            if (cls_confidence >= cls_confidence_threshold_) {
                int32_t cx = static_cast<int32_t>((data_ptr[index + OutputSpatialSize*0] - delta_x) * scale_w);
                int32_t cy = static_cast<int32_t>((data_ptr[index + OutputSpatialSize*1] - delta_y) * scale_h);
                int32_t w = static_cast<int32_t>(data_ptr[index + OutputSpatialSize*2] * scale_w);
                int32_t h = static_cast<int32_t>(data_ptr[index + OutputSpatialSize*3] * scale_h);
                int32_t x = cx - w / 2;
                int32_t y = cy - h / 2;
                bbox_list.push_back(Bbox2D(cls_id, cls_confidence, x, y, w, h));
            }
            index += 1;
#else
            float cls_confidence = data_ptr[index + 4];
            if (cls_confidence >= cls_confidence_threshold_) {
                int32_t cls_id = static_cast<int32_t>(data_ptr[index + 5]);
                int32_t x0 = static_cast<int32_t>((data_ptr[index + 0] - delta_x) * scale_w);
                int32_t y0 = static_cast<int32_t>((data_ptr[index + 1] - delta_y) * scale_h);
                int32_t x1 = static_cast<int32_t>((data_ptr[index + 2] - delta_x) * scale_w);
                int32_t y1 = static_cast<int32_t>((data_ptr[index + 3] - delta_y) * scale_h);
                // int32_t w = static_cast<int32_t>(data_ptr[index + 2] * scale_w);
                // int32_t h = static_cast<int32_t>(data_ptr[index + 3] * scale_h);
                // int32_t x = cx - w / 2;
                // int32_t y = cy - h / 2;
                std::string cls_name = cls_names_[cls_id];
                bbox_list.push_back(Bbox2D(cls_id, cls_name, cls_confidence, x0, y0, x1-x0, y1-y0));
                seg_channel_weights_list.push_back(std::pair<float, const float*>(cls_confidence, data_ptr + index + 6));
            }
            index += kBoxOutputChannelNum;
#endif
        }
    }
}

int32_t Yolov8Seg::Process(cv::Mat& original_mat, Result& result) {
    // 1. prep-rocess
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    int32_t original_w = original_mat.cols;
    int32_t original_h = original_mat.rows;
    std::string input_name = sInputNameList[0];
    bm_helper_->PreProcess(original_mat, kCropStyle::CropAll_Embedd);
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    // 2. inference
    bm_helper_->Inference();

    // 3.1 post-process, retrive bbox output and scale bboxes
    const auto& t_post_process0 = std::chrono::steady_clock::now();
    const float* output_box = bm_helper_->GetInferenceOutput(sOutputNameList[0]);
    cv::Rect dst_crop = bm_helper_->GetCropInfo().second;
    std::vector<Bbox2D> bbox_list;
    std::vector<std::pair<float, const float*>> seg_channel_weights_list;
    int32_t index = 0;
    for (const auto& grid_scale : kGridScaleList) {
        int32_t grid_w = bm_helper_->GetInputWidth(input_name) / grid_scale;
        int32_t grid_h = bm_helper_->GetInputHeight(input_name) / grid_scale;
        float scale_w = static_cast<float>(original_w) / dst_crop.width;
        float scale_h = static_cast<float>(original_h) / dst_crop.height;
        GetBoxPerLevel(output_box, index, grid_h, grid_w, dst_crop.x, dst_crop.y, scale_h, scale_w, bbox_list, seg_channel_weights_list);
    }
    for (auto& box : bbox_list) {
        box.x = box.x > 0 ? box.x : 0;
        box.y = box.y > 0 ? box.y : 0;
        box.w = box.w + box.x < original_w ? box.w : original_w - box.x;
        box.h = box.h + box.y < original_h ? box.h : original_h - box.y;
    }

    // 3.2 post-process, do nms
    std::vector<Bbox2D> bbox_nms_list;
    std::vector<const float*> seg_channel_weights_nms_lit;
    std::sort(bbox_list.begin(), bbox_list.end(), [](const Bbox2D& lhs, const Bbox2D& rhs) {
        if (lhs.cls_confidence > rhs.cls_confidence) return true;
        return false;
        });
    std::sort(seg_channel_weights_list.begin(), seg_channel_weights_list.end(), [](const std::pair<float, const float*>& lhs, const std::pair<float, const float*>& rhs) {
        if (lhs.first > rhs.first) return true;
        return false;
        }) ;
    std::vector<bool> is_merged(bbox_list.size());
    for (size_t i = 0; i < is_merged.size(); i++) is_merged[i] = false;
    for (size_t i = 0; i < bbox_list.size(); i++) {
        if (is_merged[i]) continue;
        bbox_nms_list.push_back(bbox_list[i]);
        seg_channel_weights_nms_lit.push_back(seg_channel_weights_list[i].second);
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

    // 3.3 post-process, retrive seg output & sclae bboxes to seg output & calculate weighted mask for each bbox
    const auto& t_post_process1 = std::chrono::steady_clock::now();
    const float* output_seg = bm_helper_->GetInferenceOutput(sOutputNameList[1]);
    int32_t seg_grid_offset_x = dst_crop.x / kSegGridScale;
    int32_t seg_grid_offset_y = dst_crop.y / kSegGridScale;
    int32_t seg_grid_width = bm_helper_->GetInputWidth(input_name) / kSegGridScale;
    int32_t seg_grid_height = bm_helper_->GetInputHeight(input_name) / kSegGridScale;
    std::vector<cv::Mat> masks;
    masks.reserve(bbox_nms_list.size());
#if 1   // my implemention, first crop in the seg output, then resize to its box'size, faster
    float ori2seg_scale_w = (seg_grid_width - 2 * seg_grid_offset_x) / static_cast<float>(original_w);
    float ori2seg_scale_h = (seg_grid_height - 2 * seg_grid_offset_y) / static_cast<float>(original_h);
    for (const auto& bbox : bbox_nms_list) {
        int32_t seg_crop_x = static_cast<int32_t>(bbox.x * ori2seg_scale_w + seg_grid_offset_x);
        int32_t seg_crop_y = static_cast<int32_t>(bbox.y * ori2seg_scale_h + seg_grid_offset_y);
        int32_t seg_crop_w = static_cast<int32_t>(bbox.w * ori2seg_scale_w);
        int32_t seg_crop_h = static_cast<int32_t>(bbox.h * ori2seg_scale_h);
        seg_crop_w = std::max(seg_crop_w, 1);
        seg_crop_h = std::max(seg_crop_h, 1);
        cv::Mat mask_mat(seg_crop_h, seg_crop_w, CV_32FC1, cv::Scalar(0));
        float* ptr = (float*)mask_mat.data;
        
        int32_t mat_index = 0;
        for (int32_t i = 0; i < seg_crop_h; i++) {
            int32_t seg_spatial_index = (i + seg_crop_y) * seg_grid_width + seg_crop_x;
            for (int32_t j = 0; j < seg_crop_w; j++) {
                // seg_spatial_index += j; // fatal error, just a reminder
                int32_t seg_channel_index = seg_spatial_index * kSegChannelNum;
                for (int32_t k = 0; k < kSegChannelNum; k++) {
                    ptr[mat_index] += seg_channel_weights_nms_lit[masks.size()][k] * output_seg[seg_channel_index + k];
                }
                // ptr[mat_index] = 1 / (1 + exp(0 - ptr[mat_index]));   // no need to do sigmoid, as sigmoid is mono
                seg_spatial_index++;
                mat_index++;
            }
        }
        cv::resize(mask_mat, mask_mat, cv::Size(bbox.w, bbox.h), cv::INTER_LINEAR);
        masks.push_back(mask_mat > 0);  // no need to do sigmoid, as sigmoid is mono; change threshold from 0.5 to 0
    }
#else   // official implementation, check https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/ops.py#L601
    int32_t seg_crop_w = seg_grid_width - 2 * seg_grid_offset_x;
    int32_t seg_crop_h = seg_grid_height - 2 * seg_grid_offset_y;
    for (const auto& seg_cls_weight : seg_channel_weights_nms_lit) {
        cv::Mat mask_mat(seg_grid_height, seg_grid_width, CV_32FC1, cv::Scalar(0));
        float* ptr = (float*)mask_mat.data;
        for (int32_t i = 0; i < seg_grid_height * seg_grid_width; i++) {
            int32_t segcls_index = i * kSegChannelNum;
            for (int32_t k = 0; k < kSegChannelNum; k++) {
                ptr[i] += seg_cls_weight[k] * output_seg[segcls_index + k];
            }
            ptr[i] = 1.0 / (1 + exp(0 - ptr[i]));   // sigmoid
        }
        // cv::imwrite("./output.jpg", mask_mat * 255);
        mask_mat = mask_mat(cv::Rect(seg_grid_offset_x, seg_grid_offset_y, seg_crop_w, seg_crop_h));
        cv::resize(mask_mat, mask_mat, cv::Size(original_w, original_h), cv::INTER_LINEAR);
        auto bbox = bbox_nms_list[masks.size()];
        masks.push_back(mask_mat(cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h)) > 0.5);
    }
#endif

    const auto& t_post_process2 = std::chrono::steady_clock::now();
    result.bbox_list = bbox_nms_list;
    result.mask_list = masks;
    result.process_time = 1.0 * (t_post_process2 - t_pre_process0).count() * 1e-6;
    std::cout << "pre-process: " << std::setw(8) << 1.0 * (t_pre_process1 - t_pre_process0).count() * 1e-6   << " ms" << std::endl;
    std::cout << "inference:   " << std::setw(8) << 1.0 * (t_post_process0 - t_pre_process1).count() * 1e-6  << " ms" << std::endl;
    std::cout << "post-process:" << std::setw(8) << 1.0 * (t_post_process2 - t_post_process0).count() * 1e-6 << " ms" << std::endl;

    return 1;
}