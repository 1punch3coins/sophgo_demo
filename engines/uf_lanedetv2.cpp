#include <iostream>
#include <chrono>
#include <algorithm>
#include <math.h>
#include <opencv2/opencv.hpp>

#include "uf_lanedetv2.h"

#define INPUT_NCHW true
#define INPUT_RGB true
#define OUTPUT_NLC true
#define IDENTIFIER "lane_det"

static constexpr int32_t kInputTensorNum = 1;
static constexpr int32_t kOutputTensorNum = 4;
static constexpr std::array<const char*, 1> sInputNameList = {"data"};
static constexpr std::array<const char*, 4> sOutputNameList = {"loc_row_Reshape_f32", "loc_col_Reshape_f32", "exist_row_Reshape_f32", "exist_col_Reshape_f32"};

static constexpr float qMeanList[] = {0.485, 0.456, 0.406};
static constexpr float qNormList[] = {0.229, 0.224, 0.225};
static constexpr int32_t kOutputChannelNum = 4;
static constexpr int32_t kOutputRowNum = 72;
static constexpr int32_t kOutputColNum = 81;
static constexpr int32_t kOutputRowAnchorNum = 200;
static constexpr int32_t kOutputColAnchorNum = 100;
static constexpr int32_t kRowLaneList[] = {1, 2};
static constexpr int32_t kColLaneList[] = {0, 3};

int32_t UfLanedetv2::Initialize(const std::string& model_pwd) {
    NetworkMeta* p_meta = new NetworkMeta(NetworkMeta::kTensorTypeFloat32, INPUT_NCHW, INPUT_RGB, OUTPUT_NLC, kInputTensorNum, kOutputTensorNum);
    p_meta->normalize.mean[0] = qMeanList[0];
    p_meta->normalize.mean[1] = qMeanList[1];
    p_meta->normalize.mean[2] = qMeanList[2];
    p_meta->normalize.norm[0] = qNormList[0];
    p_meta->normalize.norm[1] = qNormList[1];
    p_meta->normalize.norm[2] = qNormList[2];
    for (const auto& input_name : sInputNameList) {
        p_meta->AddInputTensorMeta(input_name);
    }
    for (const auto& output_name : sOutputNameList) {
        p_meta->AddOutputTensorMeta(output_name);
    }
    bmrun_helper_.reset(BmrunHelper::Create(model_pwd, kTaskTypeLaneDet, p_meta));

    if (!bmrun_helper_) {
        return 0;
    }

    if (bmrun_helper_->Initialize() != 1) {
        std::cout << "bmrun_helper initialization failed" << std::endl;
        bmrun_helper_.reset();
        return 0;
    }

    for (const auto& output_name : sOutputNameList) {
        if (bmrun_helper_->GetOutputChannelNum(output_name) != kOutputChannelNum) {
            std::cout << "output channel size mismatched" << std::endl;
            return 0;
        }
        if (output_name == "loc_row") {
            if (bmrun_helper_->GetOutputHeight(output_name) != kOutputRowAnchorNum) {
                std::cout << "output height size mismatched" << std::endl;
                return 0;
            }
            if (bmrun_helper_->GetOutputWidth(output_name) != kOutputRowNum) {
                std::cout << "output width size mismatched" << std::endl;
                return 0;
            }
            continue;
        }
        if (output_name == "exist_row") {
            if (bmrun_helper_->GetOutputWidth(output_name) != kOutputRowNum) {
                std::cout << "output width size mismatched" << std::endl;
                return 0;
            }
            continue;
        }
        if (output_name == "loc_col") {
            if (bmrun_helper_->GetOutputHeight(output_name) != kOutputColAnchorNum) {
                std::cout << "output height size mismatched" << std::endl;
                return 0;
            }
            if (bmrun_helper_->GetOutputWidth(output_name) != kOutputColNum) {
                std::cout << "output width size mismatched" << std::endl;
                return 0;
            }
            continue;
        }
        if (output_name == "exist_col") {
            if (bmrun_helper_->GetOutputWidth(output_name) != kOutputColNum) {
                std::cout << "output width size mismatched" << std::endl;
                return 0;
            }
        }
    }

    valid_lanes_rows_.resize(kOutputChannelNum);
    valid_lanes_cols_.resize(kOutputChannelNum);
    for (int32_t i = 0; i < kOutputChannelNum; i++) {
        valid_lanes_rows_[i].reserve(50);
        valid_lanes_cols_[i].reserve(50);
    }
    return 1;
}

int32_t UfLanedetv2::DoSoftmax(const std::vector<float>& input, std::vector<float>& output) {
    std::vector<float> cache;
    cache.reserve(input.size());
    output.reserve(input.size());
    float e = 2.71828182846;
    float sum = 0;
    for (const auto& element : input) {
        float res = std::pow(e, element);
        cache.push_back(res);
        sum += res;
    }
    for (const auto& element : cache) {
        output.push_back(element / sum);
    }
    return 1;
}

int32_t UfLanedetv2::Process(cv::Mat& original_mat, Result& result) {
    // 1. pre-process
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    int32_t original_w = original_mat.cols;
    int32_t original_h = original_mat.rows;
    bmrun_helper_->PreProcess(original_mat);
    cv::Rect src_crop = bmrun_helper_->GetCropInfo().first;
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    // 2. inference
    bmrun_helper_->Inference();

    // 3.1 post-process, retrive outputs
    const auto& t_post_process0 = std::chrono::steady_clock::now();
    float* loc_row = bmrun_helper_->GetInfernceOutput(sOutputNameList[0]);
    float* loc_col = bmrun_helper_->GetInfernceOutput(sOutputNameList[1]);
    float* exist_row = bmrun_helper_->GetInfernceOutput(sOutputNameList[2]);
    float* exist_col = bmrun_helper_->GetInfernceOutput(sOutputNameList[3]);

    int32_t offset_row = kOutputRowNum * kOutputChannelNum;
    int32_t offset_col = kOutputColNum * kOutputChannelNum;
    for (int32_t i = 0; i < offset_row; i++) {
        if (exist_row[i] < exist_row[i + offset_row]) {
            valid_lanes_rows_[i % kOutputChannelNum].push_back(i / kOutputChannelNum);
        }
    }
    for (int32_t i = 0; i < offset_col; i++) {
        if (exist_col[i] < exist_col[i + offset_col]) {
            valid_lanes_cols_[i % kOutputChannelNum].push_back(i / kOutputChannelNum);
        }
    }

    // 3.2 post-process, scan the output "loc_row" vertically
    std::vector<std::vector<std::pair<float, float>>> scanned_lanes_poses(kOutputChannelNum);
    offset_row = kOutputRowNum * kOutputChannelNum;
    float scale_width = static_cast<float>(original_w) / (kOutputRowAnchorNum - 1);
    float scale_height = static_cast<float>(src_crop.height) / kOutputRowNum;
    // loop for each lane scanned by row
    for (const auto& lane_index : kRowLaneList) {
        // if the lane has enough valid points scanned by row
        if (valid_lanes_rows_[lane_index].size() > kOutputRowNum / 2) {
            // loop for every valid row
            for (const auto& row_position : valid_lanes_rows_[lane_index]) {
                int32_t start_index = row_position * kOutputChannelNum + lane_index;
                int32_t max_conf_index = 0;
                float max_confidence = 0;
                // loop for every anchor
                for (int32_t i = 0; i < kOutputRowAnchorNum; i++) {
                    float cur_confidence = loc_row[start_index + i * offset_row];
                    if (cur_confidence > max_confidence) {
                        max_confidence = cur_confidence;
                        max_conf_index = i;
                    }
                }
                float avg_col_pos;
                // do avg weighted locs
                if (max_conf_index > 1 && max_conf_index < kOutputRowAnchorNum - 1) {
                    float pre_anchor_conf = loc_row[start_index + (max_conf_index - 1) * offset_row];
                    float post_anchor_conf = loc_row[start_index + (max_conf_index + 1) * offset_row];
                    std::vector<float> input = {pre_anchor_conf, max_confidence, post_anchor_conf};
                    std::vector<float> weight;
                    DoSoftmax(input, weight);
                    avg_col_pos = weight[0] * (max_conf_index - 1) + weight[1] * max_conf_index + weight[2] * ((max_conf_index + 1)) + 0.5;
                } else {
                    avg_col_pos = max_conf_index;
                }
                int32_t scaled_x = static_cast<float>(avg_col_pos * scale_width) + src_crop.x;
                int32_t scaled_y = static_cast<float>(row_position * scale_height) + src_crop.y;
                scanned_lanes_poses[lane_index].push_back(std::pair<float, float>(avg_col_pos, row_position));
                result.lanes[lane_index].push_back(Point2D(scaled_x, scaled_y));
            }
        }
    }

    // 3.3 post-process, scan the output "loc_col" horizontally
    offset_col = kOutputColNum * kOutputChannelNum;
    scale_width = static_cast<float>(src_crop.width) / kOutputColNum;
    scale_height = static_cast<float>(original_h) / (kOutputColAnchorNum - 1);
    // loop for each lane scanned by col
    for (const auto& lane_index : kColLaneList) {
        // if the lane has enough valid points scanned by col
        if (valid_lanes_cols_[lane_index].size() > kOutputColNum / 4) {
            // loop for every valid col
            for (const auto& col_position : valid_lanes_cols_[lane_index]) {
                int32_t start_index = col_position * kOutputChannelNum + lane_index;
                int32_t max_conf_index = 0;
                float max_confidence = 0;
                // loop for every anchor
                for (int32_t i = 0; i < kOutputColAnchorNum; i++) {
                    float cur_confidence = loc_col[start_index + i * offset_col];
                    if (cur_confidence > max_confidence) {
                        max_confidence = cur_confidence;
                        max_conf_index = i;
                    }
                }
                float avg_row_pos;
                // do avg weighted locs
                if (max_conf_index > 1 && max_conf_index < kOutputRowAnchorNum - 1) {
                    float pre_anchor_conf = loc_col[start_index + (max_conf_index - 1) * offset_row];
                    float post_anchor_conf = loc_col[start_index + (max_conf_index + 1) * offset_row];
                    std::vector<float> input = {pre_anchor_conf, max_confidence, post_anchor_conf};
                    std::vector<float> weight;
                    DoSoftmax(input, weight);
                    avg_row_pos = weight[0] * (max_conf_index - 1) + weight[1] * max_conf_index + weight[2] * ((max_conf_index + 1)) + 0.5;
                } else {
                    avg_row_pos = max_conf_index;
                }
                int32_t scaled_x = static_cast<float>(avg_row_pos * scale_width) + src_crop.x;
                int32_t scaled_y = static_cast<float>(col_position * scale_height) + src_crop.y;
                scanned_lanes_poses[lane_index].push_back(std::pair<float, float>(col_position, avg_row_pos));
                result.lanes[lane_index].push_back(Point2D(scaled_x, scaled_y));
            }
        }
    }

    // 3.4 post-process, rescale the lane points
    // int32_t valid_lane_num = 0;
    // for (int32_t i = 0; i < kOutputChannelNum; i++) {
    //     if (scanned_lanes_poses[i].size() > 0) {
    //         valid_lane_num++;
    //         result.lanes.resize(valid_lane_num);
    //         result.lanes[valid_lane_num-1].first.reserve(40);
    //         result.lanes[valid_lane_num-1].second = i;
    //         for (const auto& scanned_lane_point : scanned_lanes_poses[i]) {
    //             int32_t scaled_x = static_cast<int32_t>(scanned_lane_point.first * (scale_width));
    //             int32_t scaled_y = static_cast<int32_t>(scanned_lane_point.second * (scale_height)) + src_crop.y;
    //             result.lanes[valid_lane_num-1].first.push_back(Point2D(scaled_x, scaled_y));
    //         }
    //     }
    // }

    const auto& t_post_process1 = std::chrono::steady_clock::now();
    std::cout << "pre-process: " << std::setw(8) << 1.0 * (t_pre_process1 - t_pre_process0).count() * 1e-6   << " ms" << std::endl;
    std::cout << "inference:   " << std::setw(8) << 1.0 * (t_post_process0 - t_pre_process1).count() * 1e-6  << " ms" << std::endl;
    std::cout << "post-process:" << std::setw(8) << 1.0 * (t_post_process1 - t_post_process0).count() * 1e-6 << " ms" << std::endl;;

    return 1;
}