#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "openvino_roadseg.h"

#define INPUT_NCHW true
#define INPUT_RGB false
#define OUTPUT_NLC true
#define MODEL_POST_ORIGIN true
#define IDENTIFIER "road_seg"

static constexpr std::array<const char*, 1> sInputNameList = {"data"};
static constexpr std::array<const char*, 1> sOutputNameList = {"tf.identity_Softmax"};
static constexpr float qMeanList[] = {0.0, 0.0, 0.0};
static constexpr float qNormList[] = {1/255.0, 1/255.0, 1/255.0};
static constexpr int32_t kOutputChannelNum = 4;
static constexpr int32_t kOutputHeight = 320;
static constexpr int32_t kOutputWidth = 896;

int32_t OpenvinoRoadseg::Initialize(const std::string& model) {
    NetworkMeta* p_meta = new NetworkMeta(NetworkMeta::kTensorTypeFloat32, INPUT_NCHW, INPUT_RGB, OUTPUT_NLC);
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
    bmrun_helper_.reset(BmrunHelper::Create(model, kTaskTypeRoadSeg, p_meta));

    if (!bmrun_helper_) {
        return 0;
    }

    if (bmrun_helper_->Initialize() != 1) {
        std::cout << "bmrun_helper initialization failed" << std::endl;
        bmrun_helper_.reset();
        return 0;
    }

    if (bmrun_helper_->GetOutputChannelNum() != kOutputChannelNum) {
        std::cout << "output channel size mismatched" << std::endl;
        return 0;
    }

    if (bmrun_helper_->GetOutputHeight() != kOutputHeight) {
        std::cout << "output height size mismatched" << std::endl;
        return 0;
    }

    if (bmrun_helper_->GetOutputWidth() != kOutputWidth) {
        std::cout << "output width size mismatched" << std::endl;
        return 0;
    }

    return 1;
}

int32_t OpenvinoRoadseg::Process(cv::Mat& original_mat, Result& result) {
    // 1. prep-rocess
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    bmrun_helper_->PreProcess(original_mat);
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    // 2. inference
    bmrun_helper_->Inference();

    // 3. post-process, retrive output and render result's mat
    const auto& t_post_process0 = std::chrono::steady_clock::now();
    float* output = bmrun_helper_->GetInfernceOutput();
    cv::Mat res(kOutputHeight, kOutputWidth, CV_8UC3);
    res.setTo(0);
    cv::Vec3b* ptr = (cv::Vec3b*)res.data;
#pragma omp parallel for num_threads(4)
    for (int32_t y = 0; y < kOutputHeight; y++) {
        for (int32_t x = 0; x < kOutputWidth; x++) {
            // float* scores = &output[y * (kOutputWidth * kOutputChannelNum) + x * kOutputChannelNum + 0];
            float* scores = output + y * (kOutputWidth * kOutputChannelNum) + x * kOutputChannelNum + 0;
            uint8_t b, g, r;
            if (scores[0] > 0.7 && scores[1] < 0.3 && scores[2] < 0.3 && scores[3] < 0.3) {
                b = 0;
                g = 0;
                r = 0;
                ptr[y * kOutputWidth + x] = { b, g, r };
                continue;
            }
            if (scores[1] > 0.3 && scores[1] > scores[2] && scores[1] > scores[3]) {
                b = 255;
                g = 0;
                r = 0;
                ptr[y * kOutputWidth + x] = { b, g, r };
                continue;
            }
            if (scores[2] > scores[1] && scores[2] > scores[3] && scores[2] > 0.3) {
                // b = 80;
                // g = 80;
                // r = 150;
                b = 0;
                g = 0;
                r = 0;
                ptr[y * kOutputWidth + x] = { b, g, r };
                continue;
            }
            if (scores[3] > scores[2] && scores[3] > scores[1] && scores[3] > 0.3) {
                // b = 0;
                // g = 0;
                // r = 255;
                b = 255;
                g = 0;
                r = 0;
                ptr[y * kOutputWidth + x] = { b, g, r };
                continue;
            }
        }
    }
    result.output_mat.create(original_mat.rows, original_mat.cols, CV_8UC3); // cv::create is difference, height is the first arg.
    result.output_mat.setTo(0);
    cv::Rect src_crop = bmrun_helper_->GetCropInfo().first;
    cv::resize(res, result.output_mat(src_crop), cv::Size(src_crop.width, src_crop.height), 0, 0, cv::INTER_NEAREST);
    const auto& t_post_process1 = std::chrono::steady_clock::now();
    
    result.process_time = 1.0 * (t_post_process1 - t_pre_process0).count() * 1e-6;
    std::cout << "pre-process: " << std::setw(8) << 1.0 * (t_pre_process1 - t_pre_process0).count() * 1e-6   << " ms" << std::endl;
    std::cout << "inference:   " << std::setw(8) << 1.0 * (t_post_process0 - t_pre_process1).count() * 1e-6  << " ms" << std::endl;
    std::cout << "post-process:" << std::setw(8) << 1.0 * (t_post_process1 - t_post_process0).count() * 1e-6 << " ms" << std::endl;

    return 1;
}