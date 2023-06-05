#ifndef OPENVINO_ROADSEG_
#define OPENVINO_ROADSEG_
#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../bmrun_helper.h"

class OpenvinoRoadseg {
public:
    struct Result {
        cv::Mat output_mat;
        float process_time;
    };

public:
    int32_t Initialize(const std::string& model);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& original_mat, Result& result);
    int32_t Process(cv::Mat& original_mat, Result& result);
    
private:
    std::unique_ptr<BmrunHelper> bmrun_helper_;
};

#endif