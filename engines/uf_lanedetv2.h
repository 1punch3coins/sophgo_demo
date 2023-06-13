#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../bmrun_helper.h"
#include "../det_structs.h"

class UfLanedetv2 {
public:
    struct Result {
        std::vector<Lane2D> lanes;
        float process_time;
        Result() {lanes.resize(4);}
    };

public:
    UfLanedetv2()
    {}

public:
    int32_t Initialize(const std::string& model_pwd);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& original_mat, Result& result);
    int32_t Process(cv::Mat& original_mat, Result& result);
    
private:
    int32_t DoSoftmax(const std::vector<float>& input, std::vector<float>& output);
    std::unique_ptr<BmrunHelper> bmrun_helper_;
    std::vector<std::vector<int32_t>> valid_lanes_rows_;
    std::vector<std::vector<int32_t>> valid_lanes_cols_;
};