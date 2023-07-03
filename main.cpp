#include <iostream>
#include <opencv2/opencv.hpp>
#include "engines/yolov5.h"
#include "engines/yolov8.h"
#include "engines/openvino_roadseg.h"
#include "engines/uf_lanedetv2.h"
#include "bmruntime_interface.h"

#include "bmcv_api.h"
#include "bmcv_api_ext.h"

#include <chrono>

int main(int argc, char* argv[]) {
    cv::Mat original_img = cv::imread("./resource/inputs/test3.png");
    Yolov8 yolo;
    if (yolo.Initialize("./resource/models/yolov8s_post_1684x_fp16.bmodel") != 1) {
        std::cout << "yolo initialization uncompleted" << std::endl;
        return 0;
    }
    OpenvinoRoadseg roadseg;
    if (roadseg.Initialize("./resource/models/roadseg_320x896_concat_params_reset_1684x_fp16.bmodel") != 1) {
        std::cout << "road_seg initialization uncompleted" << std::endl;
        return 0;
    }
    UfLanedetv2 uf_lanedet;
    if (uf_lanedet.Initialize("./resource/models/ufldv2_r18_320x1600_1684x_fp16.bmodel") != 1) {
        std::cout << "lane_det initialization uncompleted" << std::endl;
        return 0;
    }
    
    // while (!original_img.empty()) {
        cv::Mat res_img;
        OpenvinoRoadseg::Result seg_res;
        if (roadseg.Process(original_img, seg_res) != 1) {
            std::cout << "roadseg forward uncompleted" << std::endl;
        }
        cv::add(original_img, seg_res.output_mat, res_img);

        Yolov8::Result det_res;
        if (yolo.Process(original_img, det_res) != 1) {
            std::cout << "yolo forward uncompleted" << std::endl;
            return 0;
        }
        for (const auto& box: det_res.bbox_list) {
            cv::putText(res_img, box.cls_name, cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(res_img, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
        }

        UfLanedetv2::Result lane_det_res;
        if (uf_lanedet.Process(original_img, lane_det_res) != 1) {
            std::cout << "uf_lanedet forward uncompleted" << std::endl;
            return 0;
        }
        for (const auto& lane: lane_det_res.lanes) {
            if (lane.size() > 0) {
                for (const auto& lane_point : lane) {
                    cv::circle(res_img, cv::Point(lane_point.x, lane_point.y), 4, cv::Scalar(0, 255, 0), 2);
                }
            }
        }
        cv::imwrite("./resource/outputs/output3.jpg", res_img);
    // }
    return 0;
}
