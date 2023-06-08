#include <iostream>
#include <opencv2/opencv.hpp>
#include "engines/yolov5.h"
#include "engines/yolov8.h"
#include "engines/openvino_roadseg.h"
#include "bmruntime_interface.h"

#include "bmcv_api.h"
#include "bmcv_api_ext.h"

#include <chrono>

int main(int argc, char* argv[]) {
    cv::Mat original_img = cv::imread("./inputs/test2.png");
    Yolov8 yolo;
    if (yolo.Initialize("./models/yolov8s_post_1684x_fp16.bmodel") != 1) {
        std::cout << "yolo initialization uncompleted" << std::endl;
        return 0;
    }
    OpenvinoRoadseg roadseg;
    if (roadseg.Initialize("./models/roadseg_320x896_concat_params_reset_1684x_fp16.bmodel") != 1) {
        std::cout << "road_seg initialization uncompleted" << std::endl;
        return 0;
    } 
    // while (true) {
        cv::Mat res_img;
        OpenvinoRoadseg::Result seg_res;
        if (roadseg.Process(original_img, seg_res) != 1) {
            std::cout << "roadseg forward uncompleted" << std::endl;
        }
        cv::add(original_img, seg_res.output_mat, res_img);
        // cv::imwrite("./outputs/output1.jpg", original_img);

        Yolov8::Result det_res;
        if (yolo.Process(original_img, det_res) != 1) {
            std::cout << "yolo forward uncompleted" << std::endl;
            return 0;
        }
        for (const auto& box: det_res.bbox_list) {
            cv::putText(res_img, std::to_string(box.cls_id), cv::Point(box.x, box.y - 6), 0, 0.5, cv::Scalar(0, 255, 0), 1);
            cv::rectangle(res_img, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite("./outputs/output.jpg", res_img);
    // }
    return 0;
}
