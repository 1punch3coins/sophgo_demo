#include <iostream>
#include <opencv2/opencv.hpp>
#include "engines/yolov5.h"
#include "engines/yolov8.h"
#include "engines/yolov8_seg.h"
#include "engines/openvino_roadseg.h"
#include "engines/uf_lanedetv2.h"
#include "bmruntime_interface.h"

#include "bmcv_api.h"
#include "bmcv_api_ext.h"
#include "tracker/target_tracker.h"

#include <chrono>

const std::vector<cv::Scalar> mask_colors = {
    {0, 128, 128}, {72, 105, 72}, {91, 86, 105}, {100, 65, 100},
    {30, 65, 165}, {10, 90, 100}, {0, 0, 139}, {0, 69, 255}
};

int main(int argc, char* argv[]) {
    cv::Mat original_img = cv::imread("./inputs/test3.png");
    Yolov8 yolo;
    if (yolo.Initialize("./resource/models/yolov8s_post_1684x_fp16.bmodel") != 1) {
        std::cout << "yolo initialization uncompleted" << std::endl;
        return 0;
    }
    Yolov8Seg yolo_seg;
    if (yolo_seg.Initialize("./resource/models/yolov8s_seg_1684x_fp16.bmodel") != 1) {
        std::cout << "yolo_seg initialization uncompleted" << std::endl;
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
    
    cv::VideoCapture cap("./resource/inputs/campus_seg.avi");
    int32_t frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int32_t frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int32_t frame_fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter video_writer("./resource/outputs/campus_output.avi", cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 60, cv::Size(frame_width, frame_height), true);
    if (!video_writer.isOpened()) {return 1;}
    cap >> original_img;
    static int32_t i = 0;
    Tracker tracker(frame_fps);
    while (!original_img.empty()) {
        cv::Mat res_img;
        OpenvinoRoadseg::Result seg_res;
        if (roadseg.Process(original_img, seg_res) != 1) {
            std::cout << "roadseg forward uncompleted" << std::endl;
        }
        cv::add(original_img, seg_res.output_mat, res_img);

        Yolov8Seg::Result det_res;
        if (yolo_seg.Process(original_img, det_res) != 1) {
            std::cout << "yolo forward uncompleted" << std::endl;
            return 0;
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

        std::vector<const Bbox2D*> unmatched_bboxes;
        tracker.Update(det_res.bbox_list, det_res.mask_list, unmatched_bboxes);
        // for (const auto& box : unmatched_bboxes) {
        //     const auto bbox = *box;
        //     cv::putText(res_img, bbox.cls_name + " " + std::to_string(bbox.cls_confidence).substr(0, 4), cv::Point(bbox.x, bbox.y - 6), 0, 0.7, cv::Scalar(100, 100, 100), 2);
        //     cv::rectangle(res_img, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), cv::Scalar(100, 100, 100), 2);            
        // }
        cv::Mat mask_img(original_img.rows, original_img.cols, CV_8UC3, cv::Scalar(0));
        for (const auto& tracklet: *(tracker.GetNewTracklets())) {
            const auto& box = tracklet.get()->State2Bbox();
            const auto& mask_mat = tracklet.get()->GetTrackMask();
            mask_img(cv::Rect(box.x, box.y, box.w, box.h)).setTo(mask_colors[tracklet.get()->GetTrackId()%mask_colors.size()], mask_mat);
            cv::putText(res_img, std::to_string(tracklet.get()->GetTrackId()) + " " + box.cls_name, cv::Point(box.x, box.y - 6), 0, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(res_img, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
        }
        for (const auto& tracklet: *(tracker.GetActiveTracklets())) {
            const auto& box = tracklet.get()->State2Bbox();            
            const auto& mask_mat = tracklet.get()->GetTrackMask();
            int32_t x = std::max(0, box.x);
            int32_t y = std::max(0, box.y);
            int32_t w = box.w + x > frame_width ? frame_width - x : box.w;
            int32_t h = box.h + y > frame_height ? frame_height - y : box.h;

            cv::Mat mask_mat_(h, w, CV_8UC1, cv::Scalar(0));
            cv::resize(mask_mat, mask_mat_, cv::Size(w, h), cv::INTER_NEAREST);
            mask_img(cv::Rect(x, y, w, h)).setTo(mask_colors[tracklet.get()->GetTrackId()%mask_colors.size()], mask_mat_);
            int32_t len = tracklet.get()->GetTrackletLength();
            int32_t green_scalar = 255 - len / 2;
            int32_t red_scalar = len;
            green_scalar = green_scalar > 0 ? green_scalar : 0;
            red_scalar = red_scalar < 255 ? red_scalar : 255;
            cv::putText(res_img, std::to_string(tracklet.get()->GetTrackId()) + " " + box.cls_name, cv::Point(x, y - 6), 0, 0.7, cv::Scalar(0, green_scalar, red_scalar), 2);
            cv::rectangle(res_img, cv::Rect(x, y, w, h), cv::Scalar(0, green_scalar, red_scalar), 2);
        }
        cv::add(res_img, mask_img, res_img);

        float process_time = seg_res.process_time + det_res.process_time + lane_det_res.process_time;
        float fps = 1000.0 / process_time;
        cv::putText(res_img, "FPS: " + std::to_string(fps).substr(0, 4), cv::Point(30, 30), 0, 0.9, cv::Scalar(0, 255, 0), 2);
        cv::putText(res_img, "Frame: " + std::to_string(i), cv::Point(180, 30), 0, 0.9, cv::Scalar(0, 255, 0), 2);
        video_writer.write(res_img);
        cap >> original_img;
        std::cout << i++ << std::endl;
    }
    cap.release();
    video_writer.release();
    return 0;
}
