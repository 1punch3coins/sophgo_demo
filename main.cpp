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
    // request a bm_handle
    // bm_handle_t bm_handle;
    // bm_status_t status = bm_dev_request(&bm_handle, 0);
    // if (status != BM_SUCCESS) {
    //     std::cout << "!!!" << std::endl;
    //     return 0;   
    // }
    // bm_image bm_mat;
    // bm_image bm_mat_2;
    // cv::Mat original_img = cv::imread("./inputs/test1.png");
    // cv::bmcv::toBMI(original_img, &bm_mat, true);

    // bm_image_create(bm_handle, 640, 640, FORMAT_RGB_PLANAR, (bm_image_data_format_ext)DATA_TYPE_EXT_1N_BYTE, &bm_mat_2);

    // bmcv_resize_image resize_attr;
    // bmcv_resize_t attr;
    // attr.start_x = 0;
    // attr.start_y = 0;
    // attr.in_width = bm_mat.width;
    // attr.in_height = bm_mat.height;
    // attr.out_width = 640;
    // attr.out_height = 640;
    // resize_attr.roi_num = 1;
    // resize_attr.resize_img_attr = &attr;
    // bmcv_image_resize(bm_handle, 1, &resize_attr, &bm_mat, &bm_mat_2);

    // bm_handle_t handle;
    // bm_status_t status = bm_dev_request(&handle, 0);
    // int image_num = 4;
    // int crop_w = 711, crop_h = 400, resize_w = 711, resize_h = 400;
    // int image_w = 1920, image_h = 1080;
    // int img_size_i = image_w * image_h * 3;
    // int img_size_o = resize_w * resize_h * 3;
    // std::unique_ptr<unsigned char[]> img_data(new unsigned char[img_size_i * image_num]);
    // std::unique_ptr<unsigned char[]> res_data(new unsigned char[img_size_o * image_num]);
    // memset(img_data.get(), 0x11, img_size_i * image_num);
    // memset(res_data.get(), 0, img_size_o * image_num);
    // bmcv_resize_image resize_attr[image_num];
    // bmcv_resize_t resize_img_attr[image_num];
    // for (int img_idx = 0; img_idx < image_num; img_idx++) {
    //     resize_img_attr[img_idx].start_x = 0;
    //     resize_img_attr[img_idx].start_y = 0;
    //     resize_img_attr[img_idx].in_width = crop_w;
    //     resize_img_attr[img_idx].in_height = crop_h;
    //     resize_img_attr[img_idx].out_width = resize_w;
    //     resize_img_attr[img_idx].out_height = resize_h;
    // }
    // for (int img_idx = 0; img_idx < image_num; img_idx++) {
    //     resize_attr[img_idx].resize_img_attr = &resize_img_attr[img_idx];
    //     resize_attr[img_idx].roi_num = 1;
    //     resize_attr[img_idx].stretch_fit = 1;
    //     resize_attr[img_idx].interpolation = BMCV_INTER_LINEAR;
    // }
    // bm_image input[image_num];
    // bm_image output[image_num];
    // for (int img_idx = 0; img_idx < image_num; img_idx++) {
    //     int input_data_type = DATA_TYPE_EXT_1N_BYTE;
    //     bm_image_create(handle,
    //         image_h,
    //         image_w,
    //         FORMAT_BGR_PLANAR,
    //         (bm_image_data_format_ext)input_data_type,
    //         &input[img_idx]);
    // }
    // bm_image_alloc_contiguous_mem(image_num, input, 1);
    // for (int img_idx = 0; img_idx < image_num; img_idx++) {
    //     unsigned char * input_img_data = img_data.get() + img_size_i * img_idx;
    //     bm_image_copy_host_to_device(input[img_idx],
    //             (void **)&input_img_data);
    // }
    // for (int img_idx = 0; img_idx < image_num; img_idx++) {
    //     int output_data_type = DATA_TYPE_EXT_1N_BYTE;
    //     bm_image_create(handle,
    //             resize_h,
    //             resize_w,
    //             FORMAT_BGR_PLANAR,
    //             (bm_image_data_format_ext)output_data_type,
    //             &output[img_idx]);
    // }
    // bm_image_alloc_contiguous_mem(image_num, output, 1);
    // const auto& t1 = std::chrono::steady_clock::now();
    // bmcv_image_resize(handle, image_num, resize_attr, input, output);
    // const auto& t2 = std::chrono::steady_clock::now();
    // for (int img_idx = 0; img_idx < image_num; img_idx++) {
    //     unsigned char *res_img_data = res_data.get() + img_size_o * img_idx;
    //     bm_image_copy_device_to_host(output[img_idx],
    //     (void **)&res_img_data);
    // }
    // std::cout << 1.0 * (t2 - t1).count() * 1e-6 << std::endl;
    // bm_image_free_contiguous_mem(image_num, input);
    // bm_image_free_contiguous_mem(image_num, output);
    // for(int i = 0; i < image_num; i++) {
    //     bm_image_destroy(input[i]);
    //     bm_image_destroy(output[i]);
    // }

    // // create a bmruntime
    // void *p_bmrt = bmrt_create(bm_handle);
    // if (p_bmrt == NULL) {
    //     std::cout << "!!!" << std::endl;
    //     return 0;  
    // }

    // // load bmodel by file
    // bool ret = bmrt_load_bmodel(p_bmrt, "yolov5s_fp32.bmodel");
    // if (ret != true) {
    //     std::cout << "!!!" << std::endl;
    //     return 0;  
    // }

    // // make sure the bmodel file is loaded and get the network's name in bmrt
    // const char **net_names = NULL;
    // bmrt_get_network_names(p_bmrt, &net_names);
    // auto net_info = bmrt_get_network_info(p_bmrt, net_names[0]);
    // if (net_info == NULL) {
    //     std::cout << "!!!" << std::endl;
    //     return 0;  
    // }

    // // init input tensor && output_tensor
    // bm_tensor_t input_tensor, output_tensor;
    // bm_shape_t input_shape = net_info->stages[0].input_shapes[0];
    // bmrt_tensor(&input_tensor, p_bmrt, net_info->input_dtypes[0], input_shape);
    // int32_t input_element_count = bmrt_shape_count(&input_shape);
    // int32_t net_h = input_shape.dims[2];
    // int32_t net_w = input_shape.dims[3];
    // int32_t batch_size = input_shape.dims[0];
    // int32_t num_channels = input_shape.dims[1];
    // float input_scale = net_info->input_scales[0];
    // float* input_f32 = new float[input_element_count];

    // bm_shape_t output_shape = net_info->stages[0].output_shapes[0];
    // bmrt_tensor(&output_tensor, p_bmrt, net_info->output_dtypes[0], output_shape);
    // int32_t output_element_count = bmrt_shape_count(&output_shape);
    // float output_scale = net_info->output_scales[0];
    // float* output_f32 = new float[output_element_count];

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
