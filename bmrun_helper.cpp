#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "bmruntime_interface.h"
// #include "bmdef.h"
#include "bmrun_helper.h"

#define USE_BMCV true

BmrunHelper* BmrunHelper::Create(const std::string& model, NetworkMeta* p_meta) {
    BmrunHelper*p = new BmrunHelper();
    p->model_pwd_ = model.c_str();
    p->network_meta_.reset(p_meta);
    return p;
}

int32_t BmrunHelper::Initialize() {
    // 1.
    if (bm_dev_request(&bm_handle_, 0) != BM_SUCCESS)
        return 0;
    p_bmrt_ = bmrt_create(bm_handle_);
    if (p_bmrt_ == NULL)
        return 0;
    if (bmrt_load_bmodel(p_bmrt_, model_pwd_) != true)
        return 0;
    bmrt_get_network_names(p_bmrt_, &net_names_);
    auto net_info = bmrt_get_network_info(p_bmrt_, net_names_[0]);
    if (network_meta_->input_tensor_meta_list.size() != net_info->input_num) {
        return 0;
    }
    if (network_meta_->output_tensor_meta_list.size() != net_info->output_num) {
        return 0;
    }
    network_meta_->input_tensor_num = network_meta_->input_tensor_meta_list.size();
    network_meta_->output_tensor_num = network_meta_->output_tensor_meta_list.size();
    for (int32_t i = 0; i < network_meta_->input_tensor_num; i++) {
        network_meta_->input_name2index.insert({network_meta_->input_tensor_meta_list[i].tensor_name, i});
    }
    for (int32_t i = 0; i < network_meta_->output_tensor_num; i++) {
        network_meta_->output_name2index.insert({network_meta_->output_tensor_meta_list[i].tensor_name, i});
    }

    // 2.1 readin model's input metadata // if input meta's order and network's tensor order not matched, TO DO
    char const** input_names = net_info->input_names;
    for (auto& input_tensor_meta : network_meta_->input_tensor_meta_list) {
        int32_t index = 0;
        for (; index < net_info->input_num; index++) {
            if (std::string(input_names[index]) == input_tensor_meta.tensor_name)
                break;
        }
        bm_shape_t input_shape = net_info->stages[0].input_shapes[index];
        if (network_meta_->input_nchw) {
            input_tensor_meta.net_in_c = input_shape.dims[1];
            input_tensor_meta.net_in_h = input_shape.dims[2];
            input_tensor_meta.net_in_w = input_shape.dims[3];
        } else {
            input_tensor_meta.net_in_h = input_shape.dims[1];
            input_tensor_meta.net_in_w = input_shape.dims[2];
            input_tensor_meta.net_in_c = input_shape.dims[3];
        }
        network_meta_->batch_size = input_shape.dims[0];
        input_tensor_meta.input_scale = net_info->input_scales[index];
        input_tensor_meta.net_in_elements = network_meta_->batch_size * input_tensor_meta.net_in_c * input_tensor_meta.net_in_h * input_tensor_meta.net_in_w;
    }

    // 2.2 Readin model's output metadata // if output meta's order and network's tensor order not matched, TO DO
    char const** output_names = net_info->output_names;
    for (auto& output_tensor_meta : network_meta_->output_tensor_meta_list) {
        int32_t index = 0;
        for (; index < net_info->output_num; index++) {
            if (std::string(output_names[index]) == output_tensor_meta.tensor_name)
                break;
        }
        bm_shape_t output_shape = net_info->stages[0].output_shapes[index];
        if (output_shape.num_dims == 3) {
            if (network_meta_->output_nlc) {
                output_tensor_meta.net_out_l = output_shape.dims[1];
                output_tensor_meta.net_out_c = output_shape.dims[2];
            } else {
                output_tensor_meta.net_out_c = output_shape.dims[1];
                output_tensor_meta.net_out_l = output_shape.dims[2];
            }
        }
        if (output_shape.num_dims == 4) {
            if (network_meta_->output_nlc) {
                output_tensor_meta.net_out_h = output_shape.dims[1];
                output_tensor_meta.net_out_w = output_shape.dims[2];
                output_tensor_meta.net_out_c = output_shape.dims[3];
                output_tensor_meta.net_out_l = output_shape.dims[1] * output_shape.dims[2];
            } else {
                output_tensor_meta.net_out_c = output_shape.dims[1];
                output_tensor_meta.net_out_h = output_shape.dims[2];
                output_tensor_meta.net_out_w = output_shape.dims[3];
                output_tensor_meta.net_out_l = output_shape.dims[2] * output_shape.dims[3];
            }
        }
        output_tensor_meta.output_scale = net_info->output_scales[index];
        output_tensor_meta.net_out_elements = network_meta_->batch_size * output_tensor_meta.net_out_c * output_tensor_meta.net_out_l;
    }
    
    // 3. Construct network's input and output space on systeam ram; Construct memory on gpu and bind them
    input_tensors_.resize(network_meta_->input_tensor_num);
    output_tensors_.resize(network_meta_->output_tensor_num);
    for (int32_t i = 0; i < network_meta_->input_tensor_num; i++) {
        const auto& tensor_meta = network_meta_->input_tensor_meta_list[i];
        if (!USE_BMCV) {
            if (net_info->input_dtypes[i] == BM_FLOAT32) {
                network_meta_->tensor_type = NetworkMeta::kTensorTypeFloat32;
                input_ptrs_.push_back(new float[tensor_meta.net_in_elements]);
            } else if (net_info->input_dtypes[i] == BM_INT8) {
                network_meta_->tensor_type = NetworkMeta::kTensorTypeInt8;
                input_ptrs_.push_back(new int8_t[tensor_meta.net_in_elements]);
            }
        } else {
            bm_image_format_ext_ bm_formate_type;
            if (network_meta_->input_rgb) {
                bm_formate_type = FORMAT_RGB_PLANAR;
            } else {
                bm_formate_type = FORMAT_BGR_PLANAR;
            }
            if (net_info->input_dtypes[i] == BM_FLOAT32) {
                network_meta_->tensor_type = NetworkMeta::kTensorTypeFloat32;
                if (!network_meta_->input_nchw) {
                    bm_image_create(bm_handle_, tensor_meta.net_in_h, tensor_meta.net_in_w, bm_formate_type, DATA_TYPE_EXT_1N_BYTE, &bm_mat_resized_);
                    input_ptrs_.push_back(new float[tensor_meta.net_in_elements]);
                } else {
                    bm_image_create(bm_handle_, tensor_meta.net_in_h, tensor_meta.net_in_w, bm_formate_type, DATA_TYPE_EXT_1N_BYTE, &bm_mat_resized_);
                    bm_image_create(bm_handle_, tensor_meta.net_in_h, tensor_meta.net_in_w, bm_formate_type, DATA_TYPE_EXT_FLOAT32, &bm_mat_normalized_);
                }
            } else if (net_info->input_dtypes[i] == BM_INT8) {
                network_meta_->tensor_type = NetworkMeta::kTensorTypeInt8;
                if (!network_meta_->input_nchw) {
                    bm_image_create(bm_handle_, tensor_meta.net_in_h, tensor_meta.net_in_w, bm_formate_type, DATA_TYPE_EXT_1N_BYTE, &bm_mat_resized_);
                    input_ptrs_.push_back(new int8_t[tensor_meta.net_in_elements]);
                } else {
                    bm_image_create(bm_handle_, tensor_meta.net_in_h, tensor_meta.net_in_w, bm_formate_type, DATA_TYPE_EXT_1N_BYTE, &bm_mat_resized_);
                    bm_image_create(bm_handle_, tensor_meta.net_in_h, tensor_meta.net_in_w, bm_formate_type, DATA_TYPE_EXT_1N_BYTE_SIGNED, &bm_mat_normalized_);
                }
            }
        }
        bmrt_tensor(&input_tensors_[i], p_bmrt_, net_info->input_dtypes[i], net_info->stages[0].input_shapes[i]);
    }
    for (int32_t i = 0; i < network_meta_->output_tensor_num; i++) {
        const auto& tensor_meta = network_meta_->output_tensor_meta_list[i];
        if (net_info->output_dtypes[i] == BM_FLOAT32) {
            output_ptrs_.push_back(new float[tensor_meta.net_out_elements]);
        } else if (net_info->output_dtypes[i] == BM_INT8) {
            output_ptrs_.push_back(new int8_t[tensor_meta.net_out_elements]);
        }
        bmrt_tensor(&output_tensors_[i], p_bmrt_, net_info->output_dtypes[i], net_info->stages[0].output_shapes[i]);
    }
    ConvertNormalizedParameters(network_meta_->normalize.mean, network_meta_->normalize.norm);

    return 1;
}

int32_t BmrunHelper::ConvertNormalizedParameters(float* mean, float* norm) {
    // float scale = network_meta_->input_scale;
    float scale = 1.0;
    if (!USE_BMCV || !network_meta_->input_nchw) {
        // Convert to speede up normalization:  
        // (((src/255) - mean)/norm)*scale ----> ((src - mean*255) / (255*norm))*scale ----> (src - mean*255) * (scale/(255*norm))
        for (int32_t i = 0; i < 3; i++) {
            mean[i] *= 255;
            norm[i] *= 255;
            norm[i] = scale / norm[i];
        }
    } else {
        // Convert to match linear transformation:
        // (((src/255) - mean)/norm)*scale ----> src*scale/(255*norm) - mean*scale/norm ----> src*(scale/(255*norm)) - mean*scale/norm
        mean[0] = 0 - mean[0] * scale / norm[0];
        mean[1] = 0 - mean[1] * scale / norm[1];
        mean[2] = 0 - mean[2] * scale / norm[2];
        norm[0] = scale / (norm[0] * 255.0);
        norm[1] = scale / (norm[1] * 255.0);
        norm[2] = scale / (norm[2] * 255.0);
    }
    return 1;
}

template <typename T>
int32_t BmrunHelper::PermuateAndNormalize(T* input_ptr, uint8_t* src, const int32_t input_h, const int32_t input_w, const int32_t input_c) {
    // Convert NHWC to NCHW && Do normalized operation to the original input image.
    int32_t batch_size = network_meta_->batch_size;
    float* mean = network_meta_->normalize.mean;
    float* norm = network_meta_->normalize.norm;
    memset(input_ptr, 0, sizeof(T) * input_h * input_w * input_c * batch_size);
    int32_t spatial_size = input_h * input_w;
    if (network_meta_->input_nchw) {
#pragma omp parallel for num_threads(4)
        for (int32_t c = 0; c < input_c; c++) {
            for (int32_t i = 0; i < spatial_size; i++) {
                input_ptr[spatial_size * c + i] = (src[i * input_c + c] - mean[c]) * norm[c];
                //  input_ptr[spatial_size * c + i] = src[i * input_c + c] - 128;
            }
        }
    } else {
#pragma omp parallel for num_threads(4)
        for (int32_t c = 0; c < input_c; c++) {
            for (int32_t i = 0; i < spatial_size; i++) {
                input_ptr[i * input_c + c] = (src[i * input_c + c] - mean[c]) * norm[c];
            }
        }        
    }
    return 1;
}

int32_t BmrunHelper::SetCropAttr(const int32_t src_w, int32_t src_h, const int32_t dst_w, const int32_t dst_h, const kCropStyle style) {
    if (style == kCropStyle::CropAll_Coverup) {    // retain
        src_crop_.x = 0;
        src_crop_.y = 0;
        src_crop_.width = src_w;
        src_crop_.height = src_h;
        dst_crop_.x = 0;
        dst_crop_.y = 0;
        dst_crop_.width = dst_w;
        dst_crop_.height = dst_h;
        return 1;
    }
    if (style == kCropStyle::CropLower_Coverup_1) {    // a very distinct one, shed 0.4 top part of src, disgard of ratio
        src_crop_.x = 0;
        src_crop_.y = src_h * 0.4;
        src_crop_.width = src_w;
        src_crop_.height = src_h - src_crop_.y;
        dst_crop_.x = 0;
        dst_crop_.y = 0;
        dst_crop_.width = dst_w;
        dst_crop_.height = dst_h;
        return 1;
    }
    if (style == kCropStyle::CropLower_Coverup_0) {    // shed top part of src, retain width and make the crop ratio equals to model's input's
        float src_ratio = 1.0 * src_h / src_w;
        float dst_ratio = 1.0 * dst_h / dst_w;
        if (src_ratio > dst_ratio) {
            src_crop_.width = src_w;
            src_crop_.height = static_cast<int32_t>(src_w * dst_ratio);
            src_crop_.x = 0;
            src_crop_.y = src_h - src_crop_.height;
        } else {
            src_crop_.width = src_w;
            src_crop_.height = src_h;
            src_crop_.x = 0;
            src_crop_.y = 0;
        }
        dst_crop_.x = 0;
        dst_crop_.y = 0;
        dst_crop_.width = dst_w;
        dst_crop_.height = dst_h;
        return 1;
    }
    if (style == kCropStyle::CropAll_Embedd) {    // embedd src into dst's center, src's ratio not changed
        src_crop_.x = 0;
        src_crop_.y = 0;
        src_crop_.width = src_w;
        src_crop_.height = src_h;
        float src_ratio = 1.0 * src_w / src_h;
        float dst_ratio = 1.0 * dst_w / dst_h;
        if (src_ratio > dst_ratio) {
            // Use dst's width as base
            dst_crop_.width = dst_w;
            // dst_crop_.height = dst_h * dst_ratio / src_ratio;
            // dst_crop_.height = src_h * (dst_w / src_w);
            dst_crop_.height = static_cast<int32_t>(dst_w / src_ratio);
            dst_crop_.x = 0;
            dst_crop_.y = (dst_h - dst_crop_.height) / 2;
        } else {
            // Use dst's height as base
            dst_crop_.height = dst_h;
            dst_crop_.width = static_cast<int32_t>(dst_h / src_ratio);
            dst_crop_.x = (dst_w - dst_crop_.width) / 2;
            dst_crop_.y = 0;
        }
    }
    return 1;
}

int32_t BmrunHelper::PreProcess(cv::Mat& original_img, const kCropStyle& style) {
    auto tensor_meta = network_meta_->input_tensor_meta_list[0];
    int32_t input_h = tensor_meta.net_in_h;
    int32_t input_w = tensor_meta.net_in_w;
    int32_t input_c = tensor_meta.net_in_c;
    SetCropAttr(original_img.cols, original_img.rows, tensor_meta.net_in_w, tensor_meta.net_in_h, style);
    if (!USE_BMCV) {
        const auto& t0 = std::chrono::steady_clock::now();
        cv::Mat sample = cv::Mat::zeros(input_h, input_w, CV_8UC3);
        cv::Mat resized_mat = sample(dst_crop_);
        cv::resize(original_img(src_crop_), resized_mat, resized_mat.size(), 0, 0, cv::INTER_NEAREST); // Why must assign fx and fy to enable deep copy?
        // cv::imwrite("./resource/output.jpg", sample);
        if (network_meta_->input_rgb) {
            cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);
        }
        uint8_t* src = (uint8_t*)sample.data;
        const auto& t1 = std::chrono::steady_clock::now();
        PermuateAndNormalize((float*)input_ptrs_[0], src, input_h, input_w, input_c);
        const auto& t2 = std::chrono::steady_clock::now();
        std::cout << "---" << 1.0 * (t1 - t0).count() * 1e-6 << std::endl;
        std::cout << "---" << 1.0 * (t2 - t1).count() * 1e-6 << std::endl;
        return 1;
    }
    else {
        const auto& t0 = std::chrono::steady_clock::now();
        bm_image original_bm_img;
        cv::bmcv::toBMI(original_img, &original_bm_img);

        // 1. Do crop and resize & convert color channel and permute
        if (style == kCropStyle::CropAll_Coverup || style == kCropStyle::CropLower_Coverup_1 || style == kCropStyle::CropLower_Coverup_0) {
            bmcv_rect src_crop;
            src_crop.start_x = src_crop_.x;
            src_crop.start_y = src_crop_.y;
            src_crop.crop_w = src_crop_.width;
            src_crop.crop_h = src_crop_.height;
            bmcv_image_vpp_convert(bm_handle_, 1, original_bm_img, &bm_mat_resized_, &src_crop);
        } 
        if (style == kCropStyle::CropAll_Embedd) {
            bmcv_rect src_crop;
            src_crop.start_x = src_crop_.x;
            src_crop.start_y = src_crop_.y;
            src_crop.crop_w = src_crop_.width;
            src_crop.crop_h = src_crop_.height;
            bmcv_padding_atrr_s resize_padding_attr;
            resize_padding_attr.dst_crop_stx = dst_crop_.x;
            resize_padding_attr.dst_crop_sty = dst_crop_.y;
            resize_padding_attr.dst_crop_w = dst_crop_.width;
            resize_padding_attr.dst_crop_h = dst_crop_.height;
            resize_padding_attr.padding_b = 0;
            resize_padding_attr.padding_g = 0;
            resize_padding_attr.padding_r = 0;
            resize_padding_attr.if_memset = 1;
            bmcv_image_vpp_convert_padding(bm_handle_, 1, original_bm_img, &bm_mat_resized_, &resize_padding_attr, &src_crop);
        }
        // cv::Mat cv_mat_resized;
        // cv::bmcv::toMAT(&bm_mat_resized_, cv_mat_resized);
        // cv::imwrite("./resource/output.jpg", cv_mat_resized);

        // 2. Do normalization
        // Currently bmcv doesn't support normalize to nhwc format, so use cpu to normalize instead
        if (!network_meta_->input_nchw) {
            const auto& t1 = std::chrono::steady_clock::now();
            cv::Mat cv_mat_resized;
            cv::bmcv::toMAT(&bm_mat_resized_, cv_mat_resized);
            uint8_t* src = (uint8_t*)cv_mat_resized.data;
            PermuateAndNormalize((float*)input_ptrs_[0], src, input_h, input_w, input_c); // Might cause bug
            const auto& t2 = std::chrono::steady_clock::now();
            std::cout << "---" << 1.0 * (t1 - t0).count() * 1e-6 << std::endl;
            std::cout << "---" << 1.0 * (t2 - t1).count() * 1e-6 << std::endl;
        } else {
            const auto& t1 = std::chrono::steady_clock::now();
            bmcv_convert_to_attr normalize_attr;
            float* mean = network_meta_->normalize.mean;
            float* norm = network_meta_->normalize.norm;
            normalize_attr.alpha_0 = norm[0];
            normalize_attr.alpha_1 = norm[1];
            normalize_attr.alpha_2 = norm[2];
            normalize_attr.beta_0 = mean[0];
            normalize_attr.beta_1 = mean[1];
            normalize_attr.beta_2 = mean[2];
            bmcv_image_convert_to(bm_handle_, 1, normalize_attr, &bm_mat_resized_, &bm_mat_normalized_);
            const auto& t2 = std::chrono::steady_clock::now();
            std::cout << "---" << 1.0 * (t1 - t0).count() * 1e-6 << std::endl;
            std::cout << "---" << 1.0 * (t2 - t1).count() * 1e-6 << std::endl;
        }
    }
    
    return 1;
}

int32_t BmrunHelper::Inference() {
    // Prepare input data
    for (int32_t i = 0; i < input_tensors_.size(); i++) {
        if (!USE_BMCV || !network_meta_->input_nchw) {
            // Copy the input from kernel_ram to gpu_ram.
            bm_memcpy_s2d(bm_handle_, input_tensors_[i].device_mem, ((void *)input_ptrs_[i]));
        } else {
            // Input data is already in gpu_ram
            bm_device_mem_t input_dev_mem;
            bm_image_get_device_mem(bm_mat_normalized_, &input_dev_mem);
            input_tensors_[i].device_mem = input_dev_mem;
        }
    }

    // Command the gpu to excute inference.
    bool ret = bmrt_launch_tensor_ex(p_bmrt_, net_names_[0], &input_tensors_[0], network_meta_->input_tensor_num, &output_tensors_[0], network_meta_->output_tensor_num, true, false);
    // Wait for gpu's infenernce finished.
    bm_thread_sync(bm_handle_);

    // Retrieve the output from gpu_ram to kernel_ram.
    for (int32_t i = 0; i < output_tensors_.size(); i++) {
        size_t size = bmrt_tensor_bytesize(&output_tensors_[i]);
        bm_memcpy_d2s_partial(bm_handle_, output_ptrs_[i], output_tensors_[i].device_mem, size);
    }
    return 1;
}

int32_t BmrunHelper::Finalize() {
    // if (!USE_BMCV) {
    //     if (network_meta_->tensor_type = NetworkMeta::kTensorTypeFloat32) {
    //         delete[] (float*)input_ptr_;
    //         delete[] (float*)output_ptr_;
    //     } else if (network_meta_->tensor_type = NetworkMeta::kTensorTypeInt8) {
    //         delete[] (int8_t*)input_ptr_;
    //         delete[] (float*)output_ptr_;
    //     }
    // } else if (!network_meta_->input_nchw) {
    //     if (network_meta_->tensor_type = NetworkMeta::kTensorTypeFloat32) {
    //         delete[] (float*)input_ptr_;
    //         delete[] (float*)output_ptr_;
    //         bm_image_destroy(bm_mat_resized_);
    //     } else if (network_meta_->tensor_type = NetworkMeta::kTensorTypeInt8) {
    //         delete[] (int8_t*)input_ptr_;
    //         delete[] (float*)output_ptr_;
    //         bm_image_destroy(bm_mat_resized_);
    //     }
    // } else {
    //     bm_image_destroy(bm_mat_resized_);
    //     bm_image_destroy(bm_mat_normalized_);
    //     delete[] (float*)output_ptr_;
    // }
    return 1;
}