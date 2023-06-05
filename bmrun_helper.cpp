#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "bmruntime_interface.h"
// #include "bmdef.h"
#include "bmrun_helper.h"

#define USE_BMCV true

BmrunHelper* BmrunHelper::Create(const std::string& model, int32_t task_id, TensorInfo* p_info) {
    BmrunHelper*p = new BmrunHelper();
    p->model_pwd_ = model.c_str();
    p->task_ = task_id;
    p->tensor_info_.reset(p_info);
    return p;
}

int32_t BmrunHelper::Initialize() {
    if (bm_dev_request(&bm_handle_, 0) != BM_SUCCESS)
        return 0;
    p_bmrt_ = bmrt_create(bm_handle_);
    if (p_bmrt_ == NULL)
        return 0;
    if (bmrt_load_bmodel(p_bmrt_, model_pwd_) != true)
        return 0;
    bmrt_get_network_names(p_bmrt_, &net_names_);
    auto net_info = bmrt_get_network_info(p_bmrt_, net_names_[0]);

    bm_shape_t input_shape = net_info->stages[0].input_shapes[0];
    if (tensor_info_->input_nchw) {
        tensor_info_->net_in_c = input_shape.dims[1];
        tensor_info_->net_in_h = input_shape.dims[2];
        tensor_info_->net_in_w = input_shape.dims[3];
    } else {
        tensor_info_->net_in_h = input_shape.dims[1];
        tensor_info_->net_in_w = input_shape.dims[2];
        tensor_info_->net_in_c = input_shape.dims[3];
    }
    tensor_info_->batch_size = input_shape.dims[0];
    tensor_info_->input_scale = net_info->input_scales[0];
    tensor_info_->output_scale = net_info->output_scales[0];

    bm_shape_t output_shape = net_info->stages[0].output_shapes[0];
    if (output_shape.num_dims == 3) {
        if (tensor_info_->output_nlc) {
            tensor_info_->net_out_l = output_shape.dims[1];
            tensor_info_->net_out_c = output_shape.dims[2];
        } else {
            tensor_info_->net_out_c = output_shape.dims[1];
            tensor_info_->net_out_l = output_shape.dims[2];
        }
    }
    if (output_shape.num_dims == 4) {
        if (tensor_info_->output_nlc) {
            tensor_info_->net_out_h = output_shape.dims[1];
            tensor_info_->net_out_w = output_shape.dims[2];
            tensor_info_->net_out_c = output_shape.dims[3];
            tensor_info_->net_out_l = output_shape.dims[1] * output_shape.dims[2];
        } else {
            tensor_info_->net_out_c = output_shape.dims[1];
            tensor_info_->net_out_h = output_shape.dims[2];
            tensor_info_->net_out_w = output_shape.dims[3];
            tensor_info_->net_out_l = output_shape.dims[2] * output_shape.dims[3];
        }
    }
    
    if (!USE_BMCV) {
        if (net_info->input_dtypes[0] == BM_FLOAT32) {
            tensor_info_->tensor_type = TensorInfo::kTensorTypeFloat32;
            input_ptr_ = new float[tensor_info_->batch_size * tensor_info_->net_in_c * tensor_info_->net_in_h * tensor_info_->net_in_w];
            output_ptr_ = new float[tensor_info_->batch_size * tensor_info_->net_out_c * tensor_info_->net_out_l];
            input_ptr_ = (float*)input_ptr_;
            output_ptr_ = (float*)output_ptr_;
        } else if (net_info->input_dtypes[0] == BM_INT8) {
            tensor_info_->tensor_type = TensorInfo::kTensorTypeInt8;
            input_ptr_ = new int8_t[tensor_info_->batch_size * tensor_info_->net_in_c * tensor_info_->net_in_h * tensor_info_->net_in_w];
            output_ptr_ = new float[tensor_info_->batch_size * tensor_info_->net_out_c * tensor_info_->net_out_l];
            input_ptr_ = (int8_t*)input_ptr_;
            output_ptr_ = (float*)output_ptr_;
        }
    } else {
        bm_image_format_ext_ bm_formate_type;
        // if (tensor_info_->input_nchw && tensor_info_->input_rgb) {
        //     bm_formate_type = FORMAT_RGB_PLANAR;
        // } 
        // if (!tensor_info_->input_nchw && tensor_info_->input_rgb) {
        //     bm_formate_type = FORMAT_RGB_PACKED;
        // }
        // if (tensor_info_->input_nchw && !tensor_info_->input_rgb) {
        //     bm_formate_type = FORMAT_BGR_PLANAR;
        // }
        // if (!tensor_info_->input_nchw && !tensor_info_->input_rgb) {
        //     bm_formate_type = FORMAT_BGR_PACKED;
        // }
        if (tensor_info_->input_rgb) {
            bm_formate_type = FORMAT_RGB_PLANAR;
        } else {
            bm_formate_type = FORMAT_BGR_PLANAR;
        }
        if (net_info->input_dtypes[0] == BM_FLOAT32) {
            tensor_info_->tensor_type = TensorInfo::kTensorTypeFloat32;
            if (!tensor_info_->input_nchw) {
                bm_image_create(bm_handle_, tensor_info_->net_in_h, tensor_info_->net_in_w, bm_formate_type, DATA_TYPE_EXT_1N_BYTE, &bm_mat_resized_);
                input_ptr_ = new float[tensor_info_->batch_size * tensor_info_->net_in_c * tensor_info_->net_in_h * tensor_info_->net_in_w];
                output_ptr_ = new float[tensor_info_->batch_size * tensor_info_->net_out_c * tensor_info_->net_out_l];
                input_ptr_ = (float*)input_ptr_;
                output_ptr_ = (float*)output_ptr_;
            } else {
                bm_image_create(bm_handle_, tensor_info_->net_in_h, tensor_info_->net_in_w, bm_formate_type, DATA_TYPE_EXT_1N_BYTE, &bm_mat_resized_);
                bm_image_create(bm_handle_, tensor_info_->net_in_h, tensor_info_->net_in_w, bm_formate_type, DATA_TYPE_EXT_FLOAT32, &bm_mat_normalized_);
                output_ptr_ = new float[tensor_info_->batch_size * tensor_info_->net_out_c * tensor_info_->net_out_l];
            }
        } else if (net_info->input_dtypes[0] == BM_INT8) {
            tensor_info_->tensor_type = TensorInfo::kTensorTypeInt8;
            if (!tensor_info_->input_nchw) {
                bm_image_create(bm_handle_, tensor_info_->net_in_h, tensor_info_->net_in_w, bm_formate_type, DATA_TYPE_EXT_1N_BYTE, &bm_mat_resized_);
                input_ptr_ = new int8_t[tensor_info_->batch_size * tensor_info_->net_in_c * tensor_info_->net_in_h * tensor_info_->net_in_w];
                output_ptr_ = new float[tensor_info_->batch_size * tensor_info_->net_out_c * tensor_info_->net_out_l];
                input_ptr_ = (int8_t*)input_ptr_;
                output_ptr_ = (float*)output_ptr_;
            } else {
                bm_image_create(bm_handle_, tensor_info_->net_in_h, tensor_info_->net_in_w, bm_formate_type, DATA_TYPE_EXT_1N_BYTE, &bm_mat_resized_);
                bm_image_create(bm_handle_, tensor_info_->net_in_h, tensor_info_->net_in_w, bm_formate_type, DATA_TYPE_EXT_1N_BYTE_SIGNED, &bm_mat_normalized_);
                output_ptr_ = new float[tensor_info_->batch_size * tensor_info_->net_out_c * tensor_info_->net_out_l];
            }
        }
    }

    bmrt_tensor(&input_tensor_, p_bmrt_, net_info->input_dtypes[0], input_shape);
    bmrt_tensor(&output_tensor_, p_bmrt_, net_info->output_dtypes[0], output_shape);
    ConvertNormalizedParameters(tensor_info_->normalize.mean, tensor_info_->normalize.norm);

    return 1;
}

int32_t BmrunHelper::ConvertNormalizedParameters(float* mean, float* norm) {
    float scale = tensor_info_->input_scale;
    if (!USE_BMCV || !tensor_info_->input_nchw) {
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
        norm[0] = scale / (norm[0] * 255.0);
        norm[1] = scale / (norm[1] * 255.0);
        norm[2] = scale / (norm[2] * 255.0);
        mean[0] = 0 - mean[0] * scale / norm[0];
        mean[1] = 0 - mean[1] * scale / norm[1];
        mean[2] = 0 - mean[2] * scale / norm[2];
    }
    return 1;
}

template <typename T>
int32_t BmrunHelper::PermuateAndNormalize(T* input_ptr, uint8_t* src) {
    // Convert NHWC to NCHW && Do normalized operation to the original input image.
    int32_t batch_size = tensor_info_->batch_size;
    int32_t input_h = tensor_info_->net_in_h;
    int32_t input_w = tensor_info_->net_in_w;
    int32_t input_c = tensor_info_->net_in_c;
    float input_scale = tensor_info_->input_scale;
    float* mean = tensor_info_->normalize.mean;
    float* norm = tensor_info_->normalize.norm;
    memset(input_ptr, 0, sizeof(T) * input_h * input_w * input_c * batch_size);
    int32_t spatial_size = input_h * input_w;
    if (tensor_info_->input_nchw) {
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

int32_t BmrunHelper::SetCropAttr(const int32_t src_w, int32_t src_h, const int32_t dst_w, const int32_t dst_h) {
    if (task_ == kTaskTypeSeg) {
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
    if (task_ == kTaskTypeRoadSeg) {
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
        dst_crop_.width = dst_w;
        dst_crop_.height = dst_h;
        dst_crop_.x = 0;
        dst_crop_.y = 0;
        return 1;
    }
    if (task_ == kTaskTypeDet) {
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

int32_t BmrunHelper::PreProcess(cv::Mat& original_img)
{
    SetCropAttr(original_img.cols, original_img.rows, tensor_info_->net_in_w, tensor_info_->net_in_h);
    if (!USE_BMCV) {
        const auto& t0 = std::chrono::steady_clock::now();
        int32_t input_h = tensor_info_->net_in_h;
        int32_t input_w = tensor_info_->net_in_w;
        cv::Mat sample = cv::Mat::zeros(input_w, input_h, CV_8UC3);
        cv::Mat resized_mat = sample(dst_crop_);
        cv::resize(original_img, resized_mat, resized_mat.size(), 0, 0, cv::INTER_NEAREST); // Why must assign fx and fy to enable deep copy?
        if (tensor_info_->input_rgb) {
            cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);
        }
        uint8_t* src = (uint8_t*)sample.data;
        const auto& t1 = std::chrono::steady_clock::now();
        PermuateAndNormalize((float*)input_ptr_, src);
        const auto& t2 = std::chrono::steady_clock::now();
        std::cout << "---" << 1.0 * (t1 - t0).count() * 1e-6 << std::endl;
        std::cout << "---" << 1.0 * (t2 - t1).count() * 1e-6 << std::endl;
        return 1;
    }
    
    else {
        const auto& t0 = std::chrono::steady_clock::now();
        bm_image original_bm_img;
        cv::bmcv::toBMI(original_img, &original_bm_img);

        // 1. Do crop and resize
        if (task_ == kTaskTypeSeg || task_ == kTaskTypeRoadSeg) {
            bmcv_rect src_crop;
            src_crop.start_x = src_crop_.x;
            src_crop.start_y = src_crop_.y;
            src_crop.crop_w = src_crop_.width;
            src_crop.crop_h = src_crop_.height;
            bmcv_image_vpp_convert(bm_handle_, 1, original_bm_img, &bm_mat_resized_, &src_crop);
        } 
        if (task_ == kTaskTypeDet) {
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

        // 2. Do normalization
        // Currently bmcv doesn't support normalize to nhwc format, so use cpu to normalize instead
        if (!tensor_info_->input_nchw) {
            const auto& t1 = std::chrono::steady_clock::now();
            cv::Mat cv_mat_resized;
            cv::bmcv::toMAT(&bm_mat_resized_, cv_mat_resized);
            uint8_t* src = (uint8_t*)cv_mat_resized.data;
            PermuateAndNormalize((float*)input_ptr_, src);
            const auto& t2 = std::chrono::steady_clock::now();
            std::cout << "---" << 1.0 * (t1 - t0).count() * 1e-6 << std::endl;
            std::cout << "---" << 1.0 * (t2 - t1).count() * 1e-6 << std::endl;
        } else {
            const auto& t1 = std::chrono::steady_clock::now();
            bmcv_convert_to_attr normalize_attr;
            float* mean = tensor_info_->normalize.mean;
            float* norm = tensor_info_->normalize.norm;
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
    if (!USE_BMCV || !tensor_info_->input_nchw) {
        // Copy the input from kernel_ram to gpu_ram.
        bm_memcpy_s2d(bm_handle_, input_tensor_.device_mem, ((void *)input_ptr_));
    } else {
        // Input data is already in gpu_ram
        bm_device_mem_t input_dev_mem;
        bm_image_get_device_mem(bm_mat_normalized_, &input_dev_mem);
        input_tensor_.device_mem = input_dev_mem;
    }
    // Command the gpu to excute inference.
    bool ret = bmrt_launch_tensor_ex(p_bmrt_, net_names_[0], &input_tensor_, 1, &output_tensor_, 1, true, false);
    // Wait for gpu's infenernce finished.
    bm_thread_sync(bm_handle_);
    // Retrieve the output from gpu_ram to kernel_ram.
    size_t size = bmrt_tensor_bytesize(&output_tensor_);
    bm_memcpy_d2s_partial(bm_handle_, output_ptr_, output_tensor_.device_mem, size);
    return 1;
}

int32_t BmrunHelper::Finalize() {
    if (!USE_BMCV) {
        if (tensor_info_->tensor_type = TensorInfo::kTensorTypeFloat32) {
            delete[] (float*)input_ptr_;
            delete[] (float*)output_ptr_;
        } else if (tensor_info_->tensor_type = TensorInfo::kTensorTypeInt8) {
            delete[] (int8_t*)input_ptr_;
            delete[] (float*)output_ptr_;
        }
    } else if (!tensor_info_->input_nchw) {
        if (tensor_info_->tensor_type = TensorInfo::kTensorTypeFloat32) {
            delete[] (float*)input_ptr_;
            delete[] (float*)output_ptr_;
            bm_image_destroy(bm_mat_resized_);
        } else if (tensor_info_->tensor_type = TensorInfo::kTensorTypeInt8) {
            delete[] (int8_t*)input_ptr_;
            delete[] (float*)output_ptr_;
            bm_image_destroy(bm_mat_resized_);
        }
    } else {
        bm_image_destroy(bm_mat_resized_);
        bm_image_destroy(bm_mat_normalized_);
        delete[] (float*)output_ptr_;
    }
    return 1;
}