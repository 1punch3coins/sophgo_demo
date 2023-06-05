#ifndef BMRUN_HELPER_
#define BMRUN_HELPER_
#include <vector>
#include <memory>
#include "bmcv_api.h"
#include "bmcv_api_ext.h"
#include "bmruntime_interface.h"

enum {
    kTaskTypeDet,
    kTaskTypeSeg,
    kTaskTypeRoadSeg,
};

class TensorInfo {
public:
    enum {
        kTensorTypeInt8,
        kTensorTypeFloat16,
        kTensorTypeFloat32,
    };

public:
    float input_scale;
    float output_scale;
    int32_t net_in_c;
    int32_t net_in_h;
    int32_t net_in_w;
    int32_t net_out_c;
    int32_t net_out_l;
    int32_t net_out_h;
    int32_t net_out_w;
    int32_t batch_size;

public:
    int32_t tensor_type;
    bool input_nchw, input_rgb;
    bool output_nlc;
    struct {
        float mean[3];
        float norm[3];
    } normalize;

public:
    TensorInfo(): 
        tensor_type(kTensorTypeFloat32),
        input_nchw(true),
        input_rgb(true),
        output_nlc(true)
    {}
    TensorInfo(int32_t tensor_type_, bool input_nchw_, bool input_rgb_, bool output_nlc_):
        tensor_type(tensor_type_),
        input_nchw(input_nchw_),
        input_rgb(input_rgb_),
        output_nlc(output_nlc_)
    {}
};

class BmrunHelper {
public:
    static BmrunHelper* Create(const std::string& model, int32_t task_id, TensorInfo* t_info);
    int32_t Initialize();
    int32_t PreProcess(cv::Mat& original_img);
    int32_t Inference();
    int32_t Finalize();

public:
    inline float* GetInfernceOutput() const;
    inline int8_t* GetInfernceOutput2() const;
    inline int32_t GetInputHeight() const;
    inline int32_t GetInputWidth() const;
    inline int32_t GetOutputLength() const;
    inline int32_t GetOutputHeight() const;
    inline int32_t GetOutputWidth() const;
    inline int32_t GetOutputChannelNum() const;
    inline std::pair<cv::Rect, cv::Rect> GetCropInfo() const;

private:
    template <typename T>
    int32_t PermuateAndNormalize(T* intput_ptr, uint8_t* src);
    int32_t SetCropAttr(const int32_t src_w, int32_t src_h, const int32_t dst_w, const int32_t dst_h);
    int32_t ConvertNormalizedParameters(float* mean, float* norm);

private:
    int32_t dev_id_;
    void* p_bmrt_;
    const char* model_pwd_;
    const char **net_names_;
    bm_handle_t bm_handle_;
    bm_tensor_t input_tensor_; 
    bm_tensor_t output_tensor_;

private:
    int32_t task_;
    std::unique_ptr<TensorInfo> tensor_info_;
    cv::Rect src_crop_;
    cv::Rect dst_crop_;

private:
    bm_image bm_mat_resized_;
    bm_image bm_mat_normalized_;
    void* input_ptr_;
    void* output_ptr_;
};

inline float* BmrunHelper::GetInfernceOutput() const{
    return (float*)output_ptr_;
}

inline int8_t* BmrunHelper::GetInfernceOutput2() const{
    return (int8_t*)output_ptr_;
}

inline int32_t BmrunHelper::GetInputHeight() const{
    return tensor_info_->net_in_h;
}

inline int32_t BmrunHelper::GetInputWidth() const{
    return tensor_info_->net_in_w;
}

inline int32_t BmrunHelper::GetOutputLength() const {
    return tensor_info_ ->net_out_l;
}

inline int32_t BmrunHelper::GetOutputHeight() const {
    return tensor_info_ ->net_out_h;
}

inline int32_t BmrunHelper::GetOutputWidth() const {
    return tensor_info_ ->net_out_w;
}

inline int32_t BmrunHelper::GetOutputChannelNum() const {
    return tensor_info_ ->net_out_c;
}

inline std::pair<cv::Rect, cv::Rect> BmrunHelper::GetCropInfo() const {
    return std::pair<cv::Rect, cv::Rect>(src_crop_, dst_crop_);
}

#endif