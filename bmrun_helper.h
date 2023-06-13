#ifndef BMRUN_HELPER_
#define BMRUN_HELPER_
#include <vector>
#include <map>
#include <memory>

#include "bmcv_api.h"
#include "bmcv_api_ext.h"
#include "bmruntime_interface.h"

enum {
    kTaskTypeDet,
    kTaskTypeSeg,
    kTaskTypeLaneDet,
    kTaskTypeRoadSeg,
};

class NetworkMeta {
public:
    enum {
        kTensorTypeInt8,
        kTensorTypeFloat16,
        kTensorTypeFloat32,
    };
    struct InputTensorMeta{
        // const char* tensor_name;
        std::string tensor_name;
        float input_scale;
        int32_t net_in_c;
        int32_t net_in_h;
        int32_t net_in_w;
        int32_t net_in_elements;
        InputTensorMeta(const std::string& tensor_name_):
            tensor_name(tensor_name_)
        {}
    };
    struct OutputTensorMeta{
        // const char* tensor_name;
        std::string tensor_name;
        float output_scale;
        int32_t net_out_c;
        int32_t net_out_l;
        int32_t net_out_h;
        int32_t net_out_w;
        int32_t net_out_elements;
        OutputTensorMeta(const std::string& tensor_name_):
            tensor_name(tensor_name_)
        {}
    };
public:
    int32_t input_tensor_num;
    int32_t output_tensor_num;
    std::vector<InputTensorMeta> input_tensor_meta_list;
    std::vector<OutputTensorMeta> output_tensor_meta_list;
    std::map<std::string, int32_t> input_name2index;
    std::map<std::string, int32_t> output_name2index;

public:
    int32_t tensor_type;
    int32_t batch_size;
    bool input_nchw, input_rgb;
    bool output_nlc;
    struct {
        float mean[3];
        float norm[3];
    } normalize;

public:
    NetworkMeta(): 
        tensor_type(kTensorTypeFloat32),
        input_nchw(true),
        input_rgb(true),
        output_nlc(true),
        input_tensor_num(1),
        output_tensor_num(1)
    {}
    NetworkMeta(int32_t tensor_type_, bool input_nchw_, bool input_rgb_, bool output_nlc_):
        tensor_type(tensor_type_),
        input_nchw(input_nchw_),
        input_rgb(input_rgb_),
        output_nlc(output_nlc_),
        input_tensor_num(1),
        output_tensor_num(1)
    {}
    NetworkMeta(int32_t tensor_type_, bool input_nchw_, bool input_rgb_, bool output_nlc_, int32_t input_tensor_num_, int32_t output_tensor_num_):
        tensor_type(tensor_type_),
        input_nchw(input_nchw_),
        input_rgb(input_rgb_),
        output_nlc(output_nlc_),
        input_tensor_num(input_tensor_num_),
        output_tensor_num(output_tensor_num_)
    {}
    void AddInputTensorMeta (const std::string& input_name) {
        input_tensor_meta_list.push_back(InputTensorMeta(input_name));
    }
    void AddOutputTensorMeta (const std::string& output_name) {
        output_tensor_meta_list.push_back(OutputTensorMeta(output_name));
    }
};

class BmrunHelper {
public:
    static BmrunHelper* Create(const std::string& model, int32_t task_id, NetworkMeta* t_info);
    int32_t Initialize();
    int32_t PreProcess(cv::Mat& original_img);
    int32_t PreProcess(cv::Mat& original_img, const std::string& input_name);
    int32_t Inference();
    int32_t Finalize();

public:
    float* GetInfernceOutput(const std::string& tensor_name) const {
        int32_t index = network_meta_->output_name2index.find(tensor_name)->second;
        return (float*)output_ptrs_[index];
    }
    int8_t* GetInfernceOutput2(const std::string& tensor_name) const {
        int32_t index = network_meta_->output_name2index.find(tensor_name)->second;
        return (int8_t*)output_ptrs_[index];
    }
    int32_t GetInputHeight(const std::string& tensor_name) const {
        int32_t index = network_meta_->input_name2index.find(tensor_name)->second;
        return network_meta_->input_tensor_meta_list[index].net_in_h;
    }
    int32_t GetInputWidth(const std::string& tensor_name) const {
        int32_t index = network_meta_->input_name2index.find(tensor_name)->second;
        return network_meta_->input_tensor_meta_list[index].net_in_w;
    }
    int32_t GetOutputLength(const std::string& tensor_name) const {
        int32_t index = network_meta_->output_name2index.find(tensor_name)->second;
        return network_meta_->output_tensor_meta_list[index].net_out_l;
    }
    int32_t GetOutputHeight(const std::string& tensor_name) const {
        int32_t index = network_meta_->output_name2index.find(tensor_name)->second;
        return network_meta_->output_tensor_meta_list[index].net_out_h;
    }
    int32_t GetOutputWidth(const std::string& tensor_name) const {
        int32_t index = network_meta_->output_name2index.find(tensor_name)->second;
        return network_meta_->output_tensor_meta_list[index].net_out_w;
    }
    int32_t GetOutputChannelNum(const std::string& tensor_name) const {
        int32_t index = network_meta_->output_name2index.find(tensor_name)->second;
        return network_meta_->output_tensor_meta_list[index].net_out_c;
    }
    std::pair<cv::Rect, cv::Rect> GetCropInfo() const {
        return std::pair<cv::Rect, cv::Rect>(src_crop_, dst_crop_);
    }

public:
    float* GetInfernceOutput() const {
        return (float*)output_ptrs_[0];
    }
    int32_t GetInputHeight() const {
        return network_meta_->input_tensor_meta_list[0].net_in_h;
    }
    int32_t GetInputWidth() const {
        return network_meta_->input_tensor_meta_list[0].net_in_w;
    }
    int32_t GetOutputLength() const {
        return network_meta_->output_tensor_meta_list[0].net_out_l;
    }
    int32_t GetOutputHeight() const {
        return network_meta_->output_tensor_meta_list[0].net_out_h;
    }
    int32_t GetOutputWidth() const {
        return network_meta_->output_tensor_meta_list[0].net_out_w;
    }
    int32_t GetOutputChannelNum() const {
        return network_meta_->output_tensor_meta_list[0].net_out_c;
    }

private:
    template <typename T>
    int32_t PermuateAndNormalize(T* input_ptr, uint8_t* src, const int32_t input_h, const int32_t input_w, const int32_t input_c);
    int32_t SetCropAttr(const int32_t src_w, int32_t src_h, const int32_t dst_w, const int32_t dst_h);
    int32_t ConvertNormalizedParameters(float* mean, float* norm);

private:
    int32_t dev_id_;
    void* p_bmrt_;
    const char* model_pwd_;
    const char **net_names_;
    bm_handle_t bm_handle_;

private:
    int32_t task_;
    std::unique_ptr<NetworkMeta> network_meta_;
    cv::Rect src_crop_;
    cv::Rect dst_crop_;

private:
    bm_image bm_mat_resized_;
    bm_image bm_mat_normalized_;
    std::vector<void*> input_ptrs_;
    std::vector<void*> output_ptrs_;
    std::vector<bm_tensor_t> input_tensors_; 
    std::vector<bm_tensor_t> output_tensors_;
};

#endif