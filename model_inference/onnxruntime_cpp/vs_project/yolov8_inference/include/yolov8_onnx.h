#pragma once


#include "Headfile.h"

// 如果传入的条件不成立，就抛出异常。
#define THROW_IF_NOT(condition)                                  \
  do {                                                           \
    if (!(condition)) {                                          \
      throw std::runtime_error(std::string(__FILE__) + ":" +     \
                               std::to_string(__LINE__) + ": " + \
                               "check failed: " #condition);     \
    }                                                            \
  } while (0)


class yolo_ONNX
{
public:
	yolo_ONNX(std::string pro_name);
	void ONNX_add_model(const std::filesystem::path MODEL_PATH);
	void print_model_info();
	cv::Mat Mat_preprocess(int64_t batch_size, int64_t channels, int64_t input_height, int64_t input_weight,
		const std::filesystem::path input_tensor_path);
	void convert_to_tensor(cv::Mat input_blob);
	void run_inference();

	const std::vector<Ort::Value>& get_output_tensors() const;

	void Mat_postprocess(const float confidence_threshold, const float nms_threshold);
private:
	void sys_init();
	void ONNX_init();



private:
	Ort::Env env_;
	Ort::SessionOptions session_options_;
	Ort::Session session_;

	cv::Mat image_raw_;
	float scale_;
	int left_, right_;
	int top_, bottom_;

	std::array<int64_t, 4> input_shape;

	std::vector<std::string> input_names;
	std::vector<std::string> output_names;

	std::vector<const char*> input_name_ptrs;
	std::vector<const char*> output_name_ptrs;

	std::array<Ort::Value, 1> input_tensors;
	std::vector<Ort::Value> output_tensors_;
};