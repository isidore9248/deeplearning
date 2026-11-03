#include "Headfile.h"

///yolov8 推理封装库
#include "yolov8_onnx.h"


const std::filesystem::path MODEL_PATH = std::filesystem::path(__FILE__).parent_path().parent_path() / "models/yolov8n.onnx";
const std::filesystem::path INPUTTENSOR_PATH = std::filesystem::path(__FILE__).parent_path().parent_path() / "models/inputtest.jpg";


int main() {
	try {

		yolo_ONNX yolo("yolov8_inference");
		yolo.ONNX_add_model(MODEL_PATH);
		yolo.print_model_info();
		cv::Mat input_blob = yolo.Mat_preprocess(1, 3, 640, 640, INPUTTENSOR_PATH);
		yolo.convert_to_tensor(input_blob);
		yolo.run_inference();

		///验证输出
		//const std::vector<Ort::Value>& output_tensors = yolo.get_output_tensors();

		const float confidence_threshold = 0.25f;
		const float nms_threshold = 0.45f;
		yolo.Mat_postprocess(confidence_threshold, nms_threshold);


		cv::waitKey(0);
		return 0;
	}
	catch (const Ort::Exception& error) {
		std::cerr << "ONNX Runtime error: " << error.what() << '\n';
		return 1;
	}
	catch (const cv::Exception& error) {
		std::cerr << "OpenCV error: " << error.what() << '\n';
		return 1;
	}
	catch (const std::exception& error) {
		std::cerr << "Error: " << error.what() << '\n';
		return 1;
	}
}

