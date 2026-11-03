#include <yolov8_onnx.h>



yolo_ONNX::yolo_ONNX(std::string pro_name) :env_(ORT_LOGGING_LEVEL_WARNING, pro_name.c_str()),
session_options_(), session_(nullptr), scale_(0.0f), left_(0), right_(0), top_(0), bottom_(0),
image_raw_(),
input_shape(), input_names(), output_names(), input_tensors(),
input_name_ptrs(), output_name_ptrs(), output_tensors_()
{
	sys_init();
	ONNX_init();

}

void yolo_ONNX::sys_init()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);//设置OpenCV日志级别为警告

	std::cout << "Runtime version: " << OrtGetApiBase()->GetVersionString() << '\n';

	std::string filePath = __FILE__;
	std::filesystem::path p(filePath);
	std::filesystem::path absPath = std::filesystem::absolute(p);
	std::cout << "Current file path: " << absPath << std::endl;

	return;
}

void yolo_ONNX::ONNX_init()
{
	//1.初始化ONNX Runtime环境
	///已经在构造函数中初始化了 Ort::Env env_，所以这里不需要再次初始化。

	//2.配置 ONNX Runtime 推理会话的运行方式
	//配置线程数、图优化级别等
	Ort::SessionOptions session_options_init;
	//设置不同算子之间的并行线程数
	session_options_init.SetIntraOpNumThreads(1);
	//启用最高等级的安全计算图优化
	session_options_init.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options_ = std::move(session_options_init);
}

void yolo_ONNX::ONNX_add_model(const std::filesystem::path MODEL_PATH)
{
	//3.加载onnx模型
	const std::filesystem::path model_path = MODEL_PATH;
	Ort::Session session(env_, model_path.c_str(), session_options_);
	session_ = std::move(session);

}

void yolo_ONNX::print_model_info()
{
	//4、查询模型的输入输出信息
	const size_t num_inputs = session_.GetInputCount();
	const size_t num_outputs = session_.GetOutputCount();
	std::cout << "Number of inputs: " << num_inputs << std::endl;
	std::cout << "Number of outputs: " << num_outputs << std::endl;

	Ort::AllocatorWithDefaultOptions ret_allocator;
	for (size_t i = 0; i < num_inputs; ++i) {
		auto name = session_.GetInputNameAllocated(i, ret_allocator);
		std::cout << "  Input  " << i << ": " << name.get() << "\n";
		input_names.emplace_back(name.get());
	}
	for (size_t i = 0; i < num_outputs; ++i) {
		auto name = session_.GetOutputNameAllocated(i, ret_allocator);
		std::cout << "  Output " << i << ": " << name.get() << "\n";
		output_names.emplace_back(name.get());
	}

}

cv::Mat yolo_ONNX::Mat_preprocess(int64_t batch_size, int64_t channels, int64_t input_height, int64_t input_width,
	const std::filesystem::path input_tensor_path)
{
	//5、准备输入数据
	//yolo需要的输入数据是一个4维的张量，形状为[batch_size, channels, height, width]，
	input_shape = { batch_size, channels, input_height, input_width };

	cv::Mat image = cv::imread(cv::String(input_tensor_path.string()));
	image_raw_ = image.clone();
	//计算缩放比例，保持图像的纵横比
	scale_ = std::min(
		static_cast<float>(input_width) / image.cols,
		static_cast<float>(input_height) / image.rows
	);

	const int resized_width = static_cast<int>(std::round(image.cols * scale_));
	const int resized_height = static_cast<int>(std::round(image.rows * scale_));

	//等比例缩放图像到指定大小
	cv::Mat resized_img;
	cv::resize(image, resized_img, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);


	///图像经过等比例缩放之后，不一定会满足模型要求的输入比例
	/// letterbox可以将剩余部分做填充

	// Letterbox 填充到 640×640
	const int padding_width = static_cast<int>(input_width) - resized_width;
	const int padding_height = static_cast<int>(input_height) - resized_height;

	left_ = padding_width / 2;
	right_ = padding_width - left_;
	top_ = padding_height / 2;
	bottom_ = padding_height - top_;

	cv::Mat letterboxed;
	cv::copyMakeBorder(
		resized_img,
		letterboxed,
		top_,
		bottom_,
		left_,
		right_,
		cv::BORDER_CONSTANT,
		cv::Scalar(114, 114, 114)
	);

	//cv::imshow("Original Image", image);
	//cv::imshow("resized_img", resized_img);
	cv::imshow("letterboxed", letterboxed);

	// 转为 YOLO 所需的 float32 NCHW 数据
	cv::Mat input_blob;
	///把普通的 OpenCV 图片 cv::Mat 转换成神经网络可以直接使用的四维输入数据。
	///将CHW->NCHW 。blobFromImages() 会根据 std::vector<cv::Mat> 中的图片数量自动确定输出 Blob 的 N
	cv::dnn::blobFromImage(letterboxed, input_blob, 1.0 / 255.0, //缩放比例，把像素值从0-255缩放到0-1
		cv::Size(input_width, input_height), cv::Scalar(), //减去的均值
		true, //交换 R 和 B 通道
		false, //不进行中心裁剪
		CV_32F);

	std::cout << "Input blob shape: ";
	for (int i = 0; i < input_blob.dims; ++i) {
		std::cout << input_blob.size[i] << " ";
	}
	std::cout << std::endl;

	return input_blob;

}

void yolo_ONNX::convert_to_tensor(cv::Mat input_blob)
{
	//6、 使用 OpenCV 的数据创建 ORT Tensor 并运行推理
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_blob.ptr<float>(), input_blob.total(), input_shape.data(), input_shape.size());

	THROW_IF_NOT(input_tensor.IsTensor());


	for (const auto& n : input_names) input_name_ptrs.push_back(n.c_str());
	for (const auto& n : output_names) output_name_ptrs.push_back(n.c_str());

	input_tensors = { std::move(input_tensor) };
}

void yolo_ONNX::run_inference()
{
	Ort::RunOptions run_options;
	output_tensors_ = session_.Run(run_options,
		input_name_ptrs.data(), input_tensors.data(), input_tensors.size(),
		output_name_ptrs.data(), output_name_ptrs.size());

}

const std::vector<Ort::Value>& yolo_ONNX::get_output_tensors() const
{
	return output_tensors_;
}



void yolo_ONNX::Mat_postprocess(const float confidence_threshold, const float nms_threshold)
{
	//7、后处理
	const float* output_data = output_tensors_[0].GetTensorData<float>();
	auto output_info = output_tensors_[0].GetTensorTypeAndShapeInfo();
	std::vector<int64_t> output_shape = output_info.GetShape();
	std::cout << "Output shape: [";
	for (size_t i = 0; i < output_shape.size(); ++i) {
		std::cout << (i == 0 ? "" : ", ") << output_shape[i];
	}
	std::cout << "]\n";
	///输出tensor [Batch, 4 + 类别数, 预测框数量] 
	///前四个通道分别为(center_x, center_y, width, height)

	const int64_t output_batch = output_shape[0];
	const int64_t num_channels = output_shape[1];     // 84
	const int64_t num_candidates = output_shape[2];   // 8400
	const int num_classes = static_cast<int>(num_channels - 4); // 80

	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int64_t i = 0; i < num_candidates; ++i)
	{
		// 输出布局是 [1, 84, 8400]
		const float center_x = output_data[0 * num_candidates + i];
		const float center_y = output_data[1 * num_candidates + i];
		const float width = output_data[2 * num_candidates + i];
		const float height = output_data[3 * num_candidates + i];

		// 找出分数最高的类别
		float best_score = 0.0f;
		int best_class_id = -1;

		for (int class_id = 0; class_id < num_classes; ++class_id) {
			const float class_score =
				output_data[(4 + class_id) * num_candidates + i];

			if (class_score > best_score) {
				best_score = class_score;
				best_class_id = class_id;
			}
		}

		if (best_score < confidence_threshold) {
			continue;
		}

		// YOLO 输出的是基于 640×640 letterbox 图片的中心点坐标
		float x1 = center_x - width / 2.0f;
		float y1 = center_y - height / 2.0f;
		float x2 = center_x + width / 2.0f;
		float y2 = center_y + height / 2.0f;

		// 去掉 letterbox 填充，再映射回原始图片
		x1 = (x1 - left_) / scale_;
		y1 = (y1 - top_) / scale_;
		x2 = (x2 - left_) / scale_;
		y2 = (y2 - top_) / scale_;

		// 限制在原始图片范围内 限幅
		x1 = std::clamp(x1, 0.0f, static_cast<float>(image_raw_.cols - 1));
		y1 = std::clamp(y1, 0.0f, static_cast<float>(image_raw_.rows - 1));
		x2 = std::clamp(x2, 0.0f, static_cast<float>(image_raw_.cols - 1));
		y2 = std::clamp(y2, 0.0f, static_cast<float>(image_raw_.rows - 1));

		const int box_x = static_cast<int>(x1);
		const int box_y = static_cast<int>(y1);
		const int box_width = static_cast<int>(x2 - x1);
		const int box_height = static_cast<int>(y2 - y1);

		if (box_width <= 0 || box_height <= 0) {
			continue;
		}

		boxes.emplace_back(
			box_x,
			box_y,
			box_width,
			box_height
		);

		scores.push_back(best_score);
		class_ids.push_back(best_class_id);
	}


	//非极大值抑制 NMS
	std::vector<int> selected_indices;
	cv::dnn::NMSBoxes(
		boxes,
		scores,
		confidence_threshold,
		nms_threshold,
		selected_indices
	);




	//打印并绘制识别结果
	for (int index : selected_indices) {
		const cv::Rect& box = boxes[index];
		const int class_id = class_ids[index];
		const float score = scores[index];

		std::cout
			<< "class_id=" << class_id
			<< ", score=" << score
			<< ", box=("
			<< box.x << ", "
			<< box.y << ", "
			<< box.width << ", "
			<< box.height << ")\n";

		cv::rectangle(image_raw_, box, cv::Scalar(0, 255, 0), 2);

		std::string label = "class " + std::to_string(class_id) + " " + cv::format("%.2f", score);

		int baseline = 0;

		cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline
		);

		int label_y = std::max(box.y, label_size.height);

		cv::rectangle(
			image_raw_,
			cv::Point(box.x, label_y - label_size.height),
			cv::Point(box.x + label_size.width, label_y + baseline),
			cv::Scalar(0, 255, 0),
			cv::FILLED
		);

		cv::putText(
			image_raw_,
			label,
			cv::Point(box.x, label_y),
			cv::FONT_HERSHEY_SIMPLEX,
			0.5,
			cv::Scalar(0, 0, 0),
			1
		);
	}

	//cv::imwrite("result.jpg", image);
	cv::imshow("Detection Result", image_raw_);


	std::cout << "Detected objects: "
		<< selected_indices.size() << '\n';

	//std::cout << "Result saved to result.jpg\n";

}

