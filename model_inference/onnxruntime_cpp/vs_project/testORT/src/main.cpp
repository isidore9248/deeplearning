#include <iostream>

#include <onnxruntime_cxx_api.h>

int main() {
	try {
		Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "ort_smoke" };
		Ort::SessionOptions session_options;
		session_options.SetGraphOptimizationLevel(
			GraphOptimizationLevel::ORT_ENABLE_ALL
		);

		std::cout << "ONNX Runtime C++ is ready.\n";
		std::cout << "Runtime version: "
			<< OrtGetApiBase()->GetVersionString()
			<< '\n';
		std::cout << "Execution provider: CPU\n";
		return 0;
	}
	catch (const Ort::Exception& error) {
		std::cerr << "ONNX Runtime error: " << error.what() << '\n';
		return 1;
	}
}