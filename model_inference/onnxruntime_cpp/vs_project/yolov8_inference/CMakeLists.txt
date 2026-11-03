cmake_minimum_required(VERSION 3.21)

set(EXECUTABLE_NAME "yolov8_inference")
project(${EXECUTABLE_NAME} LANGUAGES CXX)

# Change this variable to rename the generated executable and CMake target.

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime installation directory")

if(NOT ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "D:/onnxruntime" CACHE PATH "ONNX Runtime installation directory" FORCE)
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers were not found under ${ONNXRUNTIME_ROOT}/include")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    message(FATAL_ERROR "ONNX Runtime import library was not found under ${ONNXRUNTIME_ROOT}/lib")
endif()

add_executable(${EXECUTABLE_NAME} src/main.cpp "src/yolov8_onnx.cpp" "include/yolov8_onnx.h" "include/Headfile.h")
target_compile_features(${EXECUTABLE_NAME} PRIVATE cxx_std_17)
target_include_directories(${EXECUTABLE_NAME} PRIVATE "${ONNXRUNTIME_ROOT}/include"  ${OpenCV_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${EXECUTABLE_NAME} PRIVATE "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"  ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(${EXECUTABLE_NAME} PRIVATE /W4 /permissive- /utf-8)
endif()

add_custom_command(
    TARGET ${EXECUTABLE_NAME}
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>"
    VERBATIM
)
