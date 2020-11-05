cmake_minimum_required(VERSION 3.5)
project(tensorRT)

set_property(GLOBAL PROPERTY USE_FOLDERS on)

# c++ 11 / 14 / 17
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

# Output
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
message(STATUS "Project_binary_dir : ${PROJECT_BINARY_DIR}")

# Project
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/source)
link_directories(${PROJECT_SOURCE_DIR}/lib)

# find opencv
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV FOUND")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV Not Found!")
endif()

# find cuda
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA FOUND")
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
else()
    message(FATAL_ERROR "CUDA Not Found!")
endif()


###############################################
set(TRT source/tensorrt.cpp source/logger.cpp source/utils.cpp source/utils.cu)
set(INT8 source/Int8Calibrator.cu source/Int8Calibrator.cpp)

set(YOLO3 ${TRT} ${INT8} source/yolov3.cpp)
set(HOURGLASS ${TRT} ${INT8} source/hourglass.cpp)
set(RESNET ${TRT} ${INT8} source/resnet.cpp)
set(FCOS ${TRT} ${INT8} source/fcos.cpp)
set(RETINANET ${TRT} ${INT8} source/retinanet.cpp)
set(RETINAFACE ${TRT} ${INT8} source/retinaface.cpp)
set(YOLO5 ${TRT} ${INT8} source/yolov5.cpp)
set(PSENET ${TRT} ${INT8} source/psenet.cpp)
set(PSENETV2 ${TRT} ${INT8} source/psenetv2.cpp)
set(SIMPLEPOSE ${TRT} ${INT8} source/simplePose.cpp)

#set(CV_LIB libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so) // 主要用到这几个
set(TRT_LIB libnvinfer.so.7.0.0 libnvonnxparser.so.7.0.0 cuda.so)

cuda_add_executable(yolov3 yolov3_main.cpp)
target_link_libraries(yolov3 yolov3trt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(hourglass hourglass_main.cpp)
target_link_libraries(hourglass hourglasstrt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(resnet resnet_main.cpp)
target_link_libraries(resnet resnettrt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(fcos fcos_main.cpp)
target_link_libraries(fcos fcostrt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(retinanet retinanet_main.cpp)
target_link_libraries(retinanet retinanettrt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(retinaface retinaface_main.cpp)
target_link_libraries(retinaface retinafacetrt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(yolov5 yolov5_main.cpp)
target_link_libraries(yolov5 yolov5trt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(pse psenet_main.cpp)
target_link_libraries(pse psenettrt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(psev2 psenetv2_main.cpp)
target_link_libraries(psev2 psenetv2trt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(simplePose simplePose_main.cpp)
target_link_libraries(simplePose simplePosetrt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(stream stream_main.cpp)
target_link_libraries(stream yolov5trt.so ${TRT_LIB} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

#########################3#####################
cuda_add_library(yolov3trt SHARED ${YOLO3})
cuda_add_library(hourglasstrt SHARED ${HOURGLASS})
cuda_add_library(resnettrt SHARED ${RESNET})
cuda_add_library(fcostrt SHARED ${FCOS})
cuda_add_library(retinanettrt SHARED ${RETINANET})
cuda_add_library(retinafacetrt SHARED ${RETINAFACE})
cuda_add_library(yolov5trt SHARED ${YOLO5})
cuda_add_library(psenettrt SHARED ${PSENET})
cuda_add_library(psenetv2trt SHARED ${PSENETV2})
cuda_add_library(simplePosetrt SHARED ${SIMPLEPOSE})

