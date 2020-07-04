cmake_minimum_required(VERSION 3.5)
project(tensorRT)

set_property(GLOBAL PROPERTY USE_FOLDERS on)

# output
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")
message(STATUS "Project_binary_dir : ${PROJECT_BINARY_DIR}")
# c++ 11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# tensorRT
#set(tensorrt_dir /work/TensorRT-6.0.1.5)
set(tensorrt_dir /work/TensorRT-7.0.0.11)
set(project_dir /work/tensorRT-7)
include_directories(${tensorrt_dir}/include)
include_directories(${project_dir}/include)
link_directories(${tensorrt_dir}/lib)
link_directories(${project_dir}/source)
link_directories(${project_dir}/lib)

# Loggers
aux_source_directory(${common_dir}/source common_src)
set(COMMON_SRC ${common_src} CACHE INTERNAL "common_source" )
set(LOGGER_SRC ${common_dir}/source/logger.cpp CACHE INTERNAL "logger" )

message(STATUS "TensorRT Header => ${tensorrt_dir}/include")
message(STATUS "TensorRT Lib => ${tensorrt_dir}/lib")

# find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if(NOT OpenCV_LIBRARY_DIRS)
    set(OpenCV_LIBRARY_DIRS /usr/local/lib)
    message(WARING " Can not find opencv lib. It will use the default path => ${OpenCV_LIBRARY_DIRS}")
endif()
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV_INCLUDE_DIRS => ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBRARY_DIRS => ${OpenCV_LIBRARY_DIRS}")

if(NOT OpenCV_FOUND)
    message(ERROR "OpenCV not found!")
endif(NOT OpenCV_FOUND)

# find cuda
find_package(CUDA)
find_package(CUDA REQUIRED)

#include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda/targets/x86_64-linux/include)
if(NOT CUDA_LIBRARY_DIRS)
    set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64)
    message(WARING " Can not find CUDA lib. It will use the default path => ${CUDA_LIBRARY_DIRS}")
endif()
link_directories(${CUDA_LIBRARY_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS : ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARY_DIRS : ${CUDA_LIBRARY_DIRS}")

###############################################
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${project_dir}/lib)
set(TRT source/tensorrt.cpp source/logger.cpp source/utils.cpp source/utils.cu)
set(INT8 source/Int8Calibrator.cu source/Int8Calibrator.cpp)

set(YOLO3 ${TRT} ${INT8} source/yolov3.cpp)
set(HOURGLASS ${TRT} ${INT8} source/hourglass.cpp)
set(RESNET ${TRT} ${INT8} source/resnet.cpp)
set(FCOS ${TRT} ${INT8} source/fcos.cpp)
set(RETINANET ${TRT} ${INT8} source/retinanet.cpp)
set(RETINAFACE ${TRT} ${INT8} source/retinaface.cpp)
set(YOLO5 ${TRT} ${INT8} source/yolov5.cpp)

set(CV_LIB libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so)
set(TRT_LIB libnvinfer.so libnvonnxparser.so cudart.so)

cuda_add_executable(yolov3 yolov3_main.cpp)
target_link_libraries(yolov3 yolo3trt.so ${TRT_LIB} ${CV_LIB})

cuda_add_executable(hourglass hourglass_main.cpp)
target_link_libraries(hourglass hourglasstrt.so ${TRT_LIB} ${CV_LIB})

cuda_add_executable(resnet resnet_main.cpp ${RESNET})
target_link_libraries(resnet resnettrt.so ${TRT_LIB} ${CV_LIB})

cuda_add_executable(fcos fcos_main.cpp)
target_link_libraries(fcos fcostrt.so ${TRT_LIB} ${CV_LIB})

cuda_add_executable(retinanet retinanet_main.cpp)
target_link_libraries(retinanet retinanettrt.so ${TRT_LIB} ${CV_LIB})

cuda_add_executable(retinaface retinaface_main.cpp)
target_link_libraries(retinaface retinafacetrt.so ${TRT_LIB} ${CV_LIB})

cuda_add_executable(yolo5 yolov5_main.cpp ${TRT} ${INT8} source/yolov5.cpp)
target_link_libraries(yolo5 ${TRT_LIB} ${CV_LIB})

cuda_add_executable(pse psenet_main.cpp ${TRT} ${INT8} source/psenet.cpp)
target_link_libraries(pse ${TRT_LIB} ${CV_LIB})

#########################3#####################
cuda_add_library(yolo3trt SHARED ${YOLO3})
cuda_add_library(hourglasstrt SHARED ${HOURGLASS})
cuda_add_library(resnettrt SHARED ${RESNET})
cuda_add_library(fcostrt SHARED ${FCOS})
cuda_add_library(retinanettrt SHARED ${RETINANET})
cuda_add_library(retinafacetrt SHARED ${RETINAFACE})
cuda_add_library(yolo5trt SHARED ${YOLO5})

