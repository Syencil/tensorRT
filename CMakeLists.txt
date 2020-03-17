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
set(TRT source/tensorrt.cpp source/logger.cpp source/utils.cpp)
set(INT8 source/Int8Calibrator.cu source/Int8Calibrator.cpp)
set(YOLO ${TRT} ${INT8} source/yolo.cpp source/yolo.cu)
set(HOURGLASS ${TRT} ${INT8} source/hourglass.cpp)

cuda_add_executable(yolo yolo_main.cpp)
target_link_libraries(yolo yolotrt.so libnvinfer.so libnvonnxparser.so cudart.so  libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so)

cuda_add_executable(hourglass hourglass_main.cpp)
target_link_libraries(hourglass hourglasstrt.so libnvinfer.so libnvonnxparser.so cudart.so  libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so)


#########################3#####################
cuda_add_library(yolotrt SHARED ${YOLO})
cuda_add_library(hourglasstrt SHARED ${HOURGLASS})