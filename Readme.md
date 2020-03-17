# TensorRT-7 Project
## Introduction
Python ===> Onnx ===> tensorRT ===> .h/.so <br>
后面会逐步增加更多网络<br>
之前trt-6的项目存在版本兼容性问题，现已舍弃。

## What has it achieved?
* FP32，FP16，INT8量化
* serialize，deserialize
* ~~Upsample Plugin~~在trt7上已经官方支持了

## Quick Start
### Python
* Freeze Graph
* [https://github.com/onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)
```
python -m tf2onnx.convert 
    [--input SOURCE_GRAPHDEF_PB]
    [--graphdef SOURCE_GRAPHDEF_PB]
    [--checkpoint SOURCE_CHECKPOINT]
    [--saved-model SOURCE_SAVED_MODEL]
    [--output TARGET_ONNX_MODEL]
    [--inputs GRAPH_INPUTS]
    [--outputs GRAPH_OUTPUS]
    [--inputs-as-nchw inputs_provided_as_nchw]
    [--opset OPSET]
    [--target TARGET]
    [--custom-ops list-of-custom-ops]
    [--fold_const]
    [--continue_on_error]
    [--verbose]
```
### C++
```
cmake . 
make
cd bin
./project_name
```

## Attention
* Onnx必须指定为输入全尺寸，再实际中trt也不存在理想上的动态输入，所以必须在freeze阶段指明输入大小。<br>
* 所有大图输入均默认为（1, 512,512, 3）<br>
* 构建新项目时，需要继承TensorRT类，只需要实现preProcess，postProcess即可。上层封装为initSession和predOneImage两个方法，方便调用。

## Object Detection
### 简介
* 位置：yolo_main.cpp
* Python训练原版代码git：[https://github.com/YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)
* 适配TensorRT修改后的代码git：[https://github.com/Syencil/tensorflow-yolov3](https://github.com/Syencil/tensorflow-yolov3)
### 注意事项
* 训练部分同原版git相同，主要在freeze的时候使用了固定尺寸输入，并修改了python中decode的实现方法。
* NMS代码采用faster-rcnn中的NMS。改动部分在于支持多维度bbox输入，并且shared memory改为动态数组。
* INT8部分50张图差不多就够了

## Keypoints Detecton
### 简介
* 位置：hourglass_main.cpp（Hourglass）
* Python训练代码git：[https://github.com/Syencil/Keypoints](https://github.com/Syencil/Keypoints)
### 注意事项
* ~~Upsample层为自定义层，如果上采样用反卷积则不需要。具体训练细节移步到对应git。~~

