# TensorRT-7 Project
## Introduction
Python ===> Onnx ===> tensorRT ===> .h/.so <br>
后面会逐步增加更多网络<br>
之前trt-6的项目存在版本兼容性问题，现已舍弃。

## What has it achieved?
* FP32，FP16，INT8量化
* serialize，deserialize
* RetinaFace
* ResNet
* Yolov3
* RetinaNet
* FCOS
* Hourglass

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
* 构建新项目时，需要继承TensorRT类，只需要实现preProcess，postProcess即可。上层封装为initSession和predOneImage两个方法，方便调用。

## RetinaFace
### 简介
* 位置：retinaface_main.cpp
* Python训练原版代码git：[https://github.com/biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
### 注意事项
* 执行convert_to_onnx.py的时候需要更改opset_version=11，verbose=True
* 因为项目不需要关键点，所以把landmark的decode部分去掉了
* 直接使用阈值0.6（原版0.02 + topK）过滤然后接NMS

## Yolov3
### 简介
* 位置：yolo_main.cpp
* Python训练原版代码git：[https://github.com/YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)
* 适配TensorRT修改后的代码git：[https://github.com/Syencil/tensorflow-yolov3](https://github.com/Syencil/tensorflow-yolov3)
### 注意事项
* 训练部分同原版git相同，主要在freeze的时候使用了固定尺寸输入，并修改了python中decode的实现方法。
* NMS代码采用faster-rcnn中的NMS。改动部分在于支持多维度bbox输入，并且shared memory改为动态数组。
* INT8部分50张图差不多就够了

## RetinaNet
### 简介
* 位置：retinanet_main.cpp
* Python训练代码git：[mmdetection](https://github.com/open-mmlab/mmdetection) [configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py)
### 注意事项
* 使用转换onnx时候需要设置opset=11
* 如果在解析onnx时遇到 Assertion failed: ctx->tensors().count(inputName) 这个错误的话，下载最新的onnx-tensorrt源码编译，替换trt对应的lib

## ResNet
### 简介
* 位置：resnet_main.cpp
* 对应任意可直接转换的分类模型

## FCOS
### 简介
* 位置：fcos_main.cpp
* Python训练代码git：[mmdetection](https://github.com/open-mmlab/mmdetection) [configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py)
### 注意事项
* 目前trt暂时不支持Group Normalization，如果需要使用GN版本需要单独实现。
* 有空会更新GN

## Keypoints Detecton
### 简介
* 位置：hourglass_main.cpp（Hourglass）
* Python训练代码git：[https://github.com/Syencil/Keypoints](https://github.com/Syencil/Keypoints)

