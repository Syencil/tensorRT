# TensorRT-7 Network Lib
## Introduction
Python ===> Onnx ===> tensorRT ===> .h/.so <br>
支持多线程进行预处理和后处理。支持FP32，FP16，INT8量化。支持serialize，deserialize <br>
支持线程池实现。支持infer时GPU和CPU端异步进行实现延迟隐藏 (TODO) <br>

## Model Zoo
|Model|Training git|Infer Time|Total Time|
|----|----|----|----|
|PANNet(Pse++)|[https://github.com/WenmuZhou/PAN.pytorch](https://github.com/WenmuZhou/PAN.pytorch)<br>[https://github.com/Syencil/PAN.pytorch](https://github.com/Syencil/PAN.pytorch)|18.5ms|45ms|
|PSENet|[https://github.com/WenmuZhou/PSENet.pytorch](https://github.com/WenmuZhou/PSENet.pytorch)|22ms|48ms|
|Yolov5x|[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) <br> [https://github.com/Syencil/yolov5](https://github.com/Syencil/yolov5)|32.5ms|58ms|
|Yolov3|[https://github.com/YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3) <br> [https://github.com/Syencil/tensorflow-yolov3][https://github.com/Syencil/tensorflow-yolov3]|14.5ms|29.5ms|
|Retinaface|[https://github.com/biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) <br> [https://github.com/Syencil/Pytorch_Retinaface](https://github.com/Syencil/Pytorch_Retinaface)|2.3ms|12.3ms|
|Retinanet|[mmdetection](https://github.com/open-mmlab/mmdetection) + [configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py)|22.9ms|333ms|
|Fcos|[mmdetection](https://github.com/open-mmlab/mmdetection) + [configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py)|-|-|
|ResNet|-|-|-|
|Hourglass|[https://github.com/Syencil/Keypoints](https://github.com/Syencil/Keypoints)|28ms|37ms|
* 测试环境为Tesla P40 + 4个CPU线程。

## Quick Start
### Code -> Onnx
| |git|Convert|
|----|----|---|
|tensorflow|[https://github.com/onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)|```python -m tf2onnx.convert ```|
|pytorch|-|```torch.onnx.export(model, img, weights, verbose=False, opset_version=11, input_names=['images'], output_names=['output'])```|
|Onnx|[onnx-simplifier](https://github.com/daquexian/onnx-simplifier)|```python3 -m onnxsim in.onnx out.onnx```|
### C++
```
cmake -DCMAKE_BUILD_TYPE=Release . 
make
cd bin
./project_name
```

## Tips
* Onnx必须指定为输入全尺寸，再实际中trt也不存在理想上的动态输入，所以必须在freeze阶段指明输入大小。<br>
* 构建新项目时，需要继承TensorRT类，只需要实现preProcess，postProcess即可。上层封装为initSession和predOneImage两个方法，方便调用。
* 由于ONNX和TRT目前算子实现的比较完善，大多数时候只需要实现相应后处理即可，针对特定算子通常可以再python代码中用一些trick进行替换，实在不行可以考虑自定义plugin
* 关于CHW和HWC的数据格式
* * CHW: 对于GPU更优。使用CUDA做infer或者后处理的话，由于硬件DRAM的原因，CHW可以保证线程是以coalescing的方式读取。具体性能对比参考[Programming_Massively_Parallel_Processors](https://github.com/Syencil/Programming_Massively_Parallel_Processors)
* * HWC: 对于CPU更优。使用CPU进行处理的时候，HWC格式可以保证单个线程处理的数据具有连续的内存地址。而CPU缓存具有[空间局部性](https://zh.wikipedia.org/wiki/CPU%E7%BC%93%E5%AD%98)，这样能极大的提升效率。
* * 综上：如果后处理使用CPU进行decode，建议在onnx输出HWC格式，如果使用GPU进行decode，建议在onnx输出CHW格式。对于输入端则没有具体测试，主要是相信tensorflow虽然按照之前的HWC格式，但是在操作中肯定也是做了优化

## PanNet (PseNet V2)
### 简介
* 位置：psenetv2_main.cpp
* python训练原版代码git：[https://github.com/WenmuZhou/PAN.pytorch](https://github.com/WenmuZhou/PAN.pytorch)
* 适配TensorRT修改后的代码git：[https://github.com/Syencil/PAN.pytorch](https://github.com/Syencil/PAN.pytorch)
### 注意事项
* pan和pse代码其实高度相似，导出的方法可以参考PseNet也可以参考我fork后改的代码。
* pan网络中转出onnx的结果是没有经过sigmoid的(尝试一下加在后处理)
* sigmoid在CPU中计算耗时比较大，可以参考[fast-sigmoid-algorithm](https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm)。
CPU上性能对比结果```100000 times     sigmoid ==> 2.81878ms   fast sigmoid ==> 0.589737ms```，而GPU上两者差异忽略不记。
```
    fast_sigmoid(x) = (x / (1 + |x|)) * 0.5 + 0.5
```

## PseNet
### 简介
* 位置：psenet_main.cpp
* python训练原版代码git：[https://github.com/WenmuZhou/PSENet.pytorch](https://github.com/WenmuZhou/PSENet.pytorch)
### 注意事项
* torch转onnx的代码可以加在predict.py中，只需要在Pytorch_model这个类里面加一个成员函数即可
```
    def export(self, onnx_path, input_size):
        assert isinstance(input_size, list) or isinstance(input_size, tuple)
        self.net.export = True
        img = torch.zeros((1, 3, input_size[0], input_size[1])).to(self.device)
        with torch.no_grad():
            torch.onnx.export(self.net, img, onnx_path, verbose=True, opset_version=11, export_params=True, do_constant_folding=True)
        print("Onnx Simplify...")
        os.system("python3 -m onnxsim {} {}".format(onnx_path, onnx_path))
        print('Export complete. ONNX model saved to %s\nView with https://github.com/lutzroeder/netron' % onnx_path)
```
* 为了方便trt的处理，我把sigmoid加入到了torch的代码中。在models/model.py中修改PSENet的forward代码，同时__init__中加入成员变量export=False来控制
```
        if self.export:
            x = torch.sigmoid(x)
        return x
```
* 在onnx转换为trt的时候可能会出现[This version of TensorRT only supports asymmetric](https://github.com/NVIDIA/TensorRT/issues/422)这个问题，bilinear的上采样方式可能会存在问题，解决方式是将所有的F.interpolate中的align_corners=True，同时[修改onnx-tensorrt中对应的cpp然后重新编译替换trt的lib](https://github.com/onnx/onnx-tensorrt/pull/418/files)
* 如果需要看每一个kernel的特征图，只需要在psenet.cpp里面把注释打开即可。


## Yolov5
### 简介
* 位置：yolov5_main.cpp
* python训练原版代码git：[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
### 注意事项
* trt的decode针对的是BxHxWxAC的格式（方便按height方向并行化以及其他嵌入式接入）。原版yolov5导出的onnx是BxAxHxWxC，需要在models/yolo.py第28行改为
```
            if self.export:
                x[i] = x[i].permute(0, 2, 3, 1).contiguous()
            else:
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
```

## RetinaFace
### 简介
* 位置：retinaface_main.cpp
* Python训练原版代码git：[https://github.com/biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
### 注意事项
* 执行convert_to_onnx.py的时候需要更改opset_version=11，verbose=True
* 因为项目不需要关键点，所以把landmark的decode部分去掉了
* 直接使用阈值0.6（原版0.02 + topK）过滤然后接NMS
* 支持多线程操作

## Yolov3
### 简介
* 位置：yolov3_main.cpp
* Python训练原版代码git：[https://github.com/YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)
* 适配TensorRT修改后的代码git：[https://github.com/Syencil/tensorflow-yolov3](https://github.com/Syencil/tensorflow-yolov3)
### 注意事项
* 训练部分同原版git相同，主要在freeze的时候使用了固定尺寸输入，并修改了python中decode的实现方法。修改为core/yolov3.py增加一个decode_full_shape类方法
```
    def decode_full_shape(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """
        conv_shape = conv_output.get_shape().as_list()
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class), name="reshape")

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y_np = np.tile(np.arange(output_size, dtype=np.int32)[..., np.newaxis], [1, output_size])
        x_np = np.tile(np.arange(output_size, dtype=np.int32)[np.newaxis, ...], [output_size, 1])

        xy_grid_np = np.concatenate([np.reshape(x_np, [np.shape(x_np)[0], np.shape(x_np)[1], 1]), np.reshape(y_np, [np.shape(y_np)[0], np.shape(y_np)[1], 1])], axis=2)
        xy_grid_np = np.tile(np.reshape(xy_grid_np, [1, np.shape(xy_grid_np)[0], np.shape(xy_grid_np)[1], 1, np.shape(xy_grid_np)[2]]), [batch_size, 1, 1, anchor_per_scale, 1])

        anchor_np = np.tile(np.reshape(anchors, [1, 1, 1, -1]), [batch_size, output_size, output_size, 1])

        xy_grid = tf.constant(xy_grid_np, dtype=tf.float32)
        stride_tf = tf.constant(shape=[batch_size, output_size, output_size, anchor_per_scale * 2], value=stride, dtype=tf.float32)
        anchor_tf = tf.constant(anchor_np, dtype=tf.float32)

        pred_xy = tf.sigmoid(conv_raw_dxdy)
        pred_wh = tf.exp(conv_raw_dwdh)

        pred_xy = tf.reshape(pred_xy, [batch_size, output_size, output_size, anchor_per_scale * 2])
        pred_wh = tf.reshape(pred_wh, [batch_size, output_size, output_size, anchor_per_scale * 2])
        xy_grid = tf.reshape(xy_grid, [batch_size, output_size, output_size, anchor_per_scale * 2])

        pred_xy = tf.add(pred_xy, xy_grid)
        pred_xy = tf.multiply(pred_xy, stride_tf)
        pred_wh = tf.multiply(pred_wh, anchor_tf)
        pred_wh = tf.multiply(pred_wh, stride_tf)

        pred_xy = tf.reshape(pred_xy, [batch_size, output_size, output_size, anchor_per_scale, 2])
        pred_wh = tf.reshape(pred_wh, [batch_size, output_size, output_size, anchor_per_scale, 2])

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=4)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=4, name="decode")
```
* NMS代码采用faster-rcnn中的NMS。改动部分在于支持多维度bbox输入，并且shared memory改为动态数组。
* INT8部分50张图差不多就够了
* 支持多线程操作

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

## Hourglass
### 简介
* 位置：hourglass_main.cpp（Hourglass）
* Python训练代码git：[https://github.com/Syencil/Keypoints](https://github.com/Syencil/Keypoints)

## 更新日志
### 2020.08.26
1. 整体抽象化，严格按照面向对象将模型剥离出来，提高代码复用率。
2. Detection类已转换网络：yolov5，yolov3，retinaface，retinanet，fcos
3. Segmentation类已转换网络：psenet，psev2
4. 尚未转换的网络：rensnet（分类），hourglass（关键点）
### 2020.08.19
1. 重新写了CMAKE，把历史遗留的一些问题解决了。重新组织语言，尽可能用cmake内置的一些变量，同时把依赖的一些路径进行合并，提升工程的可移植性。
2. 测了OpenMP，在多线程执行任务的速度上不如std\:\:thread，但是需要频繁开启销毁线程时速度比std\:\:thread快。后来查到OpenMP是基于线程池的，故考虑用线程池来代替。
### 2020.07.13
1. 增加OCR系列的PANNet，即PSEv2。模型整体轻量化，且不需要像PSE那样设置这么多kernel。
不过讲道理，PSE和PAN在decode的时候都要遍历所有的文本像素点，并没有快很多。测试发现decode部分的实际速度差距已经很小了，感觉FPS提升主要还是换了轻量的backbone。
2. 这一次将后处理的sigmoid操作没有放到onnx中。同时在网上发现一个fast sigmoid的操作。不过如果走GPU的话差异并不大，走CPU的话速度差了5 6倍。
3. 改了一下OCR检测输出，既有mask也有RBox
### 2020.07.04
1. 增加OCR系列的PSENet。infer时间不算太慢，但是decode部分的渐进式扩张算法耗时太久，这一块其实可以再优化。
### 2020.06.30
1. 细节修复，使用cudaEvent来计算时间，而不是cpu
### 2020.06.17
1. U版的yolo性能确实好，Python代码封装的也很不错。试了一下yolo5s转成trt速度快了好多，但是准确率也不低
2. 目前准备转OCR中的ctpn，但是由于代码复现效果一直不好，可能还得等一段时间。
3. 有空更新一下ncnn的git
### 2020.05.28
今天开始决定把每次更新的内容记录一下。<br>
1. 支持多线程（部署在yolo和retinaface上）
2. 把BRG转RGB和HWC转CHW放在一起操作
3. worker=4的时候前后处理总用时可以从30ms压缩到15ms左右
