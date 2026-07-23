# Challenge Cup

目前该仓库包括两个文件夹分别为ultralytics以及data_process的代码

## 写了一个手动合成数据的软件

解压后双击BoatPicker.console.exe运行

链接: https://pan.baidu.com/s/1K0AtAJCMjOplcuBh0eJxKQ?pwd=rfah 提取码: rfah

## data_process

包括.py文件分别实现

1. check_data
   
   临时代码，用于debug
2. count_cls_data
   
   统计每个类别实例数量
3. make_10k_data
   
   实例化10k画布，然后选择图片贴在画布上，图片有缩放，旋转等增强
4. split_dataset
   
   划分数据集，按照每个类别8: 2划分
5. visualize_img
   
   可视化图片及标注框
6. visualize_spec_img
   
   可视化特定类别的图片及标注框
7. yolo2coco
   
   yolo格式标注框转coco格式
8. yolo2coco10k
   
   yolo格式标注框转coco格式，与7类似



## ultralytics

官网上下的ultralytics库 -> [官方仓库](https://github.com/ultralytics/ultralytics)

增加了一些自定义的模块以及训练，验证和辅助功能的代码

由于修改了ultralytics库里面的代码，所以环境需要在ultralytics目录下使用以下代码本地安装

```
pip install -e .
```

新建了./models, ./train, ./val文件夹

新建./ultralytics/nn/modules/custom_block.py文件存放新定义的模块

修改./ultralytics/nn/task.py注册自定义模块

修改./ultralytics/nn/modules/block.py

修改./ultralytics/utils/loss.py
