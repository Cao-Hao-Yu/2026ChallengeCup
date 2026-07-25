# Challenge Cup

目前该仓库包括两个文件夹分别为ultralytics以及data_process的代码

## 写了一些小工具

建议解压后双击带console后缀的运行，便于通过控制台查看输出

链接 -> [百度网盘](https://pan.baidu.com/s/1K0AtAJCMjOplcuBh0eJxKQ?pwd=rfah) 

## data_process

包括多个辅助.py文件分别实现统计类别数量，数据集划分，可视化标注框，格式转换等等

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
