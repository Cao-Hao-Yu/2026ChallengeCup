# Challenge Cup

### Some Tools

链接 => [百度网盘](https://pan.baidu.com/s/1K0AtAJCMjOplcuBh0eJxKQ?pwd=rfah) 

### Folder ./data_process

包括两个文件夹

data_process/our_early_dataset_process/

> 包括比赛数据集的一些处理程序

data_process/larger_dataset/

> 包括制作更大数据集所用处理程序

### Folder ./ultralytics

需要在ultralytics目录下使用以下代码本地安装ultralytics库

```python
pip install -e .
```

### What You Need To Know

1. 我把标注改了

> 原来是 N:n 形式标注 base class: spec class。现在改成 (N+1)n 形式标注
> 
> ```md
> eg. 1:8 => 28, 0:-1 => 125, 2:24 => 324, 0:2 => 12
> ```
> 
> 大类从1开始重新标注，小类保持原来不变，缺失标签从-1变成25
> 
> 这种修改是为了适配ultralytics库中无处不在的 int(lb["cls"])
> 
> 手工改代码比较痛苦，因此我决定采用痛苦最小的方法，通过混合标签藏一个数据，这样保证对原来代码的侵入性最小，代价就是兼容性不高并且还会藏下难以排查的bug。~~（代码能跑就行）~~

2. 核心代码是古法编程的

> 这意味着我的 **兼容性非常差** 因此在更改代码之前推荐先在代码>中搜索 **!?注注?!** 找到我写的注释和修改点 ~~（还有一些坑）~~
> 
> 数据处理的大部分代码是AI写的，并且大部分AI代码经过 **人工** 运行验证。而对ultralytics库的修改是 **完全** 人工代码，虽然经过运行验证，但难免会有错误
> 
> **请注意**：我做的修改仅能保证最低限度的训练和验证循环运行，其余的情况我要么觉得用不上就没管，要么注释掉会报错的断言语句~~（填混合标注挖的坑）~~

3. 我有一个测试基准值

> 代码大部分都经过测试，并且我也测试得到了一个模型基准值。因此更大的模型和更大的数据集应该能取得更好的结果，如果指标看上去有问题首先检查数据集
> 
> 并且该指标的recall还是在所有类别都是0.5的iou下测得的，所以我们的模型在测试时指标应该还会更好
> 
> ```md
> mAP50-95:72
> recall:91.3, fp_rate:23.3 when conf threshold = 0.35
> ```

### What You Need To Do

1. 确保数据集标注提取与转换正确

> 每个小数据集中有对应的处理代码与说明

2. 我将标签格式改成混合标注格式，弃用原来冒号格式，确保所有标签被正确转换

> data_process/larger_dataset/remap_class_id.py 提供转换代码

3. 确保船舶图像全是灰度图

> data_process/larger_dataset/change2grey.py提供转换代码

4. 删除过曝图像

> 未提供代码，但是给了一个空的 wash_dataset.py [doge]

5. 对大图做切片

> data_process/larger_dataset/slice_image.py提供切片代码
> 
> 代码中的参数可能需要进一步调整，可以配合提供的check_slice_image.py代码检查切片结果
> 
> 可以在wash_dataset.py中写清洗数据集的逻辑删除一些不那么好的切片图（如果我们的数据非常多，就可以随便删除，什么样算不那么好的切片图得对着切片结果才知道）
> 
> 也可以故意切一些（或者不切/切部分）比较大的图进行训练（兴许能增强模型多尺度的能力）

6. 删除空标签图像

> 有些数据集中包含空标签图像作为负标签，并且切片也会产生空标签
> 
> 对于空标签，我的想法是直接删，因为我们数据集比较大

7. 模型改完路径可以一键运行

> 改一下模型路径和数据集路径
> 
> 改一下batch和epochs（目前是200，如果数据集很大的话可能200太多了，并且时间也不一定够）
> 
> 应该就可以一键运行了
>
> ./ultralytics-main/train/train_yolo.py 训练模型
> 
> ./ultralytics-main/val/val.py 验证模型
>
> 将验证路径改成测试路径也可以正常使用
