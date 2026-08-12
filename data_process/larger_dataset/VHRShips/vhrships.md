# VHRShips
> 最有可能出现问题的数据集

原始数据是.mat格式的table表，其中包含嵌套的cell
由于不知道内部结构，无法直接使用python解析其内容
提取出json文件（这个不是标准json格式）包括图像名称，类别名称以及边界框数据（不一定正确）
包含像素坐标的边界框，由于所有图像均为1280x720，因此可以转换归一化的坐标
（我缺少图像，没法验证提取的对不对）

仅使用 check_hbb_obb.py检查了一下，可能（也许大概）没问题

read_class.py是读取json，并统计其中类别。
标签中类别名称后面跟了一个数字，但是打印出来发现那个并不是id

get_hbblabels.py
从json中提取hbb标签，根据所有图像都是1280x720转换像素坐标

relabel_class_id.py
更改hbb标签的类别映射，在duplicate2obb之前运行

duplicate2obb.py
代码与同名代码几乎相同，复制hbb标签到obb