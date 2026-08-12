# HRPlanes

只有飞机一个大类

原始数据集是txt格式的hbb框，遵循yolo标注格式，归一化的[class_id, x, y w, h]

图像和标签命名为[机场_序号]

4800x2703 需要切片

duplicate2obb.py
复制hbb标签，将xywh改成[x1, y1][x2, y2][x3, y3][x4, y4]并写入obb路径
这一步需要在relabel之后执行，因为代码直接复制"1:-1"的类别标签不做检查

relabel_class_id.py
修改类别id并写回hbb标签