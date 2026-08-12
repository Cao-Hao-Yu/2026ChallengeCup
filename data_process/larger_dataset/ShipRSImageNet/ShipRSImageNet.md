# ShipRSImageNet

原始标签提供coco格式，voc格式，并且有非常详细的层级，同时包括polygon，bndbox和rotatebox

coco格式划分了train，val，test数据集，我们需要重新划分，于是代码是基于voc格式的xml文件处理的

get_classes.py
获取类别名称和层级

extract_labels.py
提取hbb和obb标签并写入指定路径，obb优先使用polygon角点，如果角点缺失或格式错误会使用rotatebox推导

relabel_class_id.py
在最后运行，重新映射hbb和obb的标签