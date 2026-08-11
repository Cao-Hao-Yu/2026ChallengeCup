"""
这貌似不是类别id
tug                  5
bulkCarrier          10
container            11
tanker               13
undefined            20
roro                 3
dredging             15
dredgerReclamation   32
coaster              35
smallBoat            34
coastGuard           800
oreCarrier           14
generalCargo         1
yatch                18
patrolForce          600
serviceCraft         1000
bargePontoon         7
drill                19
other                1100
oilTanker            12
floatingDock         36
offshore             6
passanger            4
smallPassanger       33
ferry                2
lpg                  17
cruiser              300
destroyer            400
auxilary             900
submarine            100
landing              700
fishing              9
frigate              500
aircraft             200
"""

import json
from collections import OrderedDict
import os

# JSON文件路径
json_path = r"D:\DeepLearning\Challenger\data\LargerDataset\VHRShips\labels\annotations.json"

# 读取JSON文件
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

category_dict = OrderedDict()

# 遍历所有图像的标注
for item in data:
    annotations = item['annotations']
    
    # 【关键修改】检查annotations是否为字符串，如果是则解析为列表
    if isinstance(annotations, str):
        annotations = json.loads(annotations)
    
    # 现在annotations一定是列表了，可以安全遍历
    for annotation in annotations:
        # 为了安全，先检查annotation是否为字典类型
        if isinstance(annotation, dict):
            category = annotation['category']
            category_id = annotation['category_id']
            
            if category not in category_dict:
                category_dict[category] = category_id

# 打印结果
print("=" * 40)
print(f"{'类别名称'.ljust(20)} 类别ID")
print("=" * 40)
for category, cat_id in category_dict.items():
    print(f"{category.ljust(20)} {cat_id}")
print("=" * 40)
print(f"总共有 {len(category_dict)} 个类别")
