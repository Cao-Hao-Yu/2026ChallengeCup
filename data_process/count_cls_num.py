import os
from collections import Counter

"""
    对数据集实例数量的统计
    AI代码
"""

"""
    0       HM             17        
    1       LQS            30        
    2       QHS            641       
    3       MS             1994      
    4       A1_SU-35       1317      
    5       A2_C-130       1297      
    6       A3_C-17        998       
    7       A4_C-5         500       
    8       A5_F-16        1017      
    9       A6_TU-160      361       
    10      A7_E-3         547       
    11      A8_B-52        750       
    12      A9_P-3C        895       
    13      A10_B-1B       762       
    14      A11_E-8        432       
    15      A12_TU-22      583       
    16      A13_F-15       1265      
    17      A14_KC-135     1424      
    18      A15_F-22       493       
    19      A16_FA-18      2147      
    20      A17_TU-95      1114      
    21      A18_KC-10      262       
    22      A19_SU-34      933       
    23      A20_SU-24      752       
    24      FSC            402       
    总实例数量: 20933
"""


# 配置路径 这段代码是在划分数据集之前运行的，如果划分了之后再统计别忘了把val加上
label_dir = r"D:\DeepLearning\Challenger\data\dataset_yolo\labels\train"

# 类别名称字典
class_names = {
    0: 'HM', 1: 'LQS', 2: 'QHS', 3: 'MS', 4: 'A1_SU-35', 
    5: 'A2_C-130', 6: 'A3_C-17', 7: 'A4_C-5', 8: 'A5_F-16', 9: 'A6_TU-160', 
    10: 'A7_E-3', 11: 'A8_B-52', 12: 'A9_P-3C', 13: 'A10_B-1B', 14: 'A11_E-8', 
    15: 'A12_TU-22', 16: 'A13_F-15', 17: 'A14_KC-135', 18: 'A15_F-22', 19: 'A16_FA-18', 
    20: 'A17_TU-95', 21: 'A18_KC-10', 22: 'A19_SU-34', 23: 'A20_SU-24', 24: 'FSC'
}

# 使用 Counter 来进行统计，方便快捷
class_counts = Counter()
total_instances = 0

# 检查路径是否存在
if not os.path.exists(label_dir):
    print(f"路径不存在，请检查: {label_dir}")
else:
    # 遍历文件夹下的所有文件
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(label_dir, filename)
            
            with open(filepath, 'r') as f:
                for line in f:
                    # 去除首尾空格并按空格分割
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_instances += 1

    # 打印统计结果
    print("="*40)
    print(f"训练集标签统计结果 (路径: {label_dir})")
    print("="*40)
    print(f"{'类别ID':<8}{'类别名称':<15}{'实例数量':<10}")
    print("-" * 40)
    
    # 按照 ID 从小到大排序输出
    for class_id in sorted(class_names.keys()):
        count = class_counts.get(class_id, 0)
        print(f"{class_id:<8}{class_names[class_id]:<15}{count:<10}")
        
    print("="*40)
    print(f"总实例数量: {total_instances}")
    print("="*40)
