import os
import shutil
import random
from collections import defaultdict


"""
    # !?注注?!
    8:2划分train和val
    在原数据集文件夹中进行划分
    每个类别取20%划分而非总体取20% 这是由于先前统计出某些类别数量非常稀少
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

random.seed(42)

# 配置路径
train_img_dir = r"D:\DeepLearning\Challenger\data\new_dataset_yolo\images\train"
train_lbl_dir = r"D:\DeepLearning\Challenger\data\new_dataset_yolo\labels\train"
val_img_dir = r"D:\DeepLearning\Challenger\data\new_dataset_yolo\images\val"
val_lbl_dir = r"D:\DeepLearning\Challenger\data\new_dataset_yolo\labels\val"

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

def get_file_info(img_dir, lbl_dir):
    """扫描目录，获取所有图片及其包含的层级类别信息"""
    file_info = []
    for img_filename in os.listdir(img_dir):
        if not img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(img_dir, img_filename)
        lbl_filename = os.path.splitext(img_filename)[0] + '.txt'
        lbl_path = os.path.join(lbl_dir, lbl_filename)
        
        if not os.path.exists(lbl_path):
            continue
            
        with open(lbl_path, 'r') as f:
            classes_in_file = set()
            sub_classes_in_file = [] # 用于存储所有解析出的小类ID，方便后续处理
            
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id_str = parts[0]
                    # 保存原始类别字符串
                    classes_in_file.add(class_id_str)
                    
                    # 解析小类ID (冒号后面的数字)
                    if ':' in class_id_str:
                        # 例如 "0:0" -> "0"
                        sub_id = class_id_str.split(':')[1]
                    else:
                        # 兼容不含冒号的旧格式
                        sub_id = class_id_str
                    
                    # 转换为整数以便后续排序和比较
                    try:
                        sub_classes_in_file.append(int(sub_id))
                    except ValueError:
                        pass # 忽略无法转换的格式

        # 核心修改点：使用小类ID作为分层抽样的依据
        # 策略：取该图片中包含的最小小类ID作为主类别
        # 这样可以确保即使是包含多个类别的图片，也有明确的归属桶
        main_class = min(sub_classes_in_file) if sub_classes_in_file else -1
        
        file_info.append({
            'img_filename': img_filename,
            'lbl_filename': lbl_filename,
            'img_path': img_path,
            'lbl_path': lbl_path,
            'main_class': main_class, # 这里存的是小类ID (int)
            'classes': classes_in_file # 这里存的是原始字符串 (set)
        })
    return file_info


print("\n正在扫描 train 目录...")
all_files = get_file_info(train_img_dir, train_lbl_dir)
print(f"共发现 {len(all_files)} 个文件待划分。")

# 按小类ID分组进行分层抽样
files_by_class = defaultdict(list)
for info in all_files:
    files_by_class[info['main_class']].append(info)

val_files = []
train_files = []

for cls_id, files in files_by_class.items():
    # cls_id 现在是冒号后面的数字
    random.shuffle(files)
    val_count = int(len(files) * 0.2)
           
    val_files.extend(files[:val_count])
    train_files.extend(files[val_count:])


print(f"\n划分结果: Train: {len(train_files)} 张, Val: {len(val_files)} 张")
print(f"验证集比例: {len(val_files) / len(all_files) * 100:.2f}%")

# 统计验证集中各类别（原始ID）的分布
val_class_dist = defaultdict(int)
for info in val_files:
    for cls in info['classes']:
        val_class_dist[cls] += 1

print("\n验证集各类别实例数量检查 (原始ID):")
# 按字符串排序输出
for cls_id in sorted(val_class_dist.keys()):
    print(f"  类别 {cls_id}: {val_class_dist[cls_id]} 个实例")

moved_count = 0
for info in val_files:
    # 移动图片
    if os.path.exists(info['img_path']):
        shutil.move(info['img_path'], os.path.join(val_img_dir, info['img_filename']))
    # 移动标签
    if os.path.exists(info['lbl_path']):
        shutil.move(info['lbl_path'], os.path.join(val_lbl_dir, info['lbl_filename']))
    moved_count += 1

print(f"成功移动 {moved_count} 对文件到 val 目录。")
