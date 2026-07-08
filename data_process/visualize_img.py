import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
    可视化 用于检查yolo格式标注框
    AI代码
"""

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows自带黑体
plt.rcParams['axes.unicode_minus'] = False

# 配置路径
img_dir = r"D:\DeepLearning\Challenger\data\dataset_yolo\images\val"
lbl_dir = r"D:\DeepLearning\Challenger\data\dataset_yolo\labels\val"

# 类别名称字典
class_names = {
    0: 'HM', 1: 'LQS', 2: 'QHS', 3: 'MS', 4: 'A1_SU-35', 
    5: 'A2_C-130', 6: 'A3_C-17', 7: 'A4_C-5', 8: 'A5_F-16', 9: 'A6_TU-160', 
    10: 'A7_E-3', 11: 'A8_B-52', 12: 'A9_P-3C', 13: 'A10_B-1B', 14: 'A11_E-8', 
    15: 'A12_TU-22', 16: 'A13_F-15', 17: 'A14_KC-135', 18: 'A15_F-22', 19: 'A16_FA-18', 
    20: 'A17_TU-95', 21: 'A18_KC-10', 22: 'A19_SU-34', 23: 'A20_SU-24', 24: 'FSC'
}

# 为不同类别生成一些区分颜色 (BGR格式转RGB)
def get_color(idx):
    idx = idx * 3  # 让颜色差异大一点
    b = (idx * 50) % 256
    g = (idx * 80 + 80) % 256
    r = (idx * 110 + 160) % 256
    return (r / 255.0, g / 255.0, b / 255.0)  # matplotlib 使用 0-1 的 RGB

# 1. 获取所有符合条件的图片文件名
all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]

# 2. 随机抽取 9 张图片
sample_imgs = random.sample(all_imgs, min(9, len(all_imgs)))

# 3. 创建画布
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i, img_name in enumerate(sample_imgs):
    img_path = os.path.join(img_dir, img_name)
    lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + '.txt')
    
    # 读取图片 (opencv读取为BGR，需转为RGB供matplotlib显示)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    # 在对应的子图上绘制
    ax = axes[i]
    ax.imshow(img)
    ax.axis('off') # 隐藏坐标轴
    ax.set_title(img_name, fontsize=10)
    
    # 4. 读取并绘制 YOLO 标签
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    
                    # YOLO归一化坐标 -> 像素绝对坐标
                    # 转换为左上角坐标和长宽
                    x_min = (cx - bw / 2) * w
                    y_min = (cy - bh / 2) * h
                    box_w = bw * w
                    box_h = bh * h
                    
                    # 获取颜色和类别名
                    color = get_color(cls_id)
                    label = class_names.get(cls_id, str(cls_id))
                    
                    # 绘制矩形框
                    rect = patches.Rectangle((x_min, y_min), box_w, box_h,
                                             linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    
                    # 在框左上角写类别名称
                    ax.text(x_min, y_min - 5, label, color='white', fontsize=8,
                            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1))

# 如果图片不足9张，隐藏多余的子图
for j in range(len(sample_imgs), 9):
    axes[j].axis('off')

plt.tight_layout()
# 弹窗展示
plt.show()
