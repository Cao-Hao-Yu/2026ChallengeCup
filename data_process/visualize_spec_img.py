import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


"""
    可以检查一下yolo格式标注框是否正确
    AI代码
"""

TARGET_CLASS_ID = 0
NUM_SAMPLES = 9

# 配置路径
img_dir = r"D:\DeepLearning\Challenger\data\dataset_yolo\images\train"
lbl_dir = r"D:\DeepLearning\Challenger\data\dataset_yolo\labels\train"

# 类别名称字典
class_names = {
    0: 'HM', 1: 'LQS', 2: 'QHS', 3: 'MS', 4: 'A1_SU-35', 
    5: 'A2_C-130', 6: 'A3_C-17', 7: 'A4_C-5', 8: 'A5_F-16', 9: 'A6_TU-160', 
    10: 'A7_E-3', 11: 'A8_B-52', 12: 'A9_P-3C', 13: 'A10_B-1B', 14: 'A11_E-8', 
    15: 'A12_TU-22', 16: 'A13_F-15', 17: 'A14_KC-135', 18: 'A15_F-22', 19: 'A16_FA-18', 
    20: 'A17_TU-95', 21: 'A18_KC-10', 22: 'A19_SU-34', 23: 'A20_SU-24', 24: 'FSC'
}

# 为不同类别生成区分颜色
def get_color(idx):
    idx = idx * 3
    r = (idx * 50) % 256
    g = (idx * 80 + 80) % 256
    b = (idx * 110 + 160) % 256
    return (r / 255.0, g / 255.0, b / 255.0)

# 获取所有标签文件并打乱顺序，以便每次运行看到不同的图
all_lbls = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
import random
random.shuffle(all_lbls)

# 提取包含目标类别的图片
matched_files = []
print(f"正在搜索包含类别 [{TARGET_CLASS_ID} - {class_names.get(TARGET_CLASS_ID, 'Unknown')}] 的图片...")

for lbl_name in all_lbls:
    lbl_path = os.path.join(lbl_dir, lbl_name)
    
    # 先快速读取txt文件检查是否包含目标ID，不用先读图，速度更快
    has_target = False
    with open(lbl_path, 'r') as f:
        for line in f:
            if int(line.strip().split()[0]) == TARGET_CLASS_ID:
                has_target = True
                break
                
    if has_target:
        # 找到对应的图片文件名
        img_name = os.path.splitext(lbl_name)[0] + '.jpg'
        img_path = os.path.join(img_dir, img_name)
        
        if os.path.exists(img_path):
            matched_files.append((img_path, lbl_path))
            if len(matched_files) >= NUM_SAMPLES:
                break  # 找够了就停止搜索

if not matched_files:
    print(f"未找到包含类别 {TARGET_CLASS_ID} 的图片。")
else:
    print(f"已找到 {len(matched_files)} 张图片，正在生成可视化...")

# 创建画布
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i, (img_path, lbl_path) in enumerate(matched_files):
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    ax = axes[i]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(os.path.basename(img_path), fontsize=9)
    
    # 读取并绘制标签 (绘制该图中的所有标签，但目标类别高亮)
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                
                x_min = (cx - bw / 2) * w
                y_min = (cy - bh / 2) * h
                box_w = bw * w
                box_h = bh * h
                
                # 如果是目标类别，使用高亮颜色(红色)和粗线条；其他类别使用灰色细线条作为参考
                if cls_id == TARGET_CLASS_ID:
                    color = 'red'
                    lw = 2
                    label_txt = class_names.get(cls_id, str(cls_id))
                else:
                    color = 'gray'
                    lw = 1
                    label_txt = None # 非目标类别不显示文字，避免干扰
                
                rect = patches.Rectangle((x_min, y_min), box_w, box_h,
                                         linewidth=lw, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                if label_txt:
                    ax.text(x_min, y_min - 5, label_txt, color='white', fontsize=8,
                            bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', pad=1))

# 隐藏多余的子图
for j in range(len(matched_files), 9):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
