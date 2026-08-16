import os
import cv2
import numpy as np
import random
import math

# ================= 配置区域 =================

# 目标生成大图的数量
TARGET_IMAGE_COUNT = 50  # 例如生成50张大图

# 原数据集路径
ORIGINAL_IMAGE_DIRS = [
    r"D:\DeepLearning\Challenger\data\dataset_yolo\images\train",
    r"D:\DeepLearning\Challenger\data\dataset_yolo\images\val"
]
ORIGINAL_LABEL_DIRS = [
    r"D:\DeepLearning\Challenger\data\dataset_yolo\labels\train",
    r"D:\DeepLearning\Challenger\data\dataset_yolo\labels\val"
]

# 输出路径
OUTPUT_IMAGE_DIR = r"D:\DeepLearning\Challenger\data\dataset_10k\images\val"
OUTPUT_LABEL_DIR = r"D:\DeepLearning\Challenger\data\dataset_10k\labels\val"

# 拼接与增强参数
CANVAS_SIZE = 10000       # 大图尺寸 10k x 10k
MIN_PATCH_SIZE = 800      # 小图最小边长
MAX_PATCH_SIZE = 900      # 小图最大边长

# 是否启用数据增强 (旋转, 翻转)
ENABLE_AUGMENTATION = True 
# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_non_mar_data(image_dirs, label_dirs):
    """收集所有非MAR20开头的图像路径"""
    valid_pairs = []
    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        if not os.path.exists(img_dir): continue
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # 【修改处】统一转大写后判断是否以MAR开头，排除 MAR20_xxx.jpg
                if filename.upper().startswith('MAR'):
                    continue
                
                img_path = os.path.join(img_dir, filename)
                
                # 构造标签文件名
                label_filename = os.path.splitext(filename)[0] + ".txt"
                label_path = os.path.join(lbl_dir, label_filename)
                
                # 如果标签文件不存在，视为空标签
                if not os.path.exists(label_path):
                    label_path = None
                
                valid_pairs.append((img_path, label_path))
    return valid_pairs

def augment_image_and_labels(img, labels_str, target_w, target_h):
    """
    对图像进行Resize和增强，同时转换标签坐标
    增强: 随机Resize + 随机旋转(0, 90, 180, 270) + 随机翻转
    """
    # 1. Resize
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # 解析标签
    new_labels = []
    for line in labels_str:
        parts = line.strip().split()
        if len(parts) < 5: continue
        cls_id = parts[0]
        x, y, w, h = map(float, parts[1:5])
        new_labels.append([cls_id, x, y, w, h])
    
    if ENABLE_AUGMENTATION:
        # 2. 随机旋转 (只做90度倍数旋转)
        rotation_code = random.randint(0, 3) 
        
        if rotation_code != 0:
            if rotation_code == 1: # 90度
                img_resized = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)
                for lbl in new_labels:
                    lbl[1], lbl[2], lbl[3], lbl[4] = (1 - lbl[2]), lbl[1], lbl[4], lbl[3]
                    
            elif rotation_code == 2: # 180度
                img_resized = cv2.rotate(img_resized, cv2.ROTATE_180)
                for lbl in new_labels:
                    lbl[1], lbl[2] = 1 - lbl[1], 1 - lbl[2]
                    
            elif rotation_code == 3: # 270度
                img_resized = cv2.rotate(img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
                for lbl in new_labels:
                    lbl[1], lbl[2], lbl[3], lbl[4] = lbl[2], (1 - lbl[1]), lbl[4], lbl[3]

        # 3. 随机翻转 (水平翻转)
        if random.random() > 0.5:
            img_resized = cv2.flip(img_resized, 1)
            for lbl in new_labels:
                lbl[1] = 1 - lbl[1]
                
    current_h, current_w = img_resized.shape[:2]
    
    return img_resized, new_labels, current_w, current_h

def process_one_canvas(data_pairs, output_img_dir, output_lbl_dir, img_idx):
    """处理并生成一张大图"""
    
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    all_final_labels = []
    
    # 每次生成大图时，重新随机打乱数据顺序，实现跨大图重复采样
    shuffled_pairs = random.sample(data_pairs, len(data_pairs))
    
    current_x, current_y = 0, 0
    row_max_height = 0
    patches_used = 0
    
    for img_path, lbl_path in shuffled_pairs:
        if current_y >= CANVAS_SIZE:
            break
            
        img = cv2.imread(img_path)
        if img is None: continue
        
        labels_str = []
        if lbl_path and os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                labels_str = f.readlines()
        
        target_w = random.randint(MIN_PATCH_SIZE, MAX_PATCH_SIZE)
        target_h = random.randint(MIN_PATCH_SIZE, MAX_PATCH_SIZE)
        
        # 数据增强处理
        processed_img, processed_labels, patch_w, patch_h = augment_image_and_labels(
            img, labels_str, target_w, target_h
        )
        
        # 布局计算
        if current_x + patch_w > CANVAS_SIZE:
            current_x = 0
            current_y += row_max_height
            row_max_height = 0
            
        if current_y + patch_h > CANVAS_SIZE:
            continue

        # 粘贴图像
        end_y = min(current_y + patch_h, CANVAS_SIZE)
        end_x = min(current_x + patch_w, CANVAS_SIZE)
        actual_h = end_y - current_y
        actual_w = end_x - current_x
        
        canvas[current_y:end_y, current_x:end_x] = processed_img[:actual_h, :actual_w]
        
        # 标签坐标映射到大图
        for lbl in processed_labels:
            cls_id, x_norm, y_norm, w_norm, h_norm = lbl
            
            cx_pix = x_norm * actual_w
            cy_pix = y_norm * actual_h
            w_pix = w_norm * actual_w
            h_pix = h_norm * actual_h
            
            abs_cx = current_x + cx_pix
            abs_cy = current_y + cy_pix
            
            final_x = abs_cx / CANVAS_SIZE
            final_y = abs_cy / CANVAS_SIZE
            final_w = w_pix / CANVAS_SIZE
            final_h = h_pix / CANVAS_SIZE
            
            if 0 <= final_x <= 1 and 0 <= final_y <= 1:
                line_str = f"{cls_id} {final_x:.6f} {final_y:.6f} {final_w:.6f} {final_h:.6f}\n"
                all_final_labels.append(line_str)
        
        current_x += actual_w
        if actual_h > row_max_height:
            row_max_height = actual_h
            
        patches_used += 1

    # 保存结果
    out_filename = f"stitched_10k_{img_idx:04d}"
    out_img_path = os.path.join(output_img_dir, f"{out_filename}.jpg")
    out_lbl_path = os.path.join(output_lbl_dir, f"{out_filename}.txt")
    
    cv2.imwrite(out_img_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    with open(out_lbl_path, 'w') as f:
        f.writelines(all_final_labels)
        
    return patches_used

def main():
    ensure_dir(OUTPUT_IMAGE_DIR)
    ensure_dir(OUTPUT_LABEL_DIR)
    
    print("正在扫描数据集并排除 MAR20 数据...")
    all_pairs = load_non_mar_data(ORIGINAL_IMAGE_DIRS, ORIGINAL_LABEL_DIRS)
    print(f"可用原图数量: {len(all_pairs)}")
    
    if len(all_pairs) == 0:
        print("错误: 未找到图像。")
        return

    print(f"开始生成 {TARGET_IMAGE_COUNT} 张超大图...")
    
    for i in range(TARGET_IMAGE_COUNT):
        count = process_one_canvas(all_pairs, OUTPUT_IMAGE_DIR, OUTPUT_LABEL_DIR, i)
        print(f"[{i+1}/{TARGET_IMAGE_COUNT}] 完成，使用了 {count} 张小图。")

    print("全部任务完成！")

if __name__ == "__main__":
    main()
