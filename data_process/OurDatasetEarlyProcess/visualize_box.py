import os
import json
import cv2
import numpy as np

# ================= 配置路径 =================
# 图像文件夹路径
IMG_DIR = r"D:\DeepLearning\Challenger\data\dataset_yolo\images\val"
# 标签文件夹路径
LABEL_DIR = r"D:\DeepLearning\Challenger\data\dataset_yolo\labels\val"
# 预测结果JSON路径
PRED_JSON_PATH = r"D:\DeepLearning\Challenger\code\ultralytics-main\runs\detect\val_26me200_pretrained\predictions.json"
# 结果保存路径
SAVE_DIR = r"D:\DeepLearning\Challenger\code\ultralytics-main\runs\visualize_box"

# 类别数量 (用于生成不同颜色，虽然这里主要用两种颜色区分真值和预测)
NUM_CLASSES = 25

# ================= 辅助函数 =================

def make_dir(path):
    """如果文件夹不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_yolo_box(box, img_w, img_h):
    """
    将YOLO格式标签 转换为 像素坐标 (x_min, y_min, x_max, y_max)
    """
    x_center, y_center, w, h = box
    
    # 转换为实际像素坐标
    w_pixel = w * img_w
    h_pixel = h * img_h
    x_center_pixel = x_center * img_w
    y_center_pixel = y_center * img_h
    
    # 计算左上角和右下角坐标
    x_min = int(x_center_pixel - w_pixel / 2)
    y_min = int(y_center_pixel - h_pixel / 2)
    x_max = int(x_center_pixel + w_pixel / 2)
    y_max = int(y_center_pixel + h_pixel / 2)
    
    return x_min, y_min, x_max, y_max

def get_pred_box(box):
    """
    将预测格式 [x, y, w, h] (假设为COCO格式: 左上角x, 左上角y, 宽, 高)
    转换为 (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, w, h = box
    x_max = int(x_min + w)
    y_max = int(y_min + h)
    return int(x_min), int(y_min), x_max, y_max

# ================= 主处理逻辑 =================

def main():
    make_dir(SAVE_DIR)

    # 1. 读取 JSON 预测文件
    print(f"正在读取预测文件: {PRED_JSON_PATH}")
    with open(PRED_JSON_PATH, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # 2. 将预测结果按图片ID分组，方便查找
    # 结构: {"image_id": [pred_dict1, pred_dict2, ...]}
    pred_dict = {}
    for p in predictions:
        img_id = p['image_id']
        if img_id not in pred_dict:
            pred_dict[img_id] = []
        pred_dict[img_id].append(p)

    print(f"共读取到 {len(predictions)} 条预测结果，涉及 {len(pred_dict)} 张图片。")

    # 3. 遍历处理每张图片
    # 这里我们遍历预测结果中包含的图片，也可以选择遍历IMG_DIR下的所有图片
    processed_count = 0
    
    for img_id, preds in pred_dict.items():
        # 构造文件名 (假设json中的file_name字段不带路径，且标签文件名与图片名对应但后缀为.txt)
        # 注意：需要检查json中的file_name是否包含后缀，示例中有 .jpg
        file_name = preds[0]['file_name'] 
        
        # 处理可能的路径分隔符问题，确保只获取文件名
        base_name = os.path.basename(file_name)
        name_no_ext = os.path.splitext(base_name)[0]
        
        img_path = os.path.join(IMG_DIR, base_name)
        label_path = os.path.join(LABEL_DIR, name_no_ext + ".txt")

        # 检查图片是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图片未找到 {img_path}，跳过。")
            continue
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片 {img_path}，可能文件损坏，跳过。")
            continue
            
        img_h, img_w = img.shape[:2]

        # --- 绘制标签 (Ground Truth) ---
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    # YOLO格式: class_id x_center y_center w h (都是归一化的)
                    cls_id = int(parts[0])
                    box = list(map(float, parts[1:5]))
                    
                    x1, y1, x2, y2 = get_yolo_box(box, img_w, img_h)
                    
                    # 绘制绿色框 (Ground Truth)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 绘制类别标签
                    label_text = f"GT-{cls_id}"
                    cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- 绘制预测 ---
        for p in preds:
            # JSON格式: category_id, bbox [x, y, w, h], score
            # 注意：JSON中的category_id通常是从1开始的(如COCO标准)，但也可能是从0开始
            # 根据你的示例 category_id: 18，如果你的类别定义也是0-24，则直接使用
            # 如果是COCO格式，ID可能不是连续的0-24，这里直接显示ID
            cls_id = p['category_id']
            score = p['score']
            box = p['bbox']
            
            x1, y1, x2, y2 = get_pred_box(box)
            
            # 绘制红色框 (Prediction)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 绘制标签和置信度
            label_text = f"Pred-{cls_id} {score:.2f}"
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 保存结果
        save_path = os.path.join(SAVE_DIR, base_name)
        cv2.imwrite(save_path, img)
        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"已处理 {processed_count} 张图片...")

    print(f"完成！所有结果已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    main()
