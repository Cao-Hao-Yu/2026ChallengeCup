import os
import json
import cv2
import numpy as np

# ================= 配置路径 =================
IMG_DIR = r"D:\DeepLearning\Challenger\data\dataset_yolo\images\val"
LABEL_DIR = r"D:\DeepLearning\Challenger\data\dataset_yolo\labels\val"
PRED_JSON_PATH = r"D:\DeepLearning\Challenger\code\ultralytics-main\runs\detect\val_26me200_pretrained\predictions.json"
SAVE_DIR = r"D:\DeepLearning\Challenger\code\ultralytics-main\runs\visualize_box"

# ================= 辅助函数 =================

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_yolo_box(box, img_w, img_h):
    """YOLO格式 (归一化中心点) 转 像素坐标 (x_min, y_min, x_max, y_max)"""
    x_center, y_center, w, h = box
    w_pixel = w * img_w
    h_pixel = h * img_h
    x_center_pixel = x_center * img_w
    y_center_pixel = y_center * img_h
    
    x_min = int(x_center_pixel - w_pixel / 2)
    y_min = int(y_center_pixel - h_pixel / 2)
    x_max = int(x_center_pixel + w_pixel / 2)
    y_max = int(y_center_pixel + h_pixel / 2)
    
    return x_min, y_min, x_max, y_max

def get_pred_box(box):
    """预测格式 [x, y, w, h] (左上角+宽高) 转 (x_min, y_min, x_max, y_max)"""
    x_min, y_min, w, h = box
    x_max = int(x_min + w)
    y_max = int(y_min + h)
    return int(x_min), int(y_min), x_max, y_max

def draw_label_left(img, text, point, font_scale, thickness, color):
    """
    在点(point)的左侧绘制文本标签，带有黑色边框以提高对比度
    point: 边界框的左上角坐标 (x, y)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 获取文本大小
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = point
    
    # --- 计算文本位置 ---
    # 目标：文本框的右下角紧贴边框的左上角
    # text_x = x - text_w
    # text_y = y (OpenCV绘制文本的基线在y下方，所以需要调整)
    
    # 坐标微调，让文字对齐看起来更舒服
    text_x = x - text_w - 2 
    text_y = y + text_h - 2 # 基线位置调整

    # --- 边界检查 ---
    # 如果左侧空间不足，就画在框的左上角内部
    if text_x < 0:
        text_x = x + 2
        # 如果右侧也不足(极其罕见)，不做额外处理，允许遮挡
        text_y = y + text_h + 2 # 放在框内部上方一点

    # --- 绘制背景 (可选，这里用黑色描边代替，更省事且看清背景) ---
    # cv2.rectangle(img, (text_x, text_y - text_h), (text_x + text_w, text_y + baseline), (0,0,0), -1)
    
    # 绘制文字 (先画黑色描边，再画彩色字体，防止在复杂背景下看不清)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_label_right(img, text, point, font_scale, thickness, color):
    """
    在点(point)的右侧绘制文本标签
    point: 边界框的右上角坐标 (x_max, y_min)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = point
    
    # --- 计算文本位置 ---
    # 目标：文本框的左下角紧贴边框的右上角
    # 注意：putText的y坐标是基线
    text_x = x + 2
    text_y = y + text_h - 2

    # --- 边界检查 ---
    # 如果右侧空间不足 (text_x + text_w > img_width)，则画在框的右上角内部
    img_w = img.shape[1]
    if text_x + text_w > img_w:
        text_x = x - text_w - 2
        text_y = y + text_h + 2 # 放在框内部上方一点

    # 绘制文字
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

# ================= 主处理逻辑 =================

def main():
    make_dir(SAVE_DIR)

    print(f"正在读取预测文件: {PRED_JSON_PATH}")
    with open(PRED_JSON_PATH, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # 按图片ID分组
    pred_dict = {}
    for p in predictions:
        img_id = p['image_id']
        if img_id not in pred_dict:
            pred_dict[img_id] = []
        pred_dict[img_id].append(p)

    print(f"共读取到 {len(predictions)} 条预测结果，涉及 {len(pred_dict)} 张图片。")

    processed_count = 0
    
    for img_id, preds in pred_dict.items():
        file_name = preds[0]['file_name'] 
        base_name = os.path.basename(file_name)
        name_no_ext = os.path.splitext(base_name)[0]
        
        img_path = os.path.join(IMG_DIR, base_name)
        label_path = os.path.join(LABEL_DIR, name_no_ext + ".txt")

        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]

        # --- 绘制标签 ---
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0]) # 这里本来就是0-24，不需要动
                    box = list(map(float, parts[1:5]))
                    
                    x1, y1, x2, y2 = get_yolo_box(box, img_w, img_h)
                    
                    # 绿色框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # GT标签：画在框的【左侧】
                    label_text = f"GT-{cls_id}"
                    draw_label_left(img, label_text, (x1, y1), font_scale=0.5, thickness=1, color=(0, 255, 0))

        # --- 绘制预测 ---
        for p in preds:
            # >>>>>> 这里是修改的地方 <<<<<<
            # 假设JSON里的ID是1-25，减1后变为0-24，与YOLO标签一致
            cls_id = p['category_id'] - 1 
            
            score = p['score']
            box = p['bbox']
            
            x1, y1, x2, y2 = get_pred_box(box)
            
            # 红色框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Pred标签：画在框的【右侧】
            # 这里传入 (x2, y1) 即右上角坐标
            label_text = f"Pred-{cls_id} {score:.2f}"
            draw_label_right(img, label_text, (x2, y1), font_scale=0.5, thickness=1, color=(0, 0, 255))

        save_path = os.path.join(SAVE_DIR, base_name)
        cv2.imwrite(save_path, img)
        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"已处理 {processed_count} 张图片...")

    print(f"完成！所有结果已保存至: {SAVE_DIR}")


if __name__ == "__main__":
    main()
