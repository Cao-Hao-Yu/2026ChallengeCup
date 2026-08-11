import json
import os

# 定义输入输出路径
input_json_path = r"D:\DeepLearning\Challenger\data\LargerDataset\VHRShips\labels\annotations.json"
output_dir = r"D:\DeepLearning\Challenger\data\LargerDataset\VHRShips\hbblabels"

# 图像尺寸 (题目给定所有图像均为 1280x720)
IMG_WIDTH = 1280
IMG_HEIGHT = 720

def convert_bbox_to_yolo(bbox):
    """
    将像素坐标 xywh (左上角x, 左上角y, w, h) 
    转换为 YOLO 格式 (中心点x归一化, 中心点y归一化, w归一化, h归一化)
    """
    x, y, w, h = bbox
    
    # 计算中心点坐标 (像素)
    x_center = x + w / 2
    y_center = y + h / 2
    
    # 归一化
    x_center /= IMG_WIDTH
    y_center /= IMG_HEIGHT
    w_norm = w / IMG_WIDTH
    h_norm = h / IMG_HEIGHT
    
    return [x_center, y_center, w_norm, h_norm]

def process_annotations():
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在读取: {input_json_path}")
    
    # 1. 读取JSON文件
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return

    # 数据有效性检查：data 必须是一个列表
    if not isinstance(data, list):
        print("错误: JSON根节点不是列表。请检查文件格式。")
        # 尝试打印实际类型以便调试
        print(f"实际类型: {type(data)}")
        return

    # 用于收集所有出现的类别名称
    all_classes = set()

    # 2. 第一次遍历：收集所有类别名称
    for image_data in data:
        # 每一项应该是包含 'image' 和 'annotations' 的字典
        if not isinstance(image_data, dict):
            continue
            
        annotations = image_data.get('annotations', [])
        if isinstance(annotations, list):
            for ann in annotations:
                if isinstance(ann, dict) and 'category' in ann:
                    all_classes.add(ann['category'])

    # 排序类别 (保证顺序一致)
    sorted_classes = sorted(list(all_classes))
    
    # 3. 创建 classes.txt 文件
    classes_file_path = os.path.join(output_dir, 'classes.txt')
    with open(classes_file_path, 'w', encoding='utf-8') as f:
        for cls in sorted_classes:
            f.write(f"{cls}\n")
    print(f"找到 {len(sorted_classes)} 个类别，已保存至 classes.txt")

    # 4. 第二次遍历：生成YOLO标注文件
    count = 0
    for image_data in data:
        # 安全检查：确保外层是字典
        if not isinstance(image_data, dict):
            continue
        
        image_name = image_data.get('image')
        if not image_name:
            continue
            
        annotations = image_data.get('annotations', [])
        # 安全检查：确保annotations是列表
        if not isinstance(annotations, list):
            continue

        # 准备写入的内容
        yolo_lines = []
        
        for ann in annotations:
            # 安全检查：确保每个标注是字典
            if not isinstance(ann, dict):
                continue
                
            category = ann.get('category')
            bbox = ann.get('bbox')
            
            # 确保数据完整
            if category and bbox and isinstance(bbox, list) and len(bbox) == 4:
                # 转换坐标
                yolo_bbox = convert_bbox_to_yolo(bbox)
                
                # 题目要求：class id使用json中的类别名称代替
                # 格式：类别名称 x_center y_center width height
                line = f"{category} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                yolo_lines.append(line)
        
        # 写入文件
        if yolo_lines:
            # 构造输出文件名: BV002.jpg -> BV002.txt
            base_name = os.path.splitext(image_name)[0]
            output_file_path = os.path.join(output_dir, f"{base_name}.txt")
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            
            count += 1
            # 每处理100张打印一次进度
            if count % 100 == 0:
                print(f"已处理 {count} 张图片...")

    print(f"转换完成！共生成 {count} 个标注文件。保存在: {output_dir}")

if __name__ == "__main__":
    process_annotations()
