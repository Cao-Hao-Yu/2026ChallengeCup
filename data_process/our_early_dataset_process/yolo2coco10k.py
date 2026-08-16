import os
import json
from PIL import Image

# 1. 定义类别字典
class_names = {
    0: 'HM', 1: 'LQS', 2: 'QHS', 3: 'MS', 4: 'A1_SU-35', 
    5: 'A2_C-130', 6: 'A3_C-17', 7: 'A4_C-5', 8: 'A5_F-16', 9: 'A6_TU-160', 
    10: 'A7_E-3', 11: 'A8_B-52', 12: 'A9_P-3C', 13: 'A10_B-1B', 14: 'A11_E-8', 
    15: 'A12_TU-22', 16: 'A13_F-15', 17: 'A14_KC-135', 18: 'A15_F-22', 19: 'A16_FA-18', 
    20: 'A17_TU-95', 21: 'A18_KC-10', 22: 'A19_SU-34', 23: 'A20_SU-24', 24: 'FSC'
}

def yolo_labels_to_coco_json(labels_dir, images_dir, output_json_path, class_dict):
    """
    扫描 YOLO 标签文件，读取对应图片尺寸，并转换为 COCO 格式的 JSON。
    (不复制任何图片，仅生成标注文件)
    
    :param labels_dir: YOLO labels 文件夹路径 (如: .../labels/val)
    :param images_dir: YOLO images 文件夹路径 (如: .../images/val)
    :param output_json_path: 输出的 JSON 文件完整路径
    :param class_dict: 类别字典
    """
    if not os.path.exists(labels_dir):
        print(f"错误: 找不到标签目录 {labels_dir}")
        return

    # 2. 将类别字典转换为 COCO 的 categories 格式
    categories = [{"id": int(k), "name": v} for k, v in class_dict.items()]

    # 初始化 COCO 数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    image_id = 1
    annotation_id = 1
    
    # 获取目录下所有 .txt 标注文件
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    if not label_files:
        print(f"警告: 在 {labels_dir} 下未找到任何 .txt 标注文件。")
        return

    # 遍历转换
    for label_name in label_files:
        label_path = os.path.join(labels_dir, label_name)
        base_name = os.path.splitext(label_name)[0]
        
        # 尝试在 images_dir 中寻找对应的图片文件 (尝试常见后缀)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            temp_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        # 如果找不到图片，使用默认尺寸 1920x1080 (可根据你的数据集修改)
        if img_path is None:
            width, height = 10000, 10000
            file_name = base_name + ".jpg"
        else:
            with Image.open(img_path) as img:
                width, height = img.size
            file_name = os.path.basename(img_path)
        
        # 添加图像信息
        coco_data["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # 读取 YOLO 标注
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)

            # 将归一化的 YOLO 坐标转换为 COCO 绝对坐标 [x_min, y_min, w, h]
            abs_w = bbox_width * width
            abs_h = bbox_height * height
            x_min = (x_center * width) - (abs_w / 2.0)
            y_min = (y_center * height) - (abs_h / 2.0)
            area = abs_w * abs_h

            # 添加标注信息
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)],
                "area": round(area, 2),
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    # 保存 JSON 文件
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"\n转换完成！共处理 {image_id - 1} 个标注文件，生成 {annotation_id - 1} 个目标框。")
    print(f"JSON 文件已保存至: {output_json_path}")


if __name__ == '__main__':
    # 配置路径
    LABELS_DIR = r"D:\DeepLearning\Challenger\data\dataset_10k\labels\val"
    IMAGES_DIR = r"D:\DeepLearning\Challenger\data\dataset_10k\images\val"
    OUTPUT_DIR = r"D:\DeepLearning\Challenger\data\dataset_10k"
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, "val.json")
    
    yolo_labels_to_coco_json(
        labels_dir=LABELS_DIR, 
        images_dir=IMAGES_DIR,       # 传入图片目录用于自动读取宽高
        output_json_path=OUTPUT_JSON, 
        class_dict=class_names
    )
