import os
import json
import yaml
import shutil
from PIL import Image
from tqdm import tqdm

"""
    将yolo格式数据集转换为coco格式 并将图像文件复制到输出目录。
    json中id标从0开始0-24共25个类别 不是从1开始的!!!

    原格式
    D:\DeepLearning\Challenger\data\dataset_yolo
    ├── images
    │   ├── train
    │   └── val
    ├── labels
    │   ├── train
    │   └── val
    └── dataset.yaml

    转换后格式
    D:\DeepLearning\Challenger\data\dataset_coco
    │
    ├── train.json
    ├── val.json
    │
    └── images
        ├── train
        │   ├── image_name1.jpg
        │   ├── image_name2.jpg
        │   └── ...
        └── val
            ├── image_name3.jpg
            ├── image_name4.jpg
            └── ...
    
    代码copy自网上 根据本数据集做了一些修改
"""


def yolo_to_coco_with_images(yolo_dir, output_dir):
    """
    :param yolo_dir: YOLO格式数据集的根目录 包含images、labels文件夹和dataset.yaml。
    :param output_dir: 输出的COCO格式数据集根目录。
    """
    # 1. 读取 dataset.yaml 配置文件
    yaml_path = os.path.join(yolo_dir, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"在 {yolo_dir} 下未找到 dataset.yaml 文件！")
        
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
        
    # 2. 获取类别字典并转换为 COCO 的 categories 格式
    names_dict = data_yaml.get('names', {})
    categories = [{"id": int(k), "name": v} for k, v in names_dict.items()]
    
    # 3. 确定数据集根目录
    yaml_path_str = data_yaml.get('path', '')
    base_dir = yaml_path_str

    # 4. 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 5. 分别处理 train 和 val 数据集
    splits = {'train': data_yaml.get('train', 'images/train'), 
              'val': data_yaml.get('val', 'images/val')}
    
    for split_name, img_rel_path in splits.items():
        print(f"\n--- 开始处理 {split_name} 数据集")
        
        # 源路径
        src_images_dir = os.path.join(base_dir, img_rel_path)
        labels_rel_path = img_rel_path.replace('images', 'labels')
        src_labels_dir = os.path.join(base_dir, labels_rel_path)
        
        if not os.path.exists(src_images_dir):
            print(f"警告: 找不到源图像目录 {src_images_dir}，跳过 {split_name}。")
            continue
        if not os.path.exists(src_labels_dir):
            print(f"警告: 找不到源标签目录 {src_labels_dir}，跳过 {split_name}。")
            continue

        # 目标路径 (图片复制到 output_dir/images/train 或 val)
        dest_images_dir = os.path.join(output_dir, 'images', split_name)
        os.makedirs(dest_images_dir, exist_ok=True)

        # 初始化 COCO 数据结构
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": categories
        }

        image_id = 1
        annotation_id = 1
        
        # 获取目录下所有图片文件
        image_files = [f for f in os.listdir(src_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for image_name in tqdm(image_files, desc=f"转换并复制 {split_name}"):
            src_image_path = os.path.join(src_images_dir, image_name)
            with Image.open(src_image_path) as img:
                width, height = img.size

            # 复制图像到新目录
            dest_image_path = os.path.join(dest_images_dir, image_name)
            shutil.copy2(src_image_path, dest_image_path)

            # 添加图像信息到 json
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height
            })

            # 读取对应的标注文件
            label_name = os.path.splitext(image_name)[0] + '.txt'
            src_label_path = os.path.join(src_labels_dir, label_name)
            

            with open(src_label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)

                # 将归一化坐标转换为绝对坐标
                x_center *= width
                y_center *= height
                bbox_width *= width
                bbox_height *= height

                # COCO 格式: [x_min, y_min, width, height]
                x_min = x_center - (bbox_width / 2.0)
                y_min = y_center - (bbox_height / 2.0)
                area = bbox_width * bbox_height

                # 添加标注信息
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [round(x_min, 2), round(y_min, 2), round(bbox_width, 2), round(bbox_height, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0
                })

                annotation_id += 1

            image_id += 1

        output_json_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=4)

        print(f"{split_name} 转换完成！包含 {len(coco_data['images'])} 张图像，{len(coco_data['annotations'])} 个标注。")
        print(f"JSON 文件已保存到: {output_json_path}")
        print(f"图像文件已复制到: {dest_images_dir}")


if __name__ == '__main__':
    # 原始 YOLO 数据集路径
    yolo_dataset_dir = r"D:\DeepLearning\Challenger\data\dataset_yolo"
    
    # 输出的 COCO 数据集目标路径
    coco_output_dir = r"D:\DeepLearning\Challenger\data\dataset_coco"
    
    yolo_to_coco_with_images(yolo_dataset_dir, coco_output_dir)
