import json
from collections import defaultdict

def align_coco_annotations(val_json_path, pred_json_path, output_val_path, output_pred_path=None, flag=True):
    with open(val_json_path, 'r', encoding='utf-8') as f:
        coco_gt = json.load(f)
        
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        yolo_preds = json.load(f)

    # 1. 构建旧标注中 文件名 -> 旧标注信息 的映射字典
    old_images_dict = {img["file_name"]: img for img in coco_gt["images"]}
    old_annotations_by_img = defaultdict(list)
    for ann in coco_gt["annotations"]:
        old_annotations_by_img[ann["image_id"]].append(ann)

    # 2. 遍历预测结果，提取所有出现过的图片，重新分配统一的整数 image_id
    new_images = []
    new_annotations = []
    new_preds = []
    
    file_name_to_new_id = {}
    new_image_id_counter = 1
    new_ann_id_counter = 1

    pred_file_names = set([pred["file_name"] for pred in yolo_preds])

    for file_name in pred_file_names:
        new_img_id = new_image_id_counter
        file_name_to_new_id[file_name] = new_img_id
        new_image_id_counter += 1

        if file_name in old_images_dict:
            old_img_info = old_images_dict[file_name]
            new_images.append({
                "id": new_img_id,
                "file_name": file_name,
                "width": old_img_info["width"],
                "height": old_img_info["height"]
            })
            
            for old_ann in old_annotations_by_img[old_img_info["id"]]:
                new_ann = old_ann.copy()
                new_ann["id"] = new_ann_id_counter
                new_ann["image_id"] = new_img_id
                
                if "area" not in new_ann:
                    new_ann["area"] = new_ann["bbox"][2] * new_ann["bbox"][3]
                if "iscrowd" not in new_ann:
                    new_ann["iscrowd"] = 0
                new_annotations.append(new_ann)
                new_ann_id_counter += 1
        else:
            print(f"警告: 预测图片 {file_name} 在原始val.json中未找到对应标注。将作为无标注图片加入。")
            new_images.append({
                "id": new_img_id,
                "file_name": file_name,
                "width": 0,  
                "height": 0 
            })

    # 3. 处理预测结果，将字符串 image_id 替换为新的整数 image_id
    print("正在转换预测结果的 image_id 和 category_id...")
    for pred in yolo_preds:
        file_name = pred["file_name"]
        new_img_id = file_name_to_new_id.get(file_name)
        
        if new_img_id is not None:
            # 将 YOLO 预测结果中的 category_id 减去 1，使其从 0 开始
            original_cat_id = pred["category_id"]
            if flag:
                new_cat_id = original_cat_id - 1 if original_cat_id > 0 else 0
            else:
                new_cat_id = original_cat_id if original_cat_id > 0 else 0

            new_pred = {
                "image_id": new_img_id,  
                "category_id": new_cat_id,  # 使用从0开始的 category_id
                "bbox": pred["bbox"],    
                "score": pred["score"]
            }
            new_preds.append(new_pred)

    # 4. 组装并保存新的 val.json
    new_coco_gt = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco_gt["categories"] 
    }
    
    with open(output_val_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco_gt, f, ensure_ascii=False, indent=4)
    print(f"新的标注文件已保存至: {output_val_path}")

    if output_pred_path:
        with open(output_pred_path, 'w', encoding='utf-8') as f:
            json.dump(new_preds, f, ensure_ascii=False, indent=4)
        print(f"对齐后的预测文件已保存至: {output_pred_path}")


if __name__ == "__main__":
    original_val_json = "D:/DeepLearning/Challenger/data/dataset_10k/val.json"
    yolo_pred_json = "D:/DeepLearning/Challenger/code/ultralytics-main/runs/temp/predictions.json"
    
    new_val_json = "D:/DeepLearning/Challenger/code/ultralytics-main/runs/temp/new_val.json"
    new_pred_json = "D:/DeepLearning/Challenger/code/ultralytics-main/runs/temp/new_predictions.json"
    
    # 有一个手动调整的flag 转换sahi切片推理时需要改成false
    align_coco_annotations(original_val_json, yolo_pred_json, new_val_json, new_pred_json, True)
