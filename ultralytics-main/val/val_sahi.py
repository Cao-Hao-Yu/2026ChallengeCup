import json
import os
import time
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm

def validate_and_export_json(
    model_path,
    source_dir,
    output_json_path,
    confidence_threshold=0.3,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    device="cuda:0"
):
    # 1. 初始化模型
    print(f"Loading model from {model_path}...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    all_predictions = []
    
    # 图片后缀名过滤
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [
        f for f in os.listdir(source_dir) 
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    total_images = len(image_files)
    print(f"Found {total_images} images. Starting inference...")

    # 2. 遍历图片进行推理
    total_inference_time = 0.0  # 用于累计所有图片的推理时间

    # 使用 tqdm 包装 image_files 迭代器，添加进度条
    pbar = tqdm(image_files, desc="Inference Progress", unit="img")
    
    for image_name in pbar:
        image_path = os.path.join(source_dir, image_name)
        image_id = os.path.splitext(image_name)[0]
        
        try:
            # --- 开始测速 ---
            start_time = time.time()
            
            result = get_sliced_prediction(
                image_path,
                detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                verbose=False
            )
            
            # --- 结束测速 ---
            elapsed_time = time.time() - start_time
            total_inference_time += elapsed_time

            # 在进度条后实时显示当前图片的耗时
            pbar.set_postfix({'当前耗时': f"{elapsed_time:.3f}s"})
            
            # 3. 解析结果
            for prediction in result.object_prediction_list:
                # 获取bbox并转为float
                bbox = prediction.bbox.to_xywh()
                # 关键修改：使用 float() 强制转换，解决 float32 序列化报错
                bbox = [float(round(x, 3)) for x in bbox]
                
                # 获取类别和分数，同样转为原生 int 和 float
                category_id = int(prediction.category.id)
                score = float(round(prediction.score.value, 5))
                
                pred_item = {
                    "image_id": image_id,
                    "file_name": image_name,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score
                }
                all_predictions.append(pred_item)
                
        except Exception as e:
            print(f"\nError processing {image_name}: {e}") # 加换行符避免打断进度条
            continue

    # 4. 打印测速统计信息
    print("\n" + "="*40)
    print("⏱️ 测速统计:")
    print(f"总图片数: {total_images}")
    print(f"总推理耗时: {total_inference_time:.2f} 秒")
    if total_images > 0:
        avg_time_per_img = total_inference_time / total_images
        print(f"平均单张耗时: {avg_time_per_img:.4f} 秒/张")
    print("="*40 + "\n")

    # 5. 保存为JSON文件
    print(f"Inference finished. Writing results to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    
    print("Done!")

# ================= 使用示例 =================
if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        "model_path": "D:/DeepLearning/Challenger/code/ultralytics-main/runs/detect/26me200/weights/best.pt",
        "source_dir": "D:/DeepLearning/Challenger/data/dataset_10k/images/val/",
        "output_json_path": "D:/DeepLearning/Challenger/code/ultralytics-main/runs/temp/predictions.json",
        "device": "cuda:0",
        "confidence_threshold": 0.25,
        "slice_height": 800,
        "slice_width": 800,
        "overlap_height_ratio": 0.2,
        "overlap_width_ratio": 0.2
    }

    # 检查文件夹是否存在
    if not os.path.exists(CONFIG['source_dir']):
        print(f"请确保图片文件夹存在: {CONFIG['source_dir']}")
    else:
        validate_and_export_json(**CONFIG)
