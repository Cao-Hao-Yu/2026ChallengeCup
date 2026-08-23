import torch
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# !?注注?!
# 可能需要跑很久 batch epochs 可能需要调一调
# 还有就是大类的损失系数 目前是 0.1 可能偏小 没有测试过其他值

# 加载预训练权重好像没什么用

if __name__ == "__main__":
    DEFAULT_CFG.save_dir = r"./runs/new_dataset/temp"

    model = YOLO(model=r"models/yolo_test.yaml")
    # 注掉以下代码即可不加载预训练权重
    ckpt = torch.load(r"models/yolov8n.pt", map_location="cpu", weights_only=False)
    pretrained_state_dict = ckpt["model"].state_dict() if "model" in ckpt else ckpt
    model_state_dict = model.model.state_dict()
    filtered_state_dict = {
        k: v for k, v in pretrained_state_dict.items() 
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    missing, unexpected = model.model.load_state_dict(filtered_state_dict, strict=False)

    print(f"All weights: {len(pretrained_state_dict)}")
    print(f"Filtered weights: {len(filtered_state_dict)}")
    print(f"Missing keys: {len(missing)}") 
    print(f"Unexpected keys: {len(unexpected)}")

    model.train(
        data=r"./train/dataset.yaml",
        epochs=5,
        imgsz=960,
        batch=16,
        device=0
    )