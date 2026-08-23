import torch
from ultralytics import RTDETR
from ultralytics.utils import DEFAULT_CFG

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    DEFAULT_CFG.save_dir = r"./runs/new_dataset/rtdetr_test"
    
    model = RTDETR(model=r"models/rtdetr-test.yaml")
    ckpt = torch.load(r"models/rtdetr-l.pt", map_location="cpu", weights_only=False)
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
        epochs=2,
        imgsz=960,
        batch=4,
        device='cpu',
    )

# !?注注?!
# 这里只有训练的代码 验证的代码还没改 但是yolo的验证代码是改好的