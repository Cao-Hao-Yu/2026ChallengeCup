# 预训练权重下载地址 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    DEFAULT_CFG.save_dir = r"./runs/detect/what_is_this"
    model = YOLO(model=r"./models/unname.yaml")
    model.train(
        data=r"./train/dataset.yaml",
        epochs=300,
        imgsz=640,
        batch=24,
        device=0
    )