# 预训练权重下载地址 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    DEFAULT_CFG.save_dir = r"./runs/new_dataset/8me200"
    model = YOLO(r"D:\DeepLearning\Challenger\code\ultralytics-main\models\yolov8m.pt")
    model.train(
        data=r"./train/dataset.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        device=0
    )