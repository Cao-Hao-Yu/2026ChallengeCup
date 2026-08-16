from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# !?注注?!
# 可能需要跑很久 batch epochs 可能需要调一调
# 还有就是大类的损失系数 目前是 0.1 可能偏小 没有测试过其他值

if __name__ == "__main__":
    DEFAULT_CFG.save_dir = r"./runs/new_dataset/test"
    model = YOLO(model=r"models/yolo_test.yaml")
    model.train(
        data=r"./train/dataset.yaml",
        epochs=200,
        imgsz=960,
        batch=16,
        device=0
    )