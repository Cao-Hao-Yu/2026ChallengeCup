import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


MODEL_PATH = r"D:\DeepLearning\Challenger\code\Yolov8\ultralytics-main\runs\detect\test\weights\best.pt"
DATA_YAML = r"D:\DeepLearning\Challenger\data\dataset.yaml"

VAL_IMG_DIR = r"D:\DeepLearning\Challenger\data\images\val"
VAL_LBL_DIR = r"D:\DeepLearning\Challenger\data\labels\val"

# 置信度阈值
CONF_THRES = 0.5


def evaluate_with_ultralytics():
    model = YOLO(MODEL_PATH)
    
    metrics = model.val(
        data=DATA_YAML,
        conf=CONF_THRES,
        imgsz=640,
        split='val',
        plots=True
    )
    
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()
    
    # 用1-precision来近似的虚警率
    false_alarm_rate = 1 - precision
    
    print(f"\n conf={CONF_THRES}")
    print(f"total precision: {precision:.4f}")
    print(f"total recall: {recall:.4f}")
    print(f"FP rate (1 - Precision): {false_alarm_rate:.4f}")
    
if __name__ == "__main__":
    evaluate_with_ultralytics()
    # D:\DeepLearning\Challenger\code\Yolov8\ultralytics-main\runs\detect\val
