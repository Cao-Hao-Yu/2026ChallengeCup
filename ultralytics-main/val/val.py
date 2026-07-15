import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


MODEL_PATH = r"D:/DeepLearning/Challenger/code/ultralytics-main/runs/detect/me200/weights/best.pt"
DATA_YAML = r"D:/DeepLearning/Challenger/data/dataset_yolo/dataset.yaml"

CONF_THRES = 0.25


def evaluate_with_ultralytics():
    model = YOLO(MODEL_PATH)
    
    metrics = model.val(
        save_json=True,
        data=DATA_YAML,
        conf=CONF_THRES,
        imgsz=640,
        split='val',
        plots=True,
    )
    
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()
    
    # 用1-precision来近似的虚警率
    false_alarm_rate = 1 - precision
    
    print(f"\nconf={CONF_THRES}")
    print(f"total precision: {precision:.4f}")
    print(f"total recall: {recall:.4f}")
    print(f"FP rate (1 - Precision): {false_alarm_rate:.4f}")


if __name__ == "__main__":
    evaluate_with_ultralytics()
    # D:/DeepLearning/Challenger/code/ultralytics-main/runs/detect/val
