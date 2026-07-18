import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

MODEL_PATH = r"./runs/detect/AKConv_ne200/weights/best.pt"
DATA_YAML = r"./train/dataset.yaml"

CONF_THRES = 0.001


def evaluate_with_ultralytics():
    model = YOLO(MODEL_PATH)
    
    metrics = model.val(
        save_json=False,
        data=DATA_YAML,
        conf=CONF_THRES,
        imgsz=640,
        split='val',
        plots=True,
        device=0
    )
    
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()
    
    # 用1-precision来近似的虚警率
    false_alarm_rate = 1 - precision
    
    print(f"conf={CONF_THRES}")
    print(f"total precision: {precision:.4f}")
    print(f"total recall: {recall:.4f}")
    print(f"FP rate (1 - Precision): {false_alarm_rate:.4f}")


if __name__ == "__main__":
    evaluate_with_ultralytics()
