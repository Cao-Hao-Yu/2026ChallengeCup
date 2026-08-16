import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

MODEL_PATH = r"./runs/new_dataset/test/weights/best.pt"
DATA_YAML = r"./train/dataset.yaml"

# !?注注?!
# 改过 validator 会直接输出指标
# 置信度阈值得调一个符合硬性指标的值

CONF_THRES = 0.35

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    model.val(
        save_json=True,
        data=DATA_YAML,
        conf=CONF_THRES,
        imgsz=960,
        split='val',
        plots=True,
        device=0
    )
