from ultralytics import YOLO

MODEL_PATH = r"temp/full.pt"
DATA_YAML = r"./train/dataset.yaml"

# 注释
# 改过 validator 会直接输出指标
# 置信度阈值得调一个符合硬性指标的值
CONF_THRES = 0.5

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    model.val(
        save=False,
        save_json=False,
        data=DATA_YAML,
        conf=CONF_THRES,
        imgsz=960,
        split='val',
        plots=False,
        device=0
    )
