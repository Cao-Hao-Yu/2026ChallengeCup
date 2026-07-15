from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

MODEL_PATH = r"D:/DeepLearning/Challenger/code/ultralytics-main/runs/detect/me200/weights/best.pt"
DATA_YAML = r"D:/DeepLearning/Challenger/data/dataset_yolo/dataset.yaml"

CONF_THRES = 0.25

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THRES,
    image_size=640,
    device="cuda",
)
