from ultralytics import YOLO
# 预训练权重下载地址 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

if __name__ == "__main__":
    # model = YOLO(model=r"D:/DeepLearning/Challenger/code/ultralytics-main/ultralytics/cfg/models/26/yolo26m.yaml")
    # model = YOLO(model=r"D:/DeepLearning/Challenger/code/ultralytics-main/ultralytics/cfg/models/v8/yolov8m.yaml")
    model = YOLO("D:/DeepLearning/Challenger/code/ultralytics-main/models/yolov8m.pt")
    model.train(
        data=r"D:/DeepLearning/Challenger/data/dataset_yolo/dataset.yaml",
        epochs=200,
        imgsz=640,
        batch=18,
        name="8me200_pretrained"
    )