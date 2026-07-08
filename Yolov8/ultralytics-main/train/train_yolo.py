from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO(r"D:\DeepLearning\Challenger\code\Yolov8\ultralytics-main\yolov8n.pt")
    model.train(
        data=r"D:\DeepLearning\Challenger\data\dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        name="test"     # D:\DeepLearning\Challenger\code\Yolov8\ultralytics-main\runs\detect\test
    )
