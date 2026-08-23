from ultralytics import YOLO

MODEL_PATH = r"./runs/new_dataset/test/weights/best.pt"
TEST_IMG_DIR = r"D:\DeepLearning\Challenger\data\new_dataset_test\images\val"

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=TEST_IMG_DIR,
        conf=0.35,
        device=0,
        save=True,
        line_width=1,
        project="inference",
        name="test",
        exist_ok=True,
        save_txt=True,
        save_json=True,
        verbose=True
    )
