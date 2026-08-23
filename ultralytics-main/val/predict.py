from ultralytics import YOLO

MODEL_PATH = r"./runs/new_dataset/test/weights/best.pt"
TEST_IMG_DIR = r"D:\DeepLearning\Challenger\data\new_dataset_test\images\val"

# !?注注?!
# 切图推理的思路为 切图 => 预测 => 拼接
# 非常普通的算法 也许有改进空间?
# 配合 val_prediction 使用 如果官方提供结构更好
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
