from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import box_iou

PRED_JSON = Path(r"D:\DeepLearning\Challenger\code\ultralytics-main\runs\detect\inference\test\labels\predictions.json")
DATASET_ROOT = Path(r"D:\DeepLearning\Challenger\data\new_dataset_test")
IMAGE_DIR = DATASET_ROOT / "images" / "val"
GT_LABEL_DIR = DATASET_ROOT / "labels" / "val"

NC = 25
NAMES = {
    0: "HM",
    1: "LQS",
    2: "QHS",
    3: "MS",
    4: "A1_SU-35",
    5: "A2_C-130",
    6: "A3_C-17",
    7: "A4_C-5",
    8: "A5_F-16",
    9: "A6_TU-160",
    10: "A7_E-3",
    11: "A8_B-52",
    12: "A9_P-3C",
    13: "A10_B-1B",
    14: "A11_E-8",
    15: "A12_TU-22",
    16: "A13_F-15",
    17: "A14_KC-135",
    18: "A15_F-22",
    19: "A16_FA-18",
    20: "A17_TU-95",
    21: "A18_KC-10",
    22: "A19_SU-34",
    23: "A20_SU-24",
    24: "FSC",
}

def split_hierarchical_class(raw_cls):
    raw_cls = torch.as_tensor(raw_cls, dtype=torch.long)
    is_two_digit_spec = raw_cls >= 100
    base = torch.where(is_two_digit_spec, raw_cls // 100, raw_cls // 10)
    spec = torch.where(is_two_digit_spec, raw_cls % 100, raw_cls % 10 )
    return base, spec

def load_gt_txt(txt_path: Path):
    if not txt_path.exists():
        return (np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float32))

    classes = []
    boxes = []

    with txt_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            values = line.split()
            if len(values) < 5:
                print(f"[WARNING] GT 格式异常: {txt_path}, line={line_no}")
                continue

            try:
                cls = int(float(values[0]))

                xc = float(values[1])
                yc = float(values[2])
                w = float(values[3])
                h = float(values[4])

            except ValueError:
                print(f"[WARNING] GT 解析失败: {txt_path}, line={line_no}")
                continue

            classes.append(cls)
            boxes.append([xc,yc,w,h,])

    if len(classes) == 0:
        return (np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float32))

    return (np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32),)

def process_gt_classes(raw_cls,boxes):
    # 这里逻辑照抄 validator
    if len(raw_cls) == 0:
        return (np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float32))

    raw_cls_tensor = torch.from_numpy(raw_cls)
    _, spec_cls = split_hierarchical_class(raw_cls_tensor)

    valid_mask = spec_cls < 25

    spec_cls = spec_cls[valid_mask]
    boxes = boxes[valid_mask.numpy()]

    return (spec_cls.numpy().astype(np.int64),boxes)

def yolo_xywhn_to_xyxy(boxes,image_width, image_height):
    if len(boxes) == 0:
        return np.empty((0, 4),dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32)

    xc = boxes[:, 0] * image_width
    yc = boxes[:, 1] * image_height
    w = boxes[:, 2] * image_width
    h = boxes[:, 3] * image_height

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return np.stack([x1, y1, x2, y2], axis=1)

def json_xywh_to_xyxy(boxes):
    # coco 标准左上宽高转 xyxy
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32)
    result = boxes.copy()
    result[:, 2] = (result[:, 0] + result[:, 2])
    result[:, 3] = (result[:, 1] + result[:, 3])

    return result

def load_predictions(json_path: Path):
    # 读取json
    if not json_path.exists():
        raise FileNotFoundError(f"Prediction JSON 不存在:\n{json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = {}

    for item_index, item in enumerate(data):
        if "file_name" not in item:
            print(f"[WARNING] 第 {item_index} 个 prediction 没有 file_name 跳过")
            continue

        file_name = item["file_name"]
        category_id = int(item["category_id"])
        bbox = item["bbox"]
        score = float(item["score"])

        if len(bbox) != 4:
            print(f"[WARNING] bbox 长度异常: {file_name}")
            continue

        predictions.setdefault(file_name, [])
        predictions[file_name].append({"cls": category_id, "bbox": bbox, "conf": score,})

    return predictions


def find_image(stem: str):
    suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP",]

    for suffix in suffixes:
        path = (IMAGE_DIR / f"{stem}{suffix}")

        if path.exists():
            return path

    return None

def evaluate():
    print(f"Prediction JSON : {PRED_JSON}")
    print(f"Dataset root    : {DATASET_ROOT}")
    print(f"Image directory : {IMAGE_DIR}")
    print(f"GT directory    : {GT_LABEL_DIR}")

    if not PRED_JSON.exists():
        raise FileNotFoundError(
            f"Prediction JSON 不存在:\n{PRED_JSON}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"图片目录不存在:\n{IMAGE_DIR}")

    if not GT_LABEL_DIR.exists():
        raise FileNotFoundError(
            f"GT label 目录不存在:\n{GT_LABEL_DIR}")
    
    # 复用改过的 validator 逻辑
    # 可能会产生bug
    validator = DetectionValidator(dataloader=None, save_dir=None, args=None)
    validator.nc = NC
    validator.names = NAMES
    validator.metrics.names = NAMES
    validator.iouv = torch.linspace(0.5, 0.95, 10 )
    validator.niou = (validator.iouv.numel())

    validator.training = False

    validator.metrics.clear_stats()
    validator.metrics.clear_image_metrics()


    validator.custom_tp = np.zeros(NC, dtype=np.float64)
    validator.custom_fp = np.zeros(NC, dtype=np.float64)
    validator.custom_gt = np.zeros(NC, dtype=np.float64)

    predictions = load_predictions(PRED_JSON)

    gt_files = sorted(GT_LABEL_DIR.glob("*.txt"))

    print(f"GT label files  : {len(gt_files)}")
    print(f"Prediction imgs  : {len(predictions)}")

    seen = 0
    missing_prediction_images = 0
    missing_image_files = 0

    for gt_index, gt_path in enumerate(gt_files, start=1,):
        stem = gt_path.stem
        image_path = find_image(stem)

        if image_path is None:
            print(f"[WARNING] image not found: {stem}")
            missing_image_files += 1
            continue

        seen += 1

        # 需要读取原图尺寸转换标签坐标
        with Image.open(image_path) as im:
            image_width, image_height = (im.size)

        raw_gt_cls, gt_boxes_xywhn = (load_gt_txt(gt_path))
        gt_cls, gt_boxes_xywhn = (process_gt_classes(raw_gt_cls, gt_boxes_xywhn))
        gt_boxes_xyxy = (yolo_xywhn_to_xyxy(gt_boxes_xywhn, image_width, image_height))

        file_name = (image_path.name)
        pred_items = predictions.get(file_name,[])
        if file_name not in predictions:
            missing_prediction_images += 1

        pred_cls_list = []
        pred_boxes_list = []
        pred_conf_list = []

        for pred in pred_items:
            cls = int(pred["cls"])

            # prediction 已经是 spec
            if not (0 <= cls < NC):
                continue

            pred_cls_list.append(cls)
            pred_boxes_list.append(pred["bbox"])
            pred_conf_list.append(pred["conf"])

        gt_cls_tensor = torch.from_numpy(gt_cls).long()
        gt_boxes_tensor = torch.from_numpy(gt_boxes_xyxy).float()
        pred_cls_tensor = torch.tensor(pred_cls_list, dtype=torch.float32)
        pred_conf_tensor = torch.tensor(pred_conf_list, dtype=torch.float32)

        pred_boxes_xyxy = (json_xywh_to_xyxy(pred_boxes_list))
        pred_boxes_tensor = torch.from_numpy(pred_boxes_xyxy).float()

        if (gt_cls_tensor.numel() == 0 or pred_cls_tensor.numel() == 0):
            tp = np.zeros((len(pred_cls_list),validator.niou), dtype=bool)
        else:
            iou = box_iou(gt_boxes_tensor, pred_boxes_tensor)
            tp = (validator.match_predictions(pred_cls_tensor, gt_cls_tensor, iou).cpu().numpy())

        # 伪造 stat batch 然后调用 validator 接口
        stat_batch = {
            "tp": tp,
            "target_cls": gt_cls.copy(),
            "target_img": np.unique(gt_cls),
            "conf": np.asarray(pred_conf_list, dtype=np.float32,),
            "pred_cls": np.asarray(pred_cls_list, dtype=np.float32,),
            "im_name": file_name,
        }

        validator.metrics.update_stats(stat_batch)

        for c in gt_cls:
            c_int = int(c)
            if 0 <= c_int < NC:
                validator.custom_gt[c_int] += 1

        # tp_col = stat_batch["tp"][:, 0] 意味着切片数组只是用 iou=0.5 的结果
        tp_col = (stat_batch["tp"][:, 0] if stat_batch["tp"].shape[0] > 0 else np.zeros(0,dtype=bool,))
        pred_cls_np = (stat_batch["pred_cls"])

        for i, is_tp in enumerate(tp_col):
            p_c = int(pred_cls_np[i])

            if not (0 <= p_c < NC):
                continue

            if is_tp:
                validator.custom_tp[p_c] += 1
            else:
                validator.custom_fp[p_c] += 1

        if (gt_index % 100 == 0 or gt_index == len(gt_files)):
            print(f"\rEvaluating: {gt_index}/{len(gt_files)}", flush=True,)

    validator.seen = seen

    print(f"Seen images                : {seen}")
    print(f"GT label files             : {len(gt_files)}")
    print(f"Images without prediction  : {missing_prediction_images}")
    print(f"Images without image file  : {missing_image_files}")
    print(f"GT instances after parsing : {int(validator.custom_gt.sum())}")
    print(f"Prediction instances       : {int(validator.custom_tp.sum() + validator.custom_fp.sum())}")

    validator.metrics.process(save_dir=None,plot=False,)
    # 已被修改 会打印原生指标和比赛指标
    header = "%22s" + "%11s" * 6
    print(header % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)"))
    validator.print_results()

if __name__ == "__main__":
    evaluate()