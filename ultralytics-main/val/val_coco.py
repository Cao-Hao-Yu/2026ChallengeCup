import argparse
import json
import os
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(pred_json, anno_json):
    print(f"正在评估 COCO 指标，使用 {pred_json} 和 {anno_json}...")
    
    # 检查文件是否存在
    for x in [pred_json, anno_json]:
        assert os.path.isfile(x), f"文件 {x} 不存在"
    
    # 初始化COCO API
    anno = COCO(str(anno_json))
    pred = anno.loadRes(str(pred_json))
    
    # 进行bbox评估
    eval_bbox = COCOeval(anno, pred, 'bbox')
    eval_bbox.evaluate()
    eval_bbox.accumulate()
    eval_bbox.summarize()


if __name__ == '__main__':
    pred_json = Path("D:/DeepLearning/Challenger/code/ultralytics-main/runs/temp/new_predictions.json")
    anno_json = Path("D:/DeepLearning/Challenger/code/ultralytics-main/runs/temp/new_val.json")
    
    stats = evaluate_coco(pred_json, anno_json)