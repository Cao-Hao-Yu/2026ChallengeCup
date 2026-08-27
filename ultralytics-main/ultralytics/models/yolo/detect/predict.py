# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torchvision
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo26n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    # def postprocess(self, preds, img, orig_imgs, **kwargs):
    #     """Post-process predictions and return a list of Results objects.

    #     This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
    #     further analysis.

    #     Args:
    #         preds (torch.Tensor): Raw predictions from the model.
    #         img (torch.Tensor): Processed input image tensor in model input format.
    #         orig_imgs (torch.Tensor | list): Original input images before preprocessing.
    #         **kwargs (Any): Additional keyword arguments.

    #     Returns:
    #         (list): List of Results objects containing the post-processed predictions.

    #     Examples:
    #         >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
    #         >>> results = predictor.predict("path/to/image.jpg")
    #         >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
    #     """
    #     save_feats = getattr(self, "_feats", None) is not None
    #     preds = nms.non_max_suppression(
    #         preds,
    #         self.args.conf,
    #         kwargs.pop("iou", self.args.iou),  # allow callers (e.g. TrackTrack loose-NMS recovery) to override IoU
    #         self.args.classes,
    #         self.args.agnostic_nms,
    #         max_det=self.args.max_det,
    #         nc=0 if self.args.task == "detect" else len(self.model.names),
    #         end2end=getattr(self.model, "end2end", False),
    #         rotated=self.args.task == "obb",
    #         return_idxs=save_feats,
    #     )

    #     if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
    #         orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

    #     if save_feats:
    #         obj_feats = self.get_obj_feats(self._feats, preds[1])
    #         preds = preds[0]

    #     results = self.construct_results(preds, img, orig_imgs, **kwargs)

    #     if save_feats:
    #         for r, f in zip(results, obj_feats):
    #             r.feats = f  # add object features to results

    #     return results
        
    # 注释
    # 重写原来的推理代码，执行切片推理
    # nms 比较简陋 从可视化图上看有很大改进空间 （有改进空间的也可能是模型）

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        all_preds = []

        for pred, offset, crop_shape in zip(
            preds,
            self._offsets,
            self._crop_shapes
        ):
            if len(pred) == 0:
                continue

            pred = pred.clone()

            # letter box 后的坐标 => 原始 tile 坐标 => 原图坐标
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], crop_shape)
            x_offset, y_offset = offset
            pred[:, [0, 2]] += x_offset
            pred[:, [1, 3]] += y_offset

            all_preds.append(pred)

        # 全局 nms
        if all_preds:
            boxes = torch.cat(all_preds, dim=0)

            b = boxes[:, :4]
            scores = boxes[:, 4]
            classes = boxes[:, 5]

            # class-aware NMS
            max_coord = b.max()
            class_offsets = classes * (max_coord + 1)
            shifted_boxes = b + class_offsets[:, None]

            keep = torchvision.ops.nms(shifted_boxes, scores, self.args.iou)
            final_pred = boxes[keep]
        else:
            final_pred = torch.zeros((0, 6), dtype=torch.float32)

        return [Results(self._orig_img, path="", names=self.model.names, boxes=final_pred)]
    
    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    # def construct_result(self, pred, img, orig_img, img_path):
    #     """Construct a single Results object from one image prediction.

    #     Args:
    #         pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
    #         img (torch.Tensor): Preprocessed image tensor used for inference.
    #         orig_img (np.ndarray): Original image before preprocessing.
    #         img_path (str): Path to the original image file.

    #     Returns:
    #         (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
    #     """
    #     pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    #     return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

    def construct_results(self, preds, img, orig_imgs, **kwargs):
        return self.postprocess(preds, img, orig_imgs, **kwargs)