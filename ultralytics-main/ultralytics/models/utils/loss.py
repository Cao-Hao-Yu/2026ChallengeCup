# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# 注释
# 注释掉的是原版
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher
from ultralytics.utils.ops import linear_sum_assignment

class DETRLoss(nn.Module):
    """DETR (DEtection TRansformer) Loss class for calculating various loss components.

    This class computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary losses for the DETR
    object detection model.

    Attributes:
        nc (int): Number of classes.
        loss_gain (dict[str, float]): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Whether to use FocalLoss.
        use_vfl (bool): Whether to use VarifocalLoss.
        use_uni_match (bool): Whether to use a fixed layer for auxiliary branch label assignment.
        uni_match_ind (int): Index of fixed layer to use if use_uni_match is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss | None): Focal Loss object if use_fl is True, otherwise None.
        vfl (VarifocalLoss | None): Varifocal Loss object if use_vfl is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
    ):
        """Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with preset cost gains. Supports auxiliary
        losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (dict[str, float], optional): Coefficients for different loss components.
            aux_loss (bool): Whether to use auxiliary losses from each decoder layer.
            use_fl (bool): Whether to use FocalLoss.
            use_vfl (bool): Whether to use VarifocalLoss.
            use_uni_match (bool): Whether to use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
            gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
            alpha (float): The balancing factor used to address class imbalance.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(
        self, pred_scores: torch.Tensor, targets: torch.Tensor, gt_scores: torch.Tensor, num_gts: int, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Compute classification loss based on predictions, target values, and ground truth scores.

        Args:
            pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
            targets (torch.Tensor): Target class indices with shape (B, N).
            gt_scores (torch.Tensor): Ground truth confidence scores with shape (B, N).
            num_gts (int): Number of ground truth objects.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing classification loss value.

        Notes:
            The function supports different classification loss types:
            - Varifocal Loss (if self.vfl is not None and num_gts > 0)
            - Focal Loss (if self.fl is not None)
            - BCE Loss (default fallback)
        """
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (N, 4).
            postfix (str, optional): String to append to the loss names for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing:
                - loss_bbox{postfix}: L1 loss between predicted and ground truth boxes, scaled by the bbox loss gain.
                - loss_giou{postfix}: GIoU loss between predicted and ground truth boxes, scaled by the giou loss gain.

        Notes:
            If no ground truth boxes are provided (empty list), zero-valued tensors are returned for both losses.
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # This function is for future RT-DETR Segment models
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        match_indices: list[tuple] | None = None,
        postfix: str = "",
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get auxiliary losses for intermediate decoder layers.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes from auxiliary layers.
            pred_scores (torch.Tensor): Predicted scores from auxiliary layers.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            match_indices (list[tuple], optional): Pre-computed matching indices.
            postfix (str, optional): String to append to loss names.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of auxiliary losses.
        """
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(match_indices: list[tuple]) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Extract batch indices, source indices, and destination indices from match indices.

        Args:
            match_indices (list[tuple]): List of tuples containing matched indices.

        Returns:
            batch_idx (tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx).
            dst_idx (torch.Tensor): Destination indices.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, match_indices: list[tuple]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            match_indices (list[tuple]): List of tuples containing matched indices.

        Returns:
            pred_assigned (torch.Tensor): Assigned predicted bounding boxes.
            gt_assigned (torch.Tensor): Assigned ground truth bounding boxes.
        """
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        postfix: str = "",
        match_indices: list[tuple] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculate losses for a single prediction layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            pred_scores (torch.Tensor): Predicted class scores.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.
            postfix (str, optional): String to append to loss names.
            match_indices (list[tuple], optional): Pre-computed matching indices.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of losses.
        """
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
            # **(self._get_loss_mask(masks, gt_mask, match_indices, postfix) if masks is not None and gt_mask is not None else {})
        }

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores, shape (L, B, N, C).
            batch (dict[str, Any]): Batch information containing cls, bboxes, and gt_groups.
            postfix (str, optional): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (dict[str, torch.Tensor]): Computed losses, including main and auxiliary (if enabled).

        Notes:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """Real-Time DEtection TRansformer (RT-DETR) Detection Loss class that extends the DETRLoss."""

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        # ###RTDETR##### start
        # 普通 RT-DETR 不使用该参数；保留参数是为了和分层 loss 调用接口一致。
        dn_base_scores: torch.Tensor | None = None,
        # ###RTDETR##### end
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass to compute detection loss with optional denoising loss."""
        _ = dn_base_scores
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss})

        return total_loss

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: list[torch.Tensor], dn_num_group: int, gt_groups: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get match indices for denoising."""
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), (
                    f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                )
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices


# ###RTDETR##### start
# 分层 RT-DETR 专用 loss：只由 HierRTDETRDecoder 触发，普通 RT-DETR 不会走到这里。
class HierRTDETRDetectionLoss(DETRLoss):
    """Real-Time DEtection TRansformer (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    # ###RTDETR##### start
    # nbc 是大类数量；当前任务为 3 个大类。
    def __init__(self, *args, nbc: int = 3, **kwargs):
        """Initialize RT-DETR hierarchical detection loss."""
        super().__init__(*args, **kwargs)
        self.nbc = nbc

    @staticmethod
    def split_hierarchical_class(raw_cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode mixed labels into base/spec labels.

        Examples:
            28 -> base 2, spec 8
            324 -> base 3, spec 24
            125 -> base 1, spec 25, which is missing-spec when nc=25
        """
        raw_cls = raw_cls.long()
        is_two_digit_spec = raw_cls >= 100
        base = torch.where(is_two_digit_spec, raw_cls // 100, raw_cls // 10)
        spec = torch.where(is_two_digit_spec, raw_cls % 100, raw_cls % 10)
        return base, spec

    def _get_loss_class_n(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        gt_scores: torch.Tensor,
        num_gts: int,
        num_classes: int,
        name_class: str,
        ignore_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute DETR classification loss for a custom class count."""
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, num_classes + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if ignore_mask is not None:
            valid_weight = (~ignore_mask.to(device=pred_scores.device)).to(pred_scores.dtype).unsqueeze(-1)
        else:
            valid_weight = None

        if self.fl and valid_weight is not None:
            labels = one_hot.float()
            if num_gts and self.vfl:
                weight = self.vfl.alpha * pred_scores.sigmoid().pow(self.vfl.gamma) * (1 - labels) + gt_scores * labels
                loss_cls = F.binary_cross_entropy_with_logits(
                    pred_scores.float(), gt_scores.float(), reduction="none"
                ) * weight
            else:
                loss_cls = F.binary_cross_entropy_with_logits(pred_scores, labels, reduction="none")
                pred_prob = pred_scores.sigmoid()
                p_t = labels * pred_prob + (1 - labels) * (1 - pred_prob)
                loss_cls *= (1.0 - p_t) ** self.fl.gamma
                if (self.fl.alpha > 0).any():
                    alpha = self.fl.alpha.to(device=pred_scores.device, dtype=pred_scores.dtype)
                    loss_cls *= labels * alpha + (1 - labels) * (1 - alpha)
            loss_cls = (loss_cls * valid_weight).mean(1).sum()
            loss_cls /= max(num_gts, 1) / nq
        elif self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores)
            if valid_weight is not None:
                loss_cls = loss_cls * valid_weight
            loss_cls = loss_cls.mean(1).sum()

        return loss_cls.squeeze() * self.loss_gain["class"]

    def _decode_hierarchical_batch(self, batch: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
        """Return a base-class batch and spec labels for hierarchical RT-DETR loss."""
        # 混合标签拆成大类和小类；小类缺失 25 在小类 loss 中记为 -1。
        gt_base, gt_spec = self.split_hierarchical_class(batch["cls"])
        base_batch = dict(batch)
        base_batch["cls"] = (gt_base - 1).clamp(min=0, max=self.nbc - 1)
        gt_spec = gt_spec.clone()
        gt_spec[gt_spec >= self.nc] = -1
        return base_batch, gt_spec

    def _get_hierarchical_match_indices(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_base_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_base_cls: torch.Tensor,
        gt_spec: torch.Tensor,
        gt_groups: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Match valid spec labels with the spec branch and missing spec labels with the base branch."""
        # 匈牙利匹配：有小类的目标用小类分类 cost，缺失小类的目标改用大类分类 cost。
        bs, nq, nc = pred_scores.shape
        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        pred_scores_flat = pred_scores.detach().view(-1, nc)
        pred_base_scores_flat = pred_base_scores.detach().view(-1, self.nbc)
        if self.matcher.use_fl:
            pred_scores_flat = pred_scores_flat.sigmoid()
            pred_base_scores_flat = pred_base_scores_flat.sigmoid()
        else:
            pred_scores_flat = F.softmax(pred_scores_flat, dim=-1)
            pred_base_scores_flat = F.softmax(pred_base_scores_flat, dim=-1)
        pred_bboxes_flat = pred_bboxes.detach().view(-1, 4)

        gt_base_cls = gt_base_cls.long().clamp(min=0, max=self.nbc - 1)
        gt_spec = gt_spec.long()
        valid_spec = (gt_spec >= 0) & (gt_spec < nc)
        cost_class = pred_scores_flat.new_zeros((bs * nq, gt_bboxes.shape[0]))

        if valid_spec.any():
            # 正常小类标签：按小类分支计算分类匹配代价。
            spec_prob = pred_scores_flat[:, gt_spec[valid_spec]]
            if self.matcher.use_fl:
                neg_cost = (1 - self.matcher.alpha) * (spec_prob**self.matcher.gamma) * (
                    -(1 - spec_prob + 1e-8).log()
                )
                pos_cost = self.matcher.alpha * ((1 - spec_prob) ** self.matcher.gamma) * (
                    -(spec_prob + 1e-8).log()
                )
                cost_class[:, valid_spec] = pos_cost - neg_cost
            else:
                cost_class[:, valid_spec] = -spec_prob

        missing_spec = ~valid_spec
        if missing_spec.any():
            # 小类缺失标签 25：不强行监督小类，改用大类分支计算匹配代价。
            base_prob = pred_base_scores_flat[:, gt_base_cls[missing_spec]]
            if self.matcher.use_fl:
                neg_cost = (1 - self.matcher.alpha) * (base_prob**self.matcher.gamma) * (
                    -(1 - base_prob + 1e-8).log()
                )
                pos_cost = self.matcher.alpha * ((1 - base_prob) ** self.matcher.gamma) * (
                    -(base_prob + 1e-8).log()
                )
                cost_class[:, missing_spec] = pos_cost - neg_cost
            else:
                cost_class[:, missing_spec] = -base_prob

        cost_bbox = (pred_bboxes_flat.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)
        cost_giou = 1.0 - bbox_iou(
            pred_bboxes_flat.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True
        ).squeeze(-1)
        cost = (
            self.matcher.cost_gain["class"] * cost_class
            + self.matcher.cost_gain["bbox"] * cost_bbox
            + self.matcher.cost_gain["giou"] * cost_giou
        )
        cost[cost.isnan() | cost.isinf()] = 0.0

        cost = cost.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(gt_groups, -1))]
        gt_offsets = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_offsets[k])
            for k, (i, j) in enumerate(indices)
        ]

    def _get_hierarchical_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_base_scores: torch.Tensor,
        batch: dict[str, Any],
        gt_spec: torch.Tensor,
        postfix: str = "",
        match_indices: list[tuple] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute base matching, bbox/GIoU, base-class loss, and valid spec-class loss."""
        gt_bboxes, gt_base_cls, gt_groups = batch["bboxes"], batch["cls"], batch["gt_groups"]
        if match_indices is None:
            match_indices = self._get_hierarchical_match_indices(
                pred_bboxes, pred_scores, pred_base_scores, gt_bboxes, gt_base_cls, gt_spec, gt_groups
            )

        idx, gt_idx = self._get_index(match_indices)
        # ###RTDETR##### start
        # 匈牙利匹配在 CPU 上完成，这里把索引移回预测张量所在设备。
        idx = (idx[0].to(pred_scores.device), idx[1].to(pred_scores.device))
        gt_idx = gt_idx.to(pred_scores.device)
        # ###RTDETR##### end
        pred_assigned, gt_assigned = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_assigned):
            gt_scores[idx] = bbox_iou(pred_assigned.detach(), gt_assigned, xywh=True).squeeze(-1)

        base_targets = torch.full((bs, nq), self.nbc, device=pred_base_scores.device, dtype=gt_base_cls.dtype)
        base_targets[idx] = gt_base_cls[gt_idx]
        # 大类分支对所有匹配目标计算 base_loss，权重系数为 0.3。
        loss_base = self._get_loss_class_n(
            pred_base_scores,
            base_targets,
            gt_scores,
            len(gt_assigned),
            self.nbc,
            f"loss_base{postfix}",
        )
        loss_base = loss_base * 0.3

        spec_targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_base_cls.dtype)
        spec_scores = torch.zeros([bs, nq], device=pred_scores.device)
        spec_ignore = torch.zeros([bs, nq], device=pred_scores.device, dtype=torch.bool)
        valid_spec = torch.zeros_like(gt_idx, dtype=torch.bool)
        if gt_idx.numel():
            # 小类分支只监督 0..nc-1；缺失小类的匹配 query 会被 ignore。
            assigned_spec = gt_spec[gt_idx]
            valid_spec = (assigned_spec >= 0) & (assigned_spec < self.nc)
            if valid_spec.any():
                spec_idx = (idx[0][valid_spec], idx[1][valid_spec])
                spec_targets[spec_idx] = assigned_spec[valid_spec]
                spec_scores[spec_idx] = gt_scores[spec_idx]
            missing_spec = ~valid_spec
            if missing_spec.any():
                spec_ignore[(idx[0][missing_spec], idx[1][missing_spec])] = True
        loss_class = self._get_loss_class_n(
            pred_scores,
            spec_targets,
            spec_scores,
            int(valid_spec.sum().item()),
            self.nc,
            f"loss_class{postfix}",
            ignore_mask=spec_ignore,
        )

        return {
            f"loss_class{postfix}": loss_class,
            **self._get_loss_bbox(pred_assigned, gt_assigned, postfix),
            f"loss_base{postfix}": loss_base,
        }

    def _get_hierarchical_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_base_scores: torch.Tensor,
        batch: dict[str, Any],
        gt_spec: torch.Tensor,
        match_indices: list[tuple] | None = None,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Compute auxiliary hierarchical RT-DETR losses."""
        # decoder 辅助层同样计算小类、大类、bbox、giou 四类 loss。
        loss = torch.zeros(4, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self._get_hierarchical_match_indices(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                pred_base_scores[self.uni_match_ind],
                batch["bboxes"],
                batch["cls"],
                gt_spec,
                batch["gt_groups"],
            )
        for aux_bboxes, aux_scores, aux_base_scores in zip(pred_bboxes, pred_scores, pred_base_scores):
            loss_ = self._get_hierarchical_loss(
                aux_bboxes, aux_scores, aux_base_scores, batch, gt_spec, postfix=postfix, match_indices=match_indices
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            loss[3] += loss_[f"loss_base{postfix}"]
        return {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
            f"loss_base_aux{postfix}": loss[3],
        }
    # ###RTDETR##### end

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        # ###RTDETR##### start
        # 分层 denoising 的大类 logits；普通 RT-DETR 该值为 None。
        dn_base_scores: torch.Tensor | None = None,
        # ###RTDETR##### end
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass to compute detection loss with optional denoising loss.

        Args:
            preds (tuple[torch.Tensor, torch.Tensor]): Tuple containing predicted bounding boxes and scores.
            batch (dict[str, Any]): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes.
            dn_scores (torch.Tensor, optional): Denoising scores.
            dn_meta (dict[str, Any], optional): Metadata for denoising.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing total loss and denoising loss if applicable.
        """
        # ###RTDETR##### start
        # 分层头输出 3 个预测张量时，先解码混合标签再计算分层 loss。
        if len(preds) == 3:
            pred_bboxes, pred_scores, pred_base_scores = preds
            base_batch, gt_spec = self._decode_hierarchical_batch(batch)
            self.device = pred_bboxes.device
            total_loss = self._get_hierarchical_loss(
                pred_bboxes[-1], pred_scores[-1], pred_base_scores[-1], base_batch, gt_spec
            )
            if self.aux_loss:
                total_loss.update(
                    self._get_hierarchical_loss_aux(
                        pred_bboxes[:-1], pred_scores[:-1], pred_base_scores[:-1], base_batch, gt_spec
                    )
                )
        else:
            pred_bboxes, pred_scores = preds
            total_loss = super().forward(pred_bboxes, pred_scores, batch)
        # ###RTDETR##### end

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            # ###RTDETR##### start
            # denoising 分支也按同一套分层标签规则计算 loss。
            if dn_base_scores is not None:
                base_batch, gt_spec = self._decode_hierarchical_batch(batch)
                dn_loss = self._get_hierarchical_loss(
                    dn_bboxes[-1],
                    dn_scores[-1],
                    dn_base_scores[-1],
                    base_batch,
                    gt_spec,
                    postfix="_dn",
                    match_indices=match_indices,
                )
                if self.aux_loss:
                    dn_loss.update(
                        self._get_hierarchical_loss_aux(
                            dn_bboxes[:-1],
                            dn_scores[:-1],
                            dn_base_scores[:-1],
                            base_batch,
                            gt_spec,
                            match_indices=match_indices,
                            postfix="_dn",
                        )
                    )
            else:
                dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            # ###RTDETR##### end
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss})

        return total_loss

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: list[torch.Tensor], dn_num_group: int, gt_groups: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get match indices for denoising.

        Args:
            dn_pos_idx (list[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (list[int]): List of integers representing number of ground truths per image.

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), (
                    f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                )
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices
