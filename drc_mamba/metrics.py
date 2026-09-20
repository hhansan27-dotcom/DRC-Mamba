"""Evaluation metrics for infrared small target detection.

Definitions follow the metric implementation used in the experiments:
- IoU: global pixel IoU over the full evaluation set.
- nIoU: mean per-image IoU.
- Pd: target-level detection probability using one-to-one connected-component
  matching with centroid distance < ``centroid_tol`` pixels.
- Fa: pixels belonging to unmatched predicted connected components per 1e6 pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure

_EPS = 1e-6


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Expected a 2D/3D/4D tensor, got shape={tuple(tensor.shape)}")


def _to_binary_numpy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    from_logits: bool,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = _ensure_4d(predictions.detach())
    targets = _ensure_4d(targets.detach())

    if predictions.shape[-2:] != targets.shape[-2:]:
        predictions = F.interpolate(
            predictions,
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    predictions = predictions[:, :1]
    targets = targets[:, :1]
    probabilities = torch.sigmoid(predictions) if from_logits else predictions

    pred_mask = (probabilities > threshold).squeeze(1).cpu().numpy().astype(np.uint8)
    target_mask = (targets > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
    return pred_mask, target_mask


def _component_statistics(
    prediction: np.ndarray,
    target: np.ndarray,
    centroid_tol: float,
    connectivity: int,
) -> tuple[int, int, int]:
    target_labels = measure.label(target.astype(np.uint8), connectivity=connectivity)
    prediction_labels = measure.label(prediction.astype(np.uint8), connectivity=connectivity)
    target_components = measure.regionprops(target_labels)
    prediction_components = measure.regionprops(prediction_labels)

    matched_predictions: set[int] = set()
    correct_targets = 0

    for target_component in target_components:
        target_centroid = np.asarray(target_component.centroid, dtype=np.float32)
        best_index: int | None = None
        best_distance = float("inf")

        for index, prediction_component in enumerate(prediction_components):
            if index in matched_predictions:
                continue
            prediction_centroid = np.asarray(prediction_component.centroid, dtype=np.float32)
            distance = float(np.linalg.norm(target_centroid - prediction_centroid))
            if distance < best_distance:
                best_distance = distance
                best_index = index

        if best_index is not None and best_distance < centroid_tol:
            correct_targets += 1
            matched_predictions.add(best_index)

    false_alarm_pixels = sum(
        int(component.area)
        for index, component in enumerate(prediction_components)
        if index not in matched_predictions
    )
    return len(target_components), correct_targets, false_alarm_pixels


@dataclass(frozen=True)
class MetricResult:
    iou: float
    niou: float
    pd: float
    fa: float

    def as_dict(self) -> dict[str, float]:
        return {"IoU": self.iou, "nIoU": self.niou, "Pd": self.pd, "Fa": self.fa}


class IRSTDMetrics:
    """Accumulate IoU, nIoU, Pd, and Fa over an evaluation set."""

    def __init__(
        self,
        threshold: float = 0.5,
        centroid_tol: float = 3.0,
        connectivity: int = 2,
        from_logits: bool = True,
        empty_iou: float = 1.0,
    ) -> None:
        self.threshold = float(threshold)
        self.centroid_tol = float(centroid_tol)
        self.connectivity = int(connectivity)
        self.from_logits = bool(from_logits)
        self.empty_iou = float(empty_iou)
        self.reset()

    def reset(self) -> None:
        self.total_intersection = 0
        self.total_union = 0
        self.per_image_iou: list[float] = []
        self.total_targets = 0
        self.correct_targets = 0
        self.false_alarm_pixels = 0
        self.total_pixels = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pred_masks, target_masks = _to_binary_numpy(
            predictions,
            targets,
            threshold=self.threshold,
            from_logits=self.from_logits,
        )

        for prediction, target in zip(pred_masks, target_masks):
            intersection = int(np.logical_and(prediction, target).sum())
            union = int(np.logical_or(prediction, target).sum())
            self.total_intersection += intersection
            self.total_union += union
            self.per_image_iou.append(
                self.empty_iou if union == 0 else intersection / (union + _EPS)
            )

            total_targets, correct_targets, false_alarm_pixels = _component_statistics(
                prediction,
                target,
                centroid_tol=self.centroid_tol,
                connectivity=self.connectivity,
            )
            self.total_targets += total_targets
            self.correct_targets += correct_targets
            self.false_alarm_pixels += false_alarm_pixels
            self.total_pixels += int(prediction.size)

    def compute(self) -> MetricResult:
        iou = self.total_intersection / (self.total_union + _EPS)
        niou = float(np.mean(self.per_image_iou)) if self.per_image_iou else 0.0
        pd = self.correct_targets / (self.total_targets + _EPS)
        fa = (self.false_alarm_pixels / (self.total_pixels + _EPS)) * 1e6
        return MetricResult(iou=iou, niou=niou, pd=pd, fa=fa)
