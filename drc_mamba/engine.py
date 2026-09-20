"""Training and evaluation engine for DRC-Mamba."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .losses import DRCMambaLoss
from .metrics import IRSTDMetrics, MetricResult
from .utils import AverageMeter, save_checkpoint


class WarmupCosineAnnealingLR(_LRScheduler):
    """Linear warm-up followed by cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.min_lr = float(min_lr)
        self.warmup_start_lr = float(warmup_start_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.warmup_epochs == 0:
            progress = min(self.last_epoch / max(1, self.max_epochs), 1.0)
            return [
                self.min_lr
                + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]

        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]

        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.max_epochs - self.warmup_epochs
        )
        progress = min(progress, 1.0)
        return [
            self.min_lr
            + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


def morphological_edge(mask: torch.Tensor) -> torch.Tensor:
    """Generate a one-pixel morphological edge target."""
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return ((dilated - eroded) > 0.5).float().detach()


def outer_band(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    return ((dilated - mask).clamp(0.0, 1.0) > 0.5).float().detach()


def erode_with_fallback(mask: torch.Tensor, radius: int, min_keep_pixels: int) -> torch.Tensor:
    if radius <= 0:
        return mask.detach()
    kernel_size = 2 * radius + 1
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = (eroded > 0.5).float()
    output = eroded.clone()
    areas = eroded.flatten(1).sum(dim=1)
    for index in range(mask.shape[0]):
        if areas[index] < min_keep_pixels:
            output[index : index + 1] = mask[index : index + 1]
    return output.detach()


def ramp_by_epoch(epoch: int, max_epoch: int, start: float, end: float) -> float:
    """Piecewise-linear ramp in [0, 1] over fractional training progress."""
    if max_epoch <= 1:
        return 1.0
    progress = float(epoch) / float(max(1, max_epoch - 1))
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / max(1e-6, end - start)


def curriculum_scales(epoch: int, max_epoch: int, cfg: TrainConfig) -> dict[str, float]:
    return {
        "gate": ramp_by_epoch(epoch, max_epoch, cfg.gate_ramp_start, cfg.gate_ramp_end),
        "support_sem": ramp_by_epoch(
            epoch, max_epoch, cfg.support_ramp_start, cfg.support_ramp_end
        ),
        "clutter_sem": ramp_by_epoch(
            epoch, max_epoch, cfg.clutter_ramp_start, cfg.clutter_ramp_end
        ),
        "rejection_sem": ramp_by_epoch(epoch, max_epoch, cfg.rejection_ramp_start, cfg.rejection_ramp_end),
        "shrink": ramp_by_epoch(epoch, max_epoch, cfg.shrink_ramp_start, cfg.shrink_ramp_end),
    }


def update_gate_schedule(model: nn.Module, epoch: int, max_epoch: int, cfg: TrainConfig) -> None:
    progress = ramp_by_epoch(epoch, max_epoch, cfg.gate_ramp_start, cfg.gate_ramp_end)
    for module in model.modules():
        setter = getattr(module, "set_progress", None)
        if callable(setter):
            setter(progress, 1.0)


def unpack_outputs(outputs: dict[str, Any]):
    """Extract the prediction, edge head, auxiliary logits, stage packs, and extras."""
    if not isinstance(outputs, dict):
        raise TypeError("DRC-Mamba is expected to return a dictionary of outputs.")
    prediction = outputs.get("mask")
    edge = outputs.get("edge")
    aux = [outputs[key] for key in ("aux1", "aux2") if outputs.get(key) is not None]
    packs = outputs.get("packs")
    excluded = {"mask", "edge", "aux1", "aux2", "packs"}
    extras = {key: value for key, value in outputs.items() if key not in excluded and value is not None}
    return prediction, edge, aux, packs, extras


def sparse_hard_selector(
    score: torch.Tensor,
    background_mask: torch.Tensor,
    topk_ratio: float = 0.010,
    min_keep: int = 16,
    pool_kernel: int = 5,
) -> torch.Tensor:
    """Keep sparse high-risk background responses for auxiliary supervision."""
    score = (score * background_mask).clamp(0.0, 1.0)
    if pool_kernel > 1:
        pooled = F.max_pool2d(
            score, kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2
        )
        score = score * (score >= pooled).float()

    flat_score = score.flatten(1)
    flat_background = background_mask.flatten(1)
    outputs: list[torch.Tensor] = []
    for batch_index in range(score.shape[0]):
        valid = torch.nonzero(flat_background[batch_index] > 0.5, as_tuple=False).squeeze(1)
        if valid.numel() == 0:
            outputs.append(torch.zeros_like(score[batch_index : batch_index + 1]))
            continue
        k = min(max(min_keep, int(valid.numel() * topk_ratio)), valid.numel())
        values = flat_score[batch_index, valid]
        top_values, top_positions = torch.topk(values, k=k, largest=True)
        keep_indices = valid[top_positions]
        keep = torch.zeros_like(flat_score[batch_index])
        keep[keep_indices] = (top_values > 1e-6).float()
        outputs.append(keep.view_as(score[batch_index]).unsqueeze(0))
    return torch.cat(outputs, dim=0)


def _resize_optional(value: torch.Tensor | None, size: tuple[int, int]) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape[-2:] != size:
        value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
    return value


def build_refined_hard_background(
    packs: dict[str, Any] | None,
    extras: dict[str, torch.Tensor],
    labels: torch.Tensor,
    pred_final: torch.Tensor | None,
    pred_threshold: float,
) -> dict[str, torch.Tensor]:
    """Build sparse connected/repeated/halo hard-background targets."""
    background = 1.0 - labels
    outer = outer_band(labels, 3)
    size = labels.shape[-2:]

    bg_struct = _resize_optional(extras.get("bg_struct"), size)
    recon_error = _resize_optional(extras.get("bg_recon_err"), size)
    bg_sparse = _resize_optional(extras.get("bg_sparse"), size)
    rejection = _resize_optional(extras.get("rejection512"), size)
    ring_response = _resize_optional(extras.get("ring_response512"), size)

    line_map = None
    connected_map = None
    sparse_rejection = None
    if isinstance(packs, dict):
        for stage_name in ("p1", "p2", "p3", "p4"):
            stage = packs.get(stage_name)
            if not isinstance(stage, dict):
                continue
            if line_map is None:
                line_map = stage.get("line")
            if connected_map is None:
                connected_map = stage.get("conn_bg")
            if sparse_rejection is None:
                sparse_rejection = stage.get("sparse_rejection4")

    line_map = _resize_optional(line_map, size)
    connected_map = _resize_optional(connected_map, size)
    sparse_rejection = _resize_optional(sparse_rejection, size)

    zeros = torch.zeros_like(labels)
    bg_struct = zeros if bg_struct is None else bg_struct
    recon_error = zeros if recon_error is None else recon_error
    bg_sparse = zeros if bg_sparse is None else bg_sparse
    rejection = zeros if rejection is None else rejection
    ring_response = zeros if ring_response is None else ring_response
    line_map = zeros if line_map is None else line_map
    connected_map = zeros if connected_map is None else connected_map
    sparse_rejection = zeros if sparse_rejection is None else sparse_rejection

    connected_score = torch.clamp(
        0.35 * bg_struct + 0.25 * line_map + 0.20 * connected_map + 0.20 * rejection,
        0.0,
        1.0,
    )
    repeat_score = torch.clamp(
        0.35 * recon_error + 0.20 * bg_sparse + 0.20 * bg_struct + 0.25 * sparse_rejection,
        0.0,
        1.0,
    )

    halo_score = outer.clone()
    if pred_final is not None:
        pred_prob = torch.sigmoid(pred_final)
        if pred_prob.shape[-2:] != size:
            pred_prob = F.interpolate(pred_prob, size=size, mode="bilinear", align_corners=False)
        late_false_positive = ((pred_prob > pred_threshold).float() * background).detach()
        halo_score = torch.clamp(
            halo_score + 0.35 * late_false_positive + 0.30 * ring_response, 0.0, 1.0
        )

    connected_bg = sparse_hard_selector(connected_score, background, 0.008, 16)
    repeat_bg = sparse_hard_selector(repeat_score, background, 0.008, 16)
    halo_bg = sparse_hard_selector(halo_score, background, 0.010, 16)
    hard_bg = torch.clamp(connected_bg + repeat_bg + halo_bg, 0.0, 1.0)
    return {
        "conn_bg": connected_bg.detach(),
        "repeat_bg": repeat_bg.detach(),
        "halo_bg": halo_bg.detach(),
        "hard_bg": hard_bg.detach(),
    }


def semantic_pack_loss(
    packs: dict[str, Any] | None,
    labels: torch.Tensor,
    support_scale: float,
    clutter_scale: float,
    rejection_scale: float,
    refined_hard: dict[str, torch.Tensor],
) -> torch.Tensor:
    if not isinstance(packs, dict):
        return labels.new_tensor(0.0)
    if support_scale <= 0.0 and clutter_scale <= 0.0 and rejection_scale <= 0.0:
        return labels.new_tensor(0.0)

    stage_weights = {
        "p1": (0.48, 0.00, 0.00),
        "p2": (0.24, 0.10, 0.04),
        "p3": (0.16, 0.14, 0.10),
        "p4": (0.08, 0.18, 0.22),
    }
    total = labels.new_tensor(0.0)
    count = 0

    for stage_name in ("p1", "p2", "p3", "p4"):
        stage = packs.get(stage_name)
        if not isinstance(stage, dict) or not all(
            key in stage for key in ("support", "clutter", "rejection")
        ):
            continue

        size = stage["support"].shape[-2:]
        gt = F.interpolate(labels, size=size, mode="nearest")
        near_target = outer_band(gt, 3)
        hard = {
            key: F.interpolate(value, size=size, mode="bilinear", align_corners=False)
            if value.shape[-2:] != size
            else value
            for key, value in refined_hard.items()
        }
        conn_bg = hard["conn_bg"]
        repeat_bg = hard["repeat_bg"]
        halo_bg = hard["halo_bg"]
        hard_bg = hard["hard_bg"]

        if stage_name == "p1":
            clutter_target = torch.zeros_like(gt)
            rejection_target = torch.zeros_like(gt)
        elif stage_name == "p2":
            clutter_target = conn_bg
            rejection_target = torch.clamp(0.65 * conn_bg + 0.35 * halo_bg, 0.0, 1.0)
        elif stage_name == "p3":
            clutter_target = torch.clamp(0.55 * conn_bg + 0.45 * repeat_bg, 0.0, 1.0)
            rejection_target = torch.clamp(
                0.25 * conn_bg + 0.45 * repeat_bg + 0.30 * halo_bg, 0.0, 1.0
            )
        else:
            clutter_target = torch.clamp(0.40 * repeat_bg + 0.60 * hard_bg, 0.0, 1.0)
            rejection_target = torch.clamp(0.55 * hard_bg + 0.45 * halo_bg, 0.0, 1.0)

        support = stage["support"].clamp(1e-4, 1.0 - 1e-4)
        clutter = stage["clutter"].clamp(1e-4, 1.0 - 1e-4)
        rejection = stage["rejection"].clamp(1e-4, 1.0 - 1e-4)

        support_weight = 1.0 + 0.20 * near_target
        clutter_weight = 1.0 + 0.80 * conn_bg + 1.00 * repeat_bg + 1.20 * halo_bg
        rejection_weight = 1.0 + 0.80 * conn_bg + 1.10 * repeat_bg + 1.30 * halo_bg

        support_loss = (
            F.binary_cross_entropy(support, gt, reduction="none") * support_weight
        ).mean()
        clutter_loss = (
            F.binary_cross_entropy(clutter, clutter_target, reduction="none") * clutter_weight
        ).mean()
        rejection_loss = (
            F.binary_cross_entropy(rejection, rejection_target, reduction="none") * rejection_weight
        ).mean()

        ws, wc, wv = stage_weights[stage_name]
        total = total + (
            ws * support_scale * support_loss
            + wc * clutter_scale * clutter_loss
            + wv * rejection_scale * rejection_loss
        )
        count += 1

    return labels.new_tensor(0.0) if count == 0 else total / count


def hard_background_pack_loss(
    packs: dict[str, Any] | None,
    labels: torch.Tensor,
    pred_final: torch.Tensor,
    refined_hard: dict[str, torch.Tensor],
    pred_threshold: float,
) -> tuple[torch.Tensor | None, float | None, float | None]:
    if not isinstance(packs, dict):
        return None, None, None

    total = labels.new_tensor(0.0)
    count = 0
    hard_ratio = 0.0
    hard_fp = 0.0

    for stage_name, stage_scale in (("p2", 0.35), ("p3", 0.80), ("p4", 1.00)):
        stage = packs.get(stage_name)
        if not isinstance(stage, dict) or not all(key in stage for key in ("clutter", "rejection")):
            continue

        size = stage["clutter"].shape[-2:]
        gt = F.interpolate(labels, size=size, mode="nearest")
        near_target = outer_band(gt, 3)
        hard = {
            key: F.interpolate(value, size=size, mode="bilinear", align_corners=False)
            if value.shape[-2:] != size
            else value
            for key, value in refined_hard.items()
        }
        conn_bg = hard["conn_bg"]
        repeat_bg = hard["repeat_bg"]
        halo_bg = hard["halo_bg"]
        hard_bg = hard["hard_bg"]

        clutter = stage["clutter"].clamp(1e-4, 1.0 - 1e-4)
        rejection = stage["rejection"].clamp(1e-4, 1.0 - 1e-4)
        clutter_target = torch.clamp(0.55 * conn_bg + 0.45 * repeat_bg, 0.0, 1.0)
        rejection_target = torch.clamp(
            0.20 * near_target + 0.35 * halo_bg + 0.45 * hard_bg, 0.0, 1.0
        )

        clutter_weight = 1.0 + 0.90 * conn_bg + 1.10 * repeat_bg + 1.20 * halo_bg
        rejection_weight = 1.0 + 0.70 * near_target + 1.10 * halo_bg + 1.30 * hard_bg
        clutter_loss = (
            F.binary_cross_entropy(clutter, clutter_target, reduction="none") * clutter_weight
        ).mean()
        rejection_loss = (
            F.binary_cross_entropy(rejection, rejection_target, reduction="none") * rejection_weight
        ).mean()
        total = total + stage_scale * (0.55 * clutter_loss + rejection_loss)
        count += 1

        pred_prob = torch.sigmoid(
            F.interpolate(pred_final, size=size, mode="bilinear", align_corners=False)
        )
        hard_ratio += float(hard_bg.mean().detach().cpu())
        hard_fp += float(((pred_prob > pred_threshold).float() * hard_bg).mean().detach().cpu())

    if count == 0:
        return None, None, None
    return total / count, hard_ratio / count, hard_fp / count


class Trainer:
    """Paper-aligned DRC-Mamba training loop."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
        run_dir: str | Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.cfg = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = DRCMambaLoss(
            iou_weight=1.0,
            focal_weight=1.0,
            edge_weight=config.edge_weight,
            bnd_weight=config.bnd_weight,
            bnd_from_logits=True,
            bnd_bg_weight=config.bnd_bg_weight,
            bnd_bg_dilate_ks=config.bnd_bg_dilate_ks,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            outer_weight=0.0,
            inner_weight=0.0,
            tail_weight=0.0,
            outer_r=config.outer_r,
            inner_r=config.inner_r,
            far_r=config.far_r,
            tail_k=config.tail_k,
            outer_margin=config.outer_margin,
            inner_margin=config.inner_margin,
            halo_weight=0.0,
            support_weight=config.support_head_weight,
            center_weight=config.center_head_weight,
            interference_weight=config.interference_head_weight,
            boundary_rejection_weight=config.boundary_rejection_head_weight,
            confidence_weight=config.confidence_head_weight,
            rejection_weight=config.rejection_head_weight,
            risk_weight=config.risk_head_weight,
            radial_weight=0.0,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=config.warmup_epochs,
            max_epochs=config.epochs,
            min_lr=config.min_lr,
            warmup_start_lr=config.warmup_start_lr,
        )
        self.metrics = IRSTDMetrics(
            threshold=config.threshold,
            centroid_tol=config.centroid_tol,
            from_logits=True,
        )
        self.best_iou = -1.0
        self.best_niou = -1.0
        self.start_epoch = 0
        self.history_path = self.run_dir / "history.csv"
        if not self.history_path.exists():
            with self.history_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "lr", "train_loss", "val_loss", "IoU", "nIoU", "Pd", "Fa"])

    def restore_optimizer(self, checkpoint: dict[str, Any]) -> None:
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1

    def _configure_epoch(self, epoch: int) -> tuple[dict[str, float], float, float, float]:
        cfg = self.cfg
        phase = curriculum_scales(epoch, cfg.epochs, cfg)
        update_gate_schedule(self.model, epoch, cfg.epochs, cfg)

        edge_scale = min(1.0, epoch / max(1, cfg.edge_warmup))
        boundary_scale = min(1.0, epoch / max(1, cfg.bnd_warmup))
        ring_scale = min(1.0, epoch / max(1, cfg.ring_warmup))
        shrink_scale = phase["shrink"]

        self.criterion.edge_weight = cfg.edge_weight * edge_scale
        self.criterion.bnd_weight = cfg.bnd_weight * boundary_scale
        self.criterion.bnd_bg_weight = cfg.bnd_bg_weight * boundary_scale
        self.criterion.outer_weight = cfg.outer_weight * ring_scale * shrink_scale
        self.criterion.inner_weight = cfg.inner_weight * ring_scale * shrink_scale
        self.criterion.tail_weight = cfg.tail_weight * ring_scale * shrink_scale
        self.criterion.halo_weight = cfg.halo_weight * shrink_scale

        conf_scale = ramp_by_epoch(epoch, cfg.epochs, cfg.conf_ramp_start, cfg.conf_ramp_end)
        self.criterion.support_weight = cfg.support_head_weight
        self.criterion.center_weight = cfg.center_head_weight
        self.criterion.confidence_weight = cfg.confidence_head_weight * conf_scale
        self.criterion.interference_weight = cfg.interference_head_weight * phase["clutter_sem"]
        self.criterion.boundary_rejection_weight = cfg.boundary_rejection_head_weight * phase["rejection_sem"]
        self.criterion.rejection_weight = 0.04 * phase["rejection_sem"]
        self.criterion.risk_weight = 0.03 * phase["clutter_sem"]

        aux_scale = ramp_by_epoch(epoch, cfg.epochs, 0.0, cfg.aux_ramp_end)
        aux_weight = cfg.aux_weight * aux_scale
        semantic_weight = cfg.semantic_weight
        hard_sem_weight = cfg.hard_sem_weight * phase["rejection_sem"] * shrink_scale
        return phase, aux_weight, semantic_weight, hard_sem_weight

    def train_one_epoch(self, epoch: int) -> float:
        cfg = self.cfg
        phase, aux_weight, semantic_weight, hard_sem_weight = self._configure_epoch(epoch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        meter = AverageMeter()
        progress = tqdm(self.train_loader, desc=f"Train {epoch:03d}", leave=False)

        for step, (images, labels) in enumerate(progress):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            labels = (labels > 0.5).float()
            edge_targets = morphological_edge(labels)

            outputs = self.model(images)
            prediction, edge_prediction, aux_predictions, packs, extras = unpack_outputs(outputs)
            if prediction is None:
                raise RuntimeError("Model output does not contain the final mask logits.")
            if prediction.shape[-2:] != labels.shape[-2:]:
                prediction = F.interpolate(
                    prediction, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )

            boundary_predictions: list[torch.Tensor] = []
            boundary_targets: list[torch.Tensor] = []
            if self.criterion.bnd_weight > 0.0 and isinstance(packs, dict):
                for stage_name in ("p1", "p2", "p3", "p4"):
                    stage = packs.get(stage_name)
                    if not isinstance(stage, dict) or stage.get("bnd_logits") is None:
                        continue
                    logits = stage["bnd_logits"]
                    gt = (
                        F.interpolate(labels, size=logits.shape[-2:], mode="nearest")
                        if logits.shape[-2:] != labels.shape[-2:]
                        else labels
                    )
                    kernel = 3 if min(logits.shape[-2:]) <= 128 else cfg.ring_ks
                    boundary_predictions.append(logits)
                    boundary_targets.append(outer_band(gt, kernel))

            boundary_rejection_target = outer_band(labels, 3)
            center_target = erode_with_fallback(labels, cfg.center_erode_r, cfg.center_min_keep)
            refined_hard = build_refined_hard_background(
                packs,
                extras,
                labels,
                prediction.detach(),
                cfg.hard_bg_pred_thr,
            )
            interference_target = torch.clamp(
                0.55 * boundary_rejection_target
                + 0.30 * refined_hard["hard_bg"]
                + 0.15 * refined_hard["halo_bg"],
                0.0,
                1.0,
            )
            conf_target = (
                labels
                + (1.0 - labels)
                * (0.35 * refined_hard["halo_bg"] + 0.20 * refined_hard["hard_bg"])
            ).clamp(0.0, 1.0)
            retention_target = torch.clamp(0.70 * center_target + 0.30 * labels, 0.0, 1.0)
            risk_target = torch.clamp(
                0.45 * refined_hard["conn_bg"]
                + 0.35 * refined_hard["repeat_bg"]
                + 0.20 * refined_hard["halo_bg"],
                0.0,
                1.0,
            )
            rejection_target = torch.clamp(
                0.50 * refined_hard["hard_bg"]
                + 0.30 * refined_hard["halo_bg"]
                + 0.20 * boundary_rejection_target,
                0.0,
                1.0,
            )

            loss = self.criterion(
                preds=prediction,
                targets=labels,
                edge_preds=edge_prediction,
                edge_targets=edge_targets,
                bnd_preds=boundary_predictions or None,
                bnd_targets=boundary_targets or None,
                bnd_from_logits=True,
                support_preds=extras.get("support512"),
                support_targets=labels,
                center_preds=extras.get("center512"),
                center_targets=center_target,
                interference_preds=extras.get("interference512"),
                interference_targets=interference_target,
                boundary_rejection_preds=extras.get("boundary_rejection512"),
                boundary_rejection_targets=boundary_rejection_target,
                rejection_preds=extras.get("rejection512"),
                rejection_targets=rejection_target,
                risk_preds=extras.get("risk512"),
                risk_targets=risk_target,
                conf_preds=extras.get("conf512"),
                conf_targets=conf_target,
            )

            # Candidate-preservation/background auxiliary terms used by the training code.
            if extras.get("point_prior") is not None:
                value = _resize_optional(extras["point_prior"], labels.shape[-2:])
                loss = loss + 0.03 * F.binary_cross_entropy(
                    value.clamp(1e-4, 1.0 - 1e-4), center_target
                )
            if extras.get("center_prior") is not None:
                value = _resize_optional(extras["center_prior"], labels.shape[-2:])
                loss = loss + 0.03 * F.binary_cross_entropy(
                    value.clamp(1e-4, 1.0 - 1e-4), center_target
                )
            if extras.get("tiny_target_map") is not None:
                value = _resize_optional(extras["tiny_target_map"], labels.shape[-2:])
                tiny_center = erode_with_fallback(
                    center_target, cfg.center_erode_r, cfg.center_min_keep
                )
                tiny_center = torch.clamp(tiny_center + 0.35 * center_target, 0.0, 1.0)
                loss = loss + 0.04 * F.binary_cross_entropy(
                    value.clamp(1e-4, 1.0 - 1e-4), tiny_center
                )
            for key in ("retention_response", "retained_candidate512"):
                if extras.get(key) is not None:
                    value = _resize_optional(extras[key], labels.shape[-2:])
                    loss = loss + 0.05 * F.binary_cross_entropy(
                        value.clamp(1e-4, 1.0 - 1e-4), retention_target
                    )
            if extras.get("bg_recon_err") is not None:
                value = _resize_optional(extras["bg_recon_err"], labels.shape[-2:])
                background = 1.0 - labels
                loss = loss + 0.01 * (
                    (value * background).sum() / background.sum().clamp_min(1.0)
                )

            if aux_weight > 0.0:
                aux_target = erode_with_fallback(
                    labels, cfg.aux_erode_r, cfg.aux_erode_min_keep
                )
                for aux_prediction in aux_predictions:
                    aux_prediction = _resize_optional(aux_prediction, labels.shape[-2:])
                    loss = loss + aux_weight * self.criterion(
                        preds=aux_prediction, targets=aux_target, ring_tail=False
                    )

            if semantic_weight > 0.0:
                loss = loss + semantic_weight * semantic_pack_loss(
                    packs,
                    labels,
                    support_scale=phase["support_sem"],
                    clutter_scale=phase["clutter_sem"],
                    rejection_scale=phase["rejection_sem"],
                    refined_hard=refined_hard,
                )

            hard_loss, hard_ratio, hard_fp = hard_background_pack_loss(
                packs,
                labels,
                prediction,
                refined_hard,
                cfg.hard_bg_pred_thr,
            )
            if hard_loss is not None and hard_sem_weight > 0.0:
                loss = loss + hard_sem_weight * hard_loss

            scaled_loss = loss / cfg.accumulation_steps
            scaled_loss.backward()
            if (step + 1) % cfg.accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            meter.update(float(loss.detach()), images.shape[0])
            postfix = {"loss": f"{meter.avg:.4f}"}
            if hard_ratio is not None:
                postfix["hard"] = f"{hard_ratio:.3f}"
                postfix["fp"] = f"{hard_fp:.3f}"
            progress.set_postfix(postfix)

        self.scheduler.step()
        return meter.avg

    @torch.no_grad()
    def evaluate(self, epoch: int) -> tuple[float, MetricResult]:
        self.model.eval()
        update_gate_schedule(self.model, epoch, self.cfg.epochs, self.cfg)
        self.metrics.reset()
        loss_meter = AverageMeter()

        progress = tqdm(self.eval_loader, desc=f"Eval  {epoch:03d}", leave=False)
        for images, resized_labels, original_size, _, original_labels in progress:
            images = images.to(self.device, non_blocking=True)
            resized_labels = (resized_labels.to(self.device, non_blocking=True) > 0.5).float()
            outputs = self.model(images)
            prediction, _, _, _, _ = unpack_outputs(outputs)
            if prediction is None:
                raise RuntimeError("Model output does not contain the final mask logits.")

            prediction_for_loss = _resize_optional(prediction, resized_labels.shape[-2:])
            val_loss = self.criterion(prediction_for_loss, resized_labels, ring_tail=False)
            loss_meter.update(float(val_loss), images.shape[0])

            # Eval loader uses batch_size=1 so the prediction can be restored to the native size.
            original_h = int(original_size[0, 0])
            original_w = int(original_size[0, 1])
            prediction_native = F.interpolate(
                prediction,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            )
            self.metrics.update(prediction_native, original_labels)

        return loss_meter.avg, self.metrics.compute()

    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.epochs):
            learning_rate = self.optimizer.param_groups[0]["lr"]
            train_loss = self.train_one_epoch(epoch)
            val_loss, result = self.evaluate(epoch)
            metrics_dict = result.as_dict()

            with self.history_path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(
                    [
                        epoch,
                        learning_rate,
                        train_loss,
                        val_loss,
                        result.iou,
                        result.niou,
                        result.pd,
                        result.fa,
                    ]
                )

            print(
                f"Epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f} | "
                f"IoU={result.iou:.4f} nIoU={result.niou:.4f} "
                f"Pd={result.pd:.4f} Fa={result.fa:.4f}"
            )

            save_checkpoint(
                self.run_dir / "last.pth",
                self.model,
                self.optimizer,
                epoch,
                metrics_dict,
                self.scheduler,
            )
            if result.iou > self.best_iou:
                self.best_iou = result.iou
                save_checkpoint(
                    self.run_dir / "best_iou.pth",
                    self.model,
                    self.optimizer,
                    epoch,
                    metrics_dict,
                    self.scheduler,
                )
            if result.niou > self.best_niou:
                self.best_niou = result.niou
                save_checkpoint(
                    self.run_dir / "best_niou.pth",
                    self.model,
                    self.optimizer,
                    epoch,
                    metrics_dict,
                    self.scheduler,
                )
