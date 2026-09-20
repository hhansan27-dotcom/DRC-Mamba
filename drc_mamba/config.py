"""Configuration dataclasses for DRC-Mamba training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data and optimization (paper setting)
    epochs: int = 400
    batch_size: int = 4
    input_size: int = 512
    crop_size: int = 512
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 20
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    accumulation_steps: int = 1
    workers: int = 8
    seed: int = 42
    cache_images: bool = False

    # Metric configuration
    threshold: float = 0.5
    centroid_tol: float = 3.0

    # Primary/auxiliary loss configuration
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    edge_weight: float = 0.25
    edge_warmup: int = 3
    bnd_weight: float = 0.05
    bnd_warmup: int = 8
    bnd_bg_weight: float = 0.0
    bnd_bg_dilate_ks: int = 3

    support_head_weight: float = 0.10
    center_head_weight: float = 0.10
    interference_head_weight: float = 0.10
    boundary_rejection_head_weight: float = 0.10
    confidence_head_weight: float = 0.05
    rejection_head_weight: float = 0.01
    risk_head_weight: float = 0.01

    # Morphological supervision
    ring_ks: int = 5
    aux_erode_r: int = 1
    aux_erode_min_keep: int = 16
    center_erode_r: int = 1
    center_min_keep: int = 4
    outer_weight: float = 0.22
    inner_weight: float = 0.0
    tail_weight: float = 0.015
    halo_weight: float = 0.03
    outer_r: int = 3
    inner_r: int = 1
    far_r: int = 9
    tail_k: int = 64
    ring_warmup: int = 20
    outer_margin: float = 0.25
    inner_margin: float = 0.0

    # Stage-wise auxiliary supervision
    semantic_weight: float = 0.08
    hard_sem_weight: float = 0.04
    aux_weight: float = 0.10
    aux_ramp_end: float = 0.20
    coh_thr: float = 0.65
    hard_bg_pred_thr: float = 0.5

    # Deferred-decision curriculum
    gate_ramp_start: float = 0.20
    gate_ramp_end: float = 0.75
    support_ramp_start: float = 0.05
    support_ramp_end: float = 0.30
    clutter_ramp_start: float = 0.15
    clutter_ramp_end: float = 0.45
    rejection_ramp_start: float = 0.30
    rejection_ramp_end: float = 0.60
    shrink_ramp_start: float = 0.60
    shrink_ramp_end: float = 0.90
    conf_ramp_start: float = 0.15
    conf_ramp_end: float = 0.45
