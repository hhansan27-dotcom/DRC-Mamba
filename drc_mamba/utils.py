"""General utilities for training, checkpoints, and reproducibility."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn


class AverageMeter:
    """Track a running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        value = float(value)
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_weights(module: nn.Module) -> None:
    """Xavier initialization for non-Mamba convolution/linear layers."""
    name = module.__class__.__name__.lower()
    if "mamba" in name or "ssm" in name:
        return
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def create_run_dir(output_root: str | Path, dataset: str, run_name: str | None = None) -> Path:
    output_root = Path(output_root)
    if run_name is None:
        from datetime import datetime

        run_name = f"{dataset}_DRC-Mamba_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(config: Any, path: str | Path) -> None:
    if is_dataclass(config):
        payload = asdict(config)
    elif hasattr(config, "__dict__"):
        payload = vars(config)
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        raise TypeError(f"Unsupported config type: {type(config)!r}")
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# Historical aliases are intentionally isolated here so released checkpoints from
# earlier research-code revisions remain loadable after the paper-aligned rename.
_LEGACY_PREFIX_MAP = (
    ("physics.", "response_decomposition."),
    ("drm.", "response_decomposition."),
    ("detail.", "cp_branch.detail_path."),
    ("detail_stem.", "cp_branch.detail_path."),
    ("rescue.", "cp_branch.candidate_path."),
    ("dppb.", "cp_branch.candidate_path."),
    ("sce.", "stage1."),
    ("hspm.", "stage2."),
    ("cerm.", "stage3."),
    ("fab.", "stage4."),
    ("decoder.", "rcd."),
    ("bg32_proj.", "background32_proj."),
    ("clt32_proj.", "structure32_proj."),
    ("exc32_proj.", "source32_proj."),
    ("bg16_proj.", "background16_proj."),
    ("clt16_proj.", "structure16_proj."),
    ("exc16_proj.", "source16_proj."),
)

_LEGACY_SEGMENT_MAP = (
    (".purifier.", ".background_reconstruction."),
    (".src_logit.", ".source_logit."),
    (".src_conf_logit.", ".source_confidence_logit."),
    (".src_full.", ".fs_full."),
    (".clt_full.", ".fb_full."),
    (".src_down1.", ".fs_down1."),
    (".clt_down1.", ".fb_down1."),
    (".src_down2.", ".fs_down2."),
    (".clt_down2.", ".fb_down2."),
    (".escrow_head.", ".retention_head."),
    (".keep_head.", ".retention_gate_head."),
    (".src_detail_proj.", ".fs_detail_proj."),
    (".clt_detail_proj.", ".fb_detail_proj."),
    (".src_phys_proj.", ".fs_field_proj."),
    (".clt_phys_proj.", ".fb_field_proj."),
    (".fa_phys_proj.", ".ff_field_proj."),
    (".escrow_logit.", ".retention_logit."),
    (".fa_seed_logit.", ".interference_seed_logit."),
    (".bg_init.", ".background_token_builder."),
    (".clt_init.", ".structure_token_builder."),
    (".exc_init.", ".source_token_builder."),
    (".scan.", ".mp_mamba."),
    (".bg_reader.", ".background_token_reader."),
    (".clt_reader.", ".structure_token_reader."),
    (".exc_reader.", ".source_token_reader."),
    (".veto_logit.", ".rejection_logit."),
    (".escrow_gate.", ".retention_gate_head."),
    (".scene_invalid.", ".scene_interference_head."),
    (".fa_src.", ".interference_head."),
    (".sparse_veto.", ".sparse_rejection."),
    (".up3.", ".up_fuse3."),
    (".up2.", ".up_fuse2."),
    (".up1.", ".up_fuse1."),
    (".fa256.", ".interference256_head."),
    (".fa_src512.", ".interference512_head."),
    (".reject512.", ".boundary_rejection512_head."),
    (".escrow512_head.", ".retained_candidate512_head."),
    (".veto_refine512.", ".rejection_refine512."),
    (".veto512_head.", ".rejection512_head."),
)

_LEGACY_PARAMETER_MAP = (
    ("rcd.gamma_escrow512", "rcd.gamma_retention512"),
    ("rcd.alpha_veto512", "rcd.alpha_rejection512"),
)


def _remap_legacy_key(key: str) -> str:
    """Map historical checkpoint keys to the paper-aligned public module names."""
    if key.startswith("module."):
        key = key[len("module.") :]

    for old, new in _LEGACY_PREFIX_MAP:
        if key.startswith(old):
            key = new + key[len(old) :]
            break

    for old, new in _LEGACY_SEGMENT_MAP:
        key = key.replace(old, new)

    for old, new in _LEGACY_PARAMETER_MAP:
        if key == old:
            key = new
            break

    # Early ConvBNAct checkpoints used ``bn`` even after runtime BN->GN replacement.
    key = key.replace(".bn.", ".norm.")
    return key

def remap_legacy_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {_remap_legacy_key(key): value for key, value in state_dict.items()}


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str,
    strict: bool = True,
) -> dict[str, Any]:
    # PyTorch 2.6+ defaults to ``weights_only=True``. Some released legacy
    # checkpoints also store NumPy scalar metric metadata, so allowlist only the
    # small set of NumPy scalar/dtype constructors needed by those files while
    # keeping restricted deserialization enabled.
    if hasattr(torch.serialization, "safe_globals"):
        numpy_scalar = np.core.multiarray.scalar
        safe_numpy_globals: list[Any] = [
            (numpy_scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
        ]
        for dtype in (np.float16, np.float32, np.float64, np.int32, np.int64):
            dtype_type = type(np.dtype(dtype))
            if dtype_type not in safe_numpy_globals:
                safe_numpy_globals.append(dtype_type)
        with torch.serialization.safe_globals(safe_numpy_globals):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    else:  # PyTorch 2.1--2.5: the historical default already supports this metadata.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        raw_state = checkpoint.get(
            "state_dict", checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        )
    else:
        raw_state = checkpoint
    if not isinstance(raw_state, Mapping):
        raise TypeError("Checkpoint does not contain a valid state_dict mapping.")
    state_dict = remap_legacy_state_dict(raw_state)
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {"state_dict": raw_state}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    scheduler: Any | None = None,
) -> None:
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": dict(metrics),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, Path(path))
