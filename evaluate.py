#!/usr/bin/env python3
"""Evaluate a trained DRC-Mamba checkpoint on an IRSTD test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from drc_mamba.data import EvalDataset, read_split
from drc_mamba.metrics import IRSTDMetrics
from drc_mamba.model import build_drc_mamba
from drc_mamba.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DRC-Mamba")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--dataset", type=str, default="NUAA-SIRST")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--centroid-tol", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def infer_split(dataset: str, requested: str | None) -> str:
    if requested:
        return requested
    return "50_50" if "nudt" in dataset.lower() else "80_20"


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)

    split_name = infer_split(args.dataset, args.split)
    dataset_dir = args.data_root / args.dataset
    _, test_ids = read_split(dataset_dir, split_name)
    dataset = EvalDataset(
        dataset_dir,
        test_ids,
        input_size=args.input_size,
        suffix=args.suffix,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = build_drc_mamba().to(device)
    load_checkpoint(model, args.checkpoint, device=device, strict=True)
    model.eval()

    metrics = IRSTDMetrics(
        threshold=args.threshold,
        centroid_tol=args.centroid_tol,
        from_logits=True,
    )
    for images, _, original_size, _, original_labels in tqdm(loader, desc="Evaluate"):
        images = images.to(device, non_blocking=True)
        logits = model(images)["mask"]
        height = int(original_size[0, 0])
        width = int(original_size[0, 1])
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        metrics.update(logits, original_labels)

    result = metrics.compute()
    print(
        f"IoU={result.iou:.6f} | nIoU={result.niou:.6f} | "
        f"Pd={result.pd:.6f} | Fa={result.fa:.6f}"
    )


if __name__ == "__main__":
    main()
