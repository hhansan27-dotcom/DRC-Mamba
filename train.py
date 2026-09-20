#!/usr/bin/env python3
"""Train DRC-Mamba on an IRSTD dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from drc_mamba.config import TrainConfig
from drc_mamba.data import EvalDataset, TrainDataset, read_split
from drc_mamba.engine import Trainer
from drc_mamba.model import build_drc_mamba
from drc_mamba.utils import (
    create_run_dir,
    initialize_weights,
    load_checkpoint,
    save_config,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DRC-Mamba for infrared small target detection")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--dataset", type=str, default="NUAA-SIRST")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split directory under the dataset (default: 50_50 for NUDT-SIRST, otherwise 80_20)",
    )
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Paper training defaults.
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--cache-images", action="store_true")
    return parser.parse_args()


def infer_split(dataset: str, requested: str | None) -> str:
    if requested:
        return requested
    return "50_50" if "nudt" in dataset.lower() else "80_20"


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu only for debugging.")
    device = torch.device(args.device)
    split_name = infer_split(args.dataset, args.split)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_size=args.input_size,
        crop_size=args.crop_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        accumulation_steps=args.accumulation_steps,
        workers=args.workers,
        seed=args.seed,
        cache_images=args.cache_images,
    )
    seed_everything(cfg.seed)

    dataset_dir = args.data_root / args.dataset
    train_ids, eval_ids = read_split(dataset_dir, split_name)
    train_set = TrainDataset(
        dataset_dir,
        train_ids,
        base_size=cfg.input_size,
        crop_size=cfg.crop_size,
        suffix=args.suffix,
        cache_images=cfg.cache_images,
    )
    eval_set = EvalDataset(
        dataset_dir,
        eval_ids,
        input_size=cfg.input_size,
        suffix=args.suffix,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.workers > 0,
        drop_last=True,
    )
    # Native-resolution metric evaluation requires batch_size=1.
    eval_loader = DataLoader(
        eval_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.workers > 0,
    )

    model = build_drc_mamba().to(device)
    checkpoint = None
    if args.resume is None:
        model.apply(initialize_weights)
    else:
        checkpoint = load_checkpoint(model, args.resume, device=device, strict=True)

    run_dir = create_run_dir(args.output, args.dataset, args.run_name)
    save_config(
        {
            **vars(cfg),
            "dataset": args.dataset,
            "data_root": str(args.data_root),
            "split": split_name,
            "suffix": args.suffix,
            "device": args.device,
        },
        run_dir / "config.json",
    )

    trainer = Trainer(model, train_loader, eval_loader, cfg, device, run_dir)
    if checkpoint is not None:
        trainer.restore_optimizer(checkpoint)
    trainer.fit()


if __name__ == "__main__":
    main()
