#!/usr/bin/env bash
set -euo pipefail
python train.py --dataset IRSTD-1K --split 80_20 --epochs 400 --batch-size 4 --input-size 512 --crop-size 512 --lr 1e-4 --warmup-epochs 20
