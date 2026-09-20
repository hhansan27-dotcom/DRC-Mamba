#!/usr/bin/env bash
set -euo pipefail
python train.py --dataset NUDT-SIRST --split 50_50 --epochs 400 --batch-size 4 --input-size 512 --crop-size 512 --lr 1e-4 --warmup-epochs 20
