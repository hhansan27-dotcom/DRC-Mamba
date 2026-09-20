# DRC-Mamba

**Decomposition-Guided Response-Conditioning Mamba with Deferred Decision for Infrared Small Target Detection**

Official PyTorch implementation and pretrained checkpoints for infrared small target detection (IRSTD). The accompanying manuscript is a preprint submitted to *Infrared Physics & Technology*.

DRC-Mamba preserves plausible weak-target responses while gathering structural context. It describes ambiguous observations with complementary response fields, retains candidate details, reasons across multiple scan paths, and conditions final suppression on target-supporting evidence.

## At a glance

| Dataset | Train / test | IoU ↑ | F1 ↑ | nIoU ↑ | Pd ↑ | Fa ↓ | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| NUAA-SIRST | 341 / 86 | **82.91** | 90.66 | 82.16 | 99.08 | 0.99 | [NUAA-SIRST](pretrained/drc_mamba_nuaa_sirst_best_iou.pth) |
| NUDT-SIRST | 663 / 664 | **89.65** | 94.54 | 90.39 | 98.62 | 6.69 | [NUDT-SIRST](pretrained/drc_mamba_nudt_sirst_best_iou.pth) |
| IRSTD-1K | 800 / 201 | **73.72** | 84.87 | 68.65 | 91.58 | 10.38 | [IRSTD-1K](pretrained/drc_mamba_irstd1k_best_iou.pth) |

These are the DRC-Mamba results reported in Table 1 of the manuscript, using its fixed splits and evaluation protocol; they have not been independently recomputed for this README. IoU, F1, nIoU, and Pd are percentages. Fa is the number of pixels in unmatched predicted components per million evaluated pixels, so lower is better. The checkpoints are stored with [Git LFS](https://git-lfs.com/).

## Method

![Manuscript Figure 2: response decomposition and candidate preservation feed an encoder with multi-path Mamba blocks; a response-conditioned decoder produces the final prediction.](docs/figures/fig2-architecture.png)

*Manuscript Fig. 2 — overall architecture.* The processing order is **describe and preserve → contextual reasoning → conditioned suppression**:

1. **Response decomposition** describes compact-source, background-structure, and interference-sensitive evidence without forcing an early binary choice.
2. **Candidate-preservation branch (CP)** retains weak candidates and fine spatial detail before strong suppression.
3. **Multi-path Mamba (MP-Mamba)** introduces long-range structural context along complementary scan paths.
4. **Response-conditioned decoder (RCD)** uses retained target evidence to regulate interference suppression and reconstruct the segmentation map.

The public implementation uses the same names as the manuscript: `ResponseDecompositionModule`, `CandidatePreservationBranch`, `EncoderStage1`–`EncoderStage4`, `MPMambaBlock`, and `ResponseConditionedDecoder`. The three response fields are exposed as `fs`, `fb`, and `ff`, matching $F_s$, $F_b$, and $F_f$.

![Manuscript Figure 1: infrared scenes, ground truth, and three complementary response fields for two examples.](docs/figures/fig1-response-fields.png)

*Manuscript Fig. 1 — why complementary fields matter.* A local observation can activate target-like and structure-related cues at the same time.

### Qualitative comparison

![Manuscript Figure 7: infrared scenes and segmentation outputs from several methods, DRC-Mamba, and ground truth.](docs/figures/fig7-qualitative-comparison.png)

*Manuscript Fig. 7 — weak targets, bright structures, and clutter.* Insets enlarge target regions. These examples illustrate behavior; the table above gives aggregate results.

## Quick start

Install Git LFS before cloning so the three checkpoints are downloaded as weights rather than pointer files:

```bash
git lfs install
git clone https://github.com/hhansan27-dotcom/DRC-Mamba.git
cd DRC-Mamba
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Training is intended for a CUDA-capable PyTorch environment. `mamba-ssm` installation depends on the PyTorch/CUDA combination; if the requirements install fails, install compatible PyTorch and `mamba-ssm` builds for your platform. The model does not silently replace Mamba with another operator.

### Prepare datasets

Obtain the datasets from their original sources and place grayscale infrared images, binary masks, and split files under `dataset/`:

```text
dataset/
├── NUAA-SIRST/
│   ├── images/
│   ├── masks/
│   └── 80_20/{train.txt,test.txt}
├── NUDT-SIRST/
│   ├── images/
│   ├── masks/
│   └── 50_50/{train.txt,test.txt}
└── IRSTD-1K/
    ├── images/
    ├── masks/
    └── 80_20/{train.txt,test.txt}
```

Each split file contains one image ID per line without the extension; `.png` is the default. See [dataset/README.md](dataset/README.md) for the full layout. **Images, masks, and split files are not redistributed in this repository.** Reproducing the reported scores requires the same fixed partitions described in the manuscript; matching only the 80/20 or 50/50 ratio is insufficient.

### Evaluate a released checkpoint

```bash
python evaluate.py \
  --checkpoint pretrained/drc_mamba_nuaa_sirst_best_iou.pth \
  --dataset NUAA-SIRST \
  --split 80_20
```

For the other datasets, change `--checkpoint`, `--dataset`, and `--split` together:

| Dataset | `--checkpoint` | `--split` |
|---|---|---|
| NUDT-SIRST | `pretrained/drc_mamba_nudt_sirst_best_iou.pth` | `50_50` |
| IRSTD-1K | `pretrained/drc_mamba_irstd1k_best_iou.pth` | `80_20` |

The evaluator restores predictions to each image's original resolution before updating metrics. The default threshold is `0.5`; `--data-root`, `--suffix`, `--device`, and `--threshold` are available when needed. After cloning, `git lfs ls-files` should list all three checkpoints.

### Train

The manuscript uses 400 epochs, AdamW, an initial learning rate of `1e-4`, batch size 4, 512 × 512 inputs, 20 warm-up epochs, and a cosine schedule. The released model uses GroupNorm.

```bash
python train.py --dataset NUAA-SIRST --split 80_20
python train.py --dataset NUDT-SIRST --split 50_50
python train.py --dataset IRSTD-1K --split 80_20
```

Equivalent launch scripts are in [`scripts/`](scripts/). Run artifacts are written under `runs/` and ignored by Git. Resume with `--resume runs/<run>/last.pth`. Full training defaults and auxiliary loss coefficients are in [`drc_mamba/config.py`](drc_mamba/config.py).

## Evaluation protocol

- **IoU:** foreground intersection and union accumulated across all test images.
- **F1:** derived from accumulated IoU as `2 × IoU / (1 + IoU)`.
- **nIoU:** mean image-wise IoU.
- **Pd:** matched ground-truth connected components divided by ground-truth components; matching is one-to-one by nearest centroid within 3 pixels.
- **Fa:** pixels in unmatched predicted connected components per million evaluated pixels. This differs from a per-pixel background false-positive rate.

The tabulated results use a `0.5` binary threshold, eight-connected components, and no morphological post-processing. See [`drc_mamba/metrics.py`](drc_mamba/metrics.py) for the implementation.

## Repository map

```text
drc_mamba/       model, data loading, training engine, losses, metrics, utilities
pretrained/      three best-IoU checkpoints (Git LFS) and checksums
scripts/         dataset-specific training commands
tests/           metric sanity tests
docs/figures/    figures extracted from the accompanying manuscript
dataset/         layout documentation; no dataset files
train.py         training entry point
evaluate.py      evaluation entry point
```

The three checkpoints share the released architecture: 22.30 million parameters with channel progression `64 → 128 → 256 → 256`. See [pretrained/README.md](pretrained/README.md) for compatibility and SHA-256 checksums. The manuscript reports 21.45 GFLOPs for the model.

Run the included metric tests with `python -m pytest -q`.

## Citation

If this repository supports your research, please cite the manuscript. It is currently a submitted preprint; no publication DOI is claimed here.

```bibtex
@misc{zhang2026drcmamba,
  title  = {Decomposition-Guided Response-Conditioning Mamba with Deferred Decision for Infrared Small Target Detection},
  author = {Zhang, Tongliga and Sun, Yingjie and Huang, Shize and Li, Hongjie and Zheng, Yitao and Zhang, Ping},
  year   = {2026},
  note   = {Manuscript submitted to Infrared Physics & Technology},
  url    = {https://github.com/hhansan27-dotcom/DRC-Mamba}
}
```

The same authors are listed in [`CITATION.cff`](CITATION.cff). For questions about implementation or experiments, open a GitHub issue.
