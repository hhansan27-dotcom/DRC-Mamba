# Pretrained checkpoints

This directory contains the three real pretrained checkpoint binaries supplied with the project. Each file is about 89 MB and is configured for Git LFS through the repository `.gitattributes`, so a normal `git add` / `git push` workflow will upload the checkpoint objects through Git LFS.
The supplied checkpoint objects predate the final paper-aligned public naming. They can
still be loaded directly by `evaluate.py` / `drc_mamba.utils.load_checkpoint`, which
transparently remaps historical state-dict keys to the current module names without
changing tensor values.

| Dataset | File | Original best-IoU epoch |
|---|---|---:|
| NUAA-SIRST | `drc_mamba_nuaa_sirst_best_iou.pth` | 102 |
| NUDT-SIRST | `drc_mamba_nudt_sirst_best_iou.pth` | 213 |
| IRSTD-1K | `drc_mamba_irstd1k_best_iou.pth` | 180 |

## Architecture verification

The three original checkpoints were checked before conversion:

- identical ordered state-dict key sets and tensor shapes;
- 742 state tensors and 22,300,809 scalar entries in every checkpoint;
- channel progression `64 -> 128 -> 256 -> 256`, matching the released model;
- Mamba state size `d_state=16` and expansion `expand=2`;
- no BatchNorm running mean/variance/counter buffers are present.

The last point is important: the earlier research code created layers under an attribute
named `bn`, but training replaced those modules with GroupNorm at runtime. Therefore
legacy keys such as `*.bn.weight` / `*.bn.bias` are affine GroupNorm parameters, not
BatchNorm statistics. The release conversion renames those keys to `*.norm.*` without
changing any tensor values.

## Checkpoint compatibility

The public source now follows the manuscript terminology. Historical state-dict aliases
are isolated in `drc_mamba/utils.py`; loading remaps names only and does not modify any
model tensor. This keeps the supplied pretrained weights compatible with the renamed
response-decomposition, candidate-preservation, encoder-stage, MP-Mamba, and RCD modules.

SHA-256 checksums:

```text
71e5bd42c56135f421d57a72b645b8c7fb5827c0aae80fcb8bea0eafdf4a0626  drc_mamba_irstd1k_best_iou.pth
df1d15c31e46074eba10e3799053028eac0bcd4ed0b1379ad8ed57a22dc9f6d5  drc_mamba_nuaa_sirst_best_iou.pth
752e87e0e78dfbc8dee3d65d2b93fdbb881665f1c9ef70aa70d660c7a86c5560  drc_mamba_nudt_sirst_best_iou.pth
```

## Evaluation note

For paper reproduction, metrics should be recomputed with `evaluate.py`, the official
split files, and the dataset images/masks. Historical scalar metadata stored in the
original research checkpoints is not used as the canonical result table in this release.
