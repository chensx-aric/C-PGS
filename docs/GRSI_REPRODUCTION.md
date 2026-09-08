# GRSI representative reproduction

## Target result

The representative experiment is the 1/2-labeled UniMatch-V2 model from
Table 2 of the accepted manuscript. A single checkpoint is evaluated in two
explicit modes:

| Mode | Paper mIoU (%) |
|---|---:|
| SP | 64.48 |
| SP + IP | 65.16 |

The IP result uses `eta=0.6`, `r=0.6`, and `kappa=100`, hence a 60-pixel
expansion radius. The checkpoint contains the `model_ema` state from epoch 71.

## Supported system

- Ubuntu 22.04.1 LTS or later, 64-bit.
- Python 3.10, 3.11, or 3.12.
- One NVIDIA Tesla V100 GPU with a driver capable of running the PyTorch 2.5.1
  CUDA 12.1 wheel.
- At least 8 GB of system memory and approximately 2 GB of free disk space.

## Confidential asset overlay

Dataset images, author-created semantic annotations, precomputed instance-ID
maps, and the representative checkpoint are not public. They are supplied to
the assigned GRSI reviewer as a confidential archive. Extract that archive
directly into the repository root. The resulting paths must include:

```text
data/ACA/<architectural-complex>/JPEGImages/<image>.jpg
data/ACA/<architectural-complex>/SegmentationClassNpy/<image>.npy
data/ACA/<architectural-complex>/InstanceClassNpy/<image>.npy
checkpoints/unimatch_v2_sp_aca_1_2_best.pth
```

The checkpoint SHA-256 must be:

```text
347db73172344905182250feef1dddd5bd506bb516b8fdc716274425a88741fc
```

## Commands

From the repository root, run:

```bash
bash install.sh
.venv/bin/python evaluate_representative.py
```

The evaluator accepts no parameters. It checks the checkpoint hash, requires
all 145 validation samples, computes SP and SP+IP from the same model
predictions, and writes:

```text
outputs/representative_results.csv
outputs/representative_results.md
```

Successful execution ends with output equivalent to:

```text
SP: 64.48 mIoU (paper 64.48, difference +0.00)
SP+IP: 65.16 mIoU (paper 65.16, difference +0.00)
RESULT CHECK: PASS (tolerance +/-0.05)
```

## Data restriction

The source images originate from the Institute of Automation, Chinese Academy
of Sciences data portal: <http://vision.ia.ac.cn/data/>. The source-image access
contact is `wgao@nlpr.ia.ac.cn`. Pixel-level annotations and the instance-ID
maps used for evaluation were created or prepared by the C-PGS authors and are
not publicly released. The confidential reviewer copy is supplied solely for
GRSI verification and should be deleted after review.
