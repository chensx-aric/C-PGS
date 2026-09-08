# C-PGS + UniMatch-V2

This directory is the runnable ACA integration used for the paper's
representative model. It contains one training program, `train.py`; labeled
ratios and machine-specific assets are selected with command-line arguments
rather than duplicated source files.

## Environment

From the repository root, install the CUDA 12.1 PyTorch build and the full
dependencies:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-full.txt
python -m pip install -e .
```

The validated environment is Python 3.10 with PyTorch 2.5.1 + CUDA 12.1.

## Required external assets

The repository intentionally excludes ACA images/labels, generated style
priors, DINOv2 initialization weights, and trained checkpoints. Training needs:

```text
data/ACA/...
data/splits/aca/train_labeled_<fraction>.txt
data/splits/aca/train_unlabeled_<fraction>.txt
data/splits/aca/val.txt
/external/path/cached_class_mask<fraction>.pt
/external/path/dinov2_small.pth
```

The SP cache must contain one binary 18-element vector for every line of the
corresponding unlabeled manifest, in the same order. A mapping keyed by image
path is also supported.

| `--split` | Fraction suffix | Labeled | Unlabeled |
|---|---:|---:|---:|
| `1/16` | `0.0625` | 82 | 1,226 |
| `1/8` | `0.125` | 164 | 1,144 |
| `1/4` | `0.25` | 327 | 981 |
| `1/2` | `0.5` | 654 | 654 |

## Validate assets before training

```bash
python experiments/integrations/unimatch_v2/train.py \
  --split 1/2 \
  --data-root data/ACA \
  --prior-cache /external/path/cached_class_mask0.5.pt \
  --pretrained /external/path/dinov2_small.pth \
  --check-only
```

This checks every data path, the expected labeled/unlabeled/validation counts,
and SP-cache shape/alignment without allocating a GPU.

## Train

Single GPU:

```bash
python experiments/integrations/unimatch_v2/train.py \
  --split 1/2 \
  --data-root data/ACA \
  --prior-cache /external/path/cached_class_mask0.5.pt \
  --pretrained /external/path/dinov2_small.pth
```

Four GPUs on one node:

```bash
torchrun --standalone --nproc-per-node 4 \
  experiments/integrations/unimatch_v2/train.py \
  --split 1/2 \
  --data-root data/ACA \
  --prior-cache /external/path/cached_class_mask0.5.pt \
  --pretrained /external/path/dinov2_small.pth
```

Replace only `--split` and its matching cache to run another paper split.
Outputs default to `outputs/unimatch_v2/1_16`, `1_8`, `1_4`, or `1_2`, so
different ratios cannot accidentally resume each other's checkpoints. An
existing `latest.pth` is resumed automatically; use `--no-resume` to start a
new run.

Equation 16 is enabled by default with `alpha=0.6`. `--no-style-prior` runs the
same model as an ablation without SP. IP is validation-time post-processing,
not a training branch; use the repository's `evaluate_representative.py` for
the verified SP and SP+IP results.

## Evaluate a checkpoint without IP

```bash
python experiments/integrations/unimatch_v2/train.py \
  --split 1/2 \
  --data-root data/ACA \
  --checkpoint checkpoints/unimatch_v2_sp_aca_1_2_best.pth \
  --checkpoint-key model_ema \
  --evaluate-only
```

The representative epoch-71 `model_ema` state reports 64.4824 mIoU (64.48
when rounded as in the paper). The root evaluator additionally applies IP and
reports 65.16.

## Implementation notes

- `train.py` supports both a normal Python process and `torchrun` DDP.
- Paths are resolved from the repository root and can be overridden explicitly.
- Checkpoints store unwrapped model keys, making them portable between one and
  multiple GPUs. Historical `module.`-prefixed checkpoints are accepted.
- The paper protocol is in `configs/AC.yaml`; the clean Algorithm 1
  implementation is in `src/cpgs/style_prior.py`.

This integration derives from UniMatch-V2. See `LICENSE`, `MODIFICATIONS.md`,
and the repository-level `THIRD_PARTY_NOTICES.md`.
