#!/usr/bin/env bash
# Example: sbatch --export=ALL,SPLIT=1/2,PRIOR_CACHE=/path/cache.pt,PRETRAINED=/path/dinov2_small.pth scripts/slurm_train.sh
#SBATCH --job-name=cpgs-unimatch-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

set -euo pipefail

: "${SPLIT:?Set SPLIT to 1/16, 1/8, 1/4, or 1/2}"
: "${PRIOR_CACHE:?Set PRIOR_CACHE to the split-aligned style-prior cache}"
: "${PRETRAINED:?Set PRETRAINED to the DINOv2-S initialization checkpoint}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_ROOT="$(cd "$INTEGRATION_DIR/../../.." && pwd)"
NUM_GPUS="${NUM_GPUS:-4}"
DATA_ROOT="${DATA_ROOT:-$REPOSITORY_ROOT/data/ACA}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPOSITORY_ROOT/outputs/unimatch_v2/${SPLIT//\//_}}"

torchrun --standalone --nproc-per-node "$NUM_GPUS" \
  "$INTEGRATION_DIR/train.py" \
  --config "$INTEGRATION_DIR/configs/AC.yaml" \
  --split "$SPLIT" \
  --data-root "$DATA_ROOT" \
  --prior-cache "$PRIOR_CACHE" \
  --pretrained "$PRETRAINED" \
  --output-dir "$OUTPUT_DIR"
