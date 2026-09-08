#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "Usage: $0 <1/16|1/8|1/4|1/2> <prior-cache.pt> <dinov2-small.pth> [num-gpus] [data-root] [output-dir]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_ROOT="$(cd "$INTEGRATION_DIR/../../.." && pwd)"

SPLIT="$1"
PRIOR_CACHE="$2"
PRETRAINED="$3"
NUM_GPUS="${4:-1}"
DATA_ROOT="${5:-$REPOSITORY_ROOT/data/ACA}"
OUTPUT_DIR="${6:-$REPOSITORY_ROOT/outputs/unimatch_v2/${SPLIT//\//_}}"

COMMON_ARGS=(
  "$INTEGRATION_DIR/train.py"
  --config "$INTEGRATION_DIR/configs/AC.yaml"
  --split "$SPLIT"
  --data-root "$DATA_ROOT"
  --prior-cache "$PRIOR_CACHE"
  --pretrained "$PRETRAINED"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$NUM_GPUS" -gt 1 ]]; then
  torchrun --standalone --nproc-per-node "$NUM_GPUS" "${COMMON_ARGS[@]}"
else
  python "${COMMON_ARGS[@]}"
fi
