# Accepted-manuscript result ledger

This file records the headline ACA mIoU values used to map checkpoints and logs back to the paper. It is not a substitute for raw evaluation outputs.

## Table 2

| Backbone | Method | 1/16 (82) | 1/8 (164) | 1/4 (327) | 1/2 (654) |
|---|---|---:|---:|---:|---:|
| ResNet-101 | Supervised | 48.76 | 53.64 | 56.95 | 56.81 |
| ResNet-101 | DAW | 55.87 | 59.94 | 61.67 | 55.29 |
| ResNet-101 | DDFP | 48.70 | 59.53 | 58.11 | 58.07 |
| ResNet-101 | Beyond-Pixels | 52.17 | 59.78 | 58.50 | 55.16 |
| ResNet-101 | CSL | 54.81 | 61.31 | 61.63 | 57.81 |
| ResNet-101 | UniMatch | 52.72 | 59.35 | 59.42 | 54.74 |
| ResNet-101 | UniMatch + SP + IP | 53.95 | 60.64 | 59.80 | 57.66 |
| ResNet-101 | CorrMatch | 45.26 | 57.87 | 50.26 | 52.63 |
| ResNet-101 | CorrMatch + SP + IP | 45.59 | 59.12 | 50.69 | 53.52 |
| ResNet-101 | RankMatch | 50.10 | 58.35 | 58.07 | 53.23 |
| ResNet-101 | RankMatch + SP + IP | 51.31 | 59.87 | 57.83 | 55.35 |
| ResNet-101 | MCCL | 52.26 | 59.09 | 55.53 | 55.45 |
| ResNet-101 | MCCL + SP + IP | 52.25 | 60.48 | 56.99 | 57.28 |
| DINOv2-S | Supervised | 54.86 | 61.07 | 61.23 | 61.67 |
| DINOv2-S | UniMatch-V2 | 56.13 | 63.60 | 63.86 | 62.67 |
| DINOv2-S | UniMatch-V2 + SP + IP | 58.26 | 64.30 | 65.43 | 65.16 |

## ACA component-wise result ledger

These values distinguish the SP-only target from the `nor` and full SP+IP rows used during checkpoint mapping.

| Variant | 1/16 (82) | 1/8 (164) | 1/4 (327) | 1/2 (654) |
|---|---:|---:|---:|---:|
| Supervised | 54.86 | 61.07 | 61.23 | 61.67 |
| UniMatch-V2 baseline | 56.13 | 63.60 | 63.86 | 62.67 |
| Style prior (SP) | **57.87** | **64.20** | **65.20** | **64.48** |
| Region refinement (IP) | 56.44 | 63.90 | 64.27 | 62.55 |
| SP + IP | 58.26 | 64.30 | 65.43 | 65.16 |
| `nor` | 56.92 | 62.87 | 64.82 | 62.98 |

## Table 7 (1/4 labeled, UniMatch-V2)

| SP | IP | mIoU | Auxiliary parameters | Train | Inference | Memory |
|---|---|---:|---:|---:|---:|---:|
| | | 63.86 | – | 0.88 s/iter | 0.04 s/image | 10.35 GB |
| ✓ | | 65.20 | 0.101 M | 0.87 s/iter | 0.04 s/image | 10.41 GB |
| | ✓ | 64.27 | 0 | 0.88 s/iter | 3.63 s/image | 10.35 GB |
| ✓ | ✓ | 65.43 | 0.101 M | 0.87 s/iter | 3.44 s/image | 10.41 GB |
