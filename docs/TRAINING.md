# Training protocol

The accepted-paper UniMatch-V2 experiments use the following configuration:

- ACA training pool: 1,308 images.
- Labeled splits: 82 (1/16), 164 (1/8), 327 (1/4), and 654 (1/2).
- Validation set: 145 images.
- Model: DPT with DINOv2-S.
- Training: 80 epochs, crop size 322 x 322, batch size 8 per GPU.
- Hardware: four NVIDIA Tesla V100 GPUs.
- Learning rate: `5e-6`; backbone multiplier: `40`.
- Pseudo-label confidence threshold: `0.95`.
- SP defaults: `alpha=0.6`, inclusive `tau_l=1`, `tau_h=3`.
- IP defaults: `eta=0.6`, `r=0.6`, `kappa=100`.

The exact split manifests are in `data/splits/aca`, and the machine-readable
protocol is in `configs/aca.yaml`. All four splits use the single entry point
`experiments/integrations/unimatch_v2/train.py`. For the representative 1/2
run, the integration receives the ACA root, the `train_labeled_0.5.txt` and
`train_unlabeled_0.5.txt` manifests, the aligned style-prior cache, and the
DINOv2-S initialization weights.

Example:

```bash
python experiments/integrations/unimatch_v2/train.py \
  --split 1/2 \
  --data-root data/ACA \
  --prior-cache /external/path/cached_class_mask0.5.pt \
  --pretrained /external/path/dinov2_small.pth
```

Only `--split` and the matching cache path change for 1/16, 1/8, and 1/4.
Run the same command with `--check-only` first to verify every data path,
manifest count, and cache row. The generated style-prior caches are derived
from restricted ACA assets and are therefore not distributed publicly. They
may be supplied confidentially for a full training audit subject to the same
data-use restrictions as the source dataset. Instance-ID maps are required by
the SP+IP evaluator, but not by the SP training loop.

Training is not the no-argument GRSI target because it requires four GPUs and
many hours. The submitted representative package contains the 145-image
validation subset, corresponding author annotations and instance-ID maps, and
the verified epoch-71 `model_ema` checkpoint. The no-argument
`evaluate_representative.py` entry point performs the representative check.
