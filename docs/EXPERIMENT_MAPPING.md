# Paper-to-code mapping

## Canonical algorithms

| Paper component | Clean implementation | Historical source evidence |
|---|---|---|
| Algorithm 1, hypergraph style prior | `src/cpgs/style_prior.py` | ratio-specific C-PGS scripts under `experiments/integrations/unimatch_v2` |
| Equation 16, logit modulation | `inject_style_prior_numpy` / `inject_style_prior_torch` | C-PGS UniMatch-V2 training scripts |
| Algorithm 2, instance refinement | `src/cpgs/instance_prior.py` | `experiments/integrations/unimatch_v2/supervised.py` |
| SAM2 candidate masks | `src/cpgs/sam2_masks.py` | vendored source under `third_party/sam2` |

The clean implementation follows the accepted manuscript. Legacy code contains additional heuristics (area thresholds, IoU tests, class-specific forcing, fixed pixel expansion, and experimental branches) that are not part of the published pseudocode.

## Table 2 method roles

Four comparison-only snapshots use the ResNet-101/DeepLabv3+ family in the paper: Beyond-Pixels, CSL, DAW, and DDFP.

Five C-PGS integration snapshots have paired baseline and `+ SP + IP` rows: UniMatch, CorrMatch, RankMatch, MCCL, and UniMatch-V2.

For the GRSI package, only the official UniMatch-V2 ZIP and the server's modified `UniMatch-V2` directory are retained, so reviewers can inspect the representative implementation without redistributing eight unrelated full snapshots. The accepted measurements for all methods remain in `PAPER_RESULTS.md`.

## Confirmed representative settings

1. The author confirmed the inclusive Algorithm 1 support bounds are `tau_l=1` and `tau_h=3`.
2. The author confirmed Algorithm 2 uses `kappa=100`: `R=floor(100r)`. Thus the default `r=0.6` corresponds to the legacy `expand_pixels=60`.
3. The representative SP run is now identified as `1_2_New/best.pth` with the `model_ema` key; it re-evaluates to 64.48 without IP and 65.16 with IP. Multiple classifier files and `_nor` variants remain relevant only to non-representative historical runs.
4. The public no-argument evaluator is `evaluate_representative.py`. It loads the `model_ema` state and reports SP-only and SP+IP metrics separately, avoiding the historical evaluator's unconditional IP call.
