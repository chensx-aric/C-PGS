# Third-party notices

This package contains selected third-party research code needed for the representative UniMatch-V2 reproduction path. Retain all upstream notices and citations.

| Component | Location | License and retained notice |
|---|---|---|
| UniMatch-V2 official baseline | `experiments/baseline/unimatch_v2_official` | MIT; upstream `LICENSE` retained |
| C-PGS UniMatch-V2 integration | `experiments/integrations/unimatch_v2` | Derived from UniMatch-V2; upstream MIT `LICENSE` and local `MODIFICATIONS.md` retained |
| SAM2 / SAM2.1 | `third_party/sam2` | Apache-2.0 `LICENSE`; separate `LICENSE_cctorch` also retained |

Full snapshots of Beyond-Pixels, CSL, DAW, DDFP, CorrMatch, MCCL, RankMatch, and UniMatch are excluded from this GRSI package because they are not executed by the representative reproduction path. Their paper measurements remain in `docs/PAPER_RESULTS.md`, without redistributing their source trees.

The root C-PGS-authored code uses the MIT License. That choice does not relicense third-party code; each retained component remains under the license identified above. Python packages installed as dependencies remain under their respective upstream licenses.
