# Modification and provenance notice

This directory is an experimental snapshot derived from the official UniMatch-V2 implementation by Lihe Yang, Zhen Zhao, and Hengshuang Zhao:

- Upstream repository: <https://github.com/LiheYoung/UniMatch-V2>
- Upstream license: MIT; the original license text is retained in `LICENSE`.
- Local provenance: the authors' archived UniMatch-V2 experiment workspace.

The C-PGS authors adapted the code for the 18-class ACA dataset and added experiment paths for category-level style-prior injection and instance-level refinement. The snapshot also contains ratio-specific scripts and historical evaluation branches used during the study. It should therefore be read together with the clean paper-facing implementations in `src/cpgs`, the portable protocol in `configs/aca.yaml`, and `docs/EXPERIMENT_MAPPING.md`.

Upstream UniMatch-V2 portions remain under the upstream MIT License. C-PGS-authored additions are covered by the repository's root MIT License. This notice documents provenance; it does not change either license.
