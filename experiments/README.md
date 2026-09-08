# Experiment snapshots

This directory contains the UniMatch-V2 source needed to understand and run the representative C-PGS integration. Results, datasets, caches, logs, and checkpoints are deliberately excluded.

## Official baseline

`baseline/unimatch_v2_official` is a source-only copy of the official UniMatch-V2 baseline. It is retained as the upstream reference.

`integrations/unimatch_v2` contains the C-PGS adaptation for the 18-class ACA protocol, including the ratio-specific research entry points. Use `src/cpgs` for the clean implementations of the paper algorithms and the root-level `evaluate_representative.py` for portable evaluation.

Unrelated comparison-method repositories and AllSpark are not distributed here.
