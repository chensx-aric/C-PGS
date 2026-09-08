# Modifications to UniMatch-V2

This directory derives its DPT/DINOv2 model code and UniMatch-V2 training
procedure from the official UniMatch-V2 source supplied by the authors. The
upstream license is retained in `LICENSE`.

The C-PGS release makes the following paper-specific changes:

- adds the 18-class ACA configuration and split manifests;
- injects the category-level style prior into teacher logits using Equation 16;
- accepts all four labeled ratios through one `--split` argument;
- loads split-aligned SP caches through an explicit `--prior-cache` path;
- supports single-GPU and `torchrun` execution without hard-coded CUDA indices;
- uses portable, unwrapped checkpoint keys and validates split compatibility
  before automatic resume;
- includes an asset-validation mode and an SP-only checkpoint-evaluation mode;
- removes inactive experimental branches, visualization snippets, ratio-specific
  script copies, and the unused auxiliary classifier.

Instance-prior (IP) post-processing is implemented once in `src/cpgs` and is
exercised by the repository-level `evaluate_representative.py`; it is not a
separate training model.
