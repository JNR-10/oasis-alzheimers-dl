# Fall 2026 Deep Learning Workspace

This directory is reserved for the new longitudinal deep-learning implementation. It is intentionally minimal during the proposal stage and does not contain fake model or training files.

The planned organization is:

- `data/`: cohort manifests and dataset-specific metadata; raw and processed imaging remain outside version control.
- `preprocessing/`: orientation, registration, resampling, skull-stripping/brain-extraction integration, and intensity normalization.
- `models/`: the 3D CNN/ResNet baseline, SOTA adapters, shared visit encoder, temporal module, and anatomy-guided heads.
- `training/`: training loops, loss functions, subject-level splitting, checkpointing, and early stopping.
- `evaluation/`: ROC-AUC, balanced accuracy, macro-F1, sensitivity, specificity, confusion matrices, confidence intervals, and ablation reports.
- `configs/`: reproducible experiment settings.
- `notebooks/` and `scripts/`: lightweight exploration, cohort construction, preprocessing, training, and evaluation entrypoints.

The actual implementation will be added after OASIS-3 access and cohort feasibility are confirmed. No OASIS imaging data or model weights should be committed to this repository.
