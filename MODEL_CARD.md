# MQP model card

## Intended use

MQP is a research prototype for PIK3CA versus MAP3K3 classification from MRI,
MPM pathology, and English clinical text under incomplete-modality settings.

## Architecture

- MRI: MedCLIP ResNet-50 ROI features.
- MPM: DINOv2-small three-scale tokens with learned scale-aware processing.
- Text: BioBERT CLS features.
- Fusion: modality-specific Gaussian shared/private disentanglement,
  cross-modal attention, and uncertainty-aware gated aggregation.
- Missing MPM: MRI-conditioned latent reconstruction trained with
  SmoothL1/cosine reconstruction and classification distillation losses.

The design is inspired by MIDAS (TPAMI, 2026), but this repository is an
independent medical-task adaptation rather than an official MIDAS implementation.

## Limitations

The reported experiment contains 41 local cases. Metrics are exploratory and
must not be interpreted as evidence for clinical deployment. External validation,
confidence intervals, calibration analysis, and prospective testing are required.

No raw images, reports, labels, patient identifiers, or per-case predictions are
included in this repository. Released checkpoints have also been stripped of
fold membership and local file paths.

