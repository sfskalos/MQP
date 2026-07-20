# Data interface

Patient data are intentionally excluded from this repository.

Place the following files under `data/` before training:

```text
data/
  dataset_index.csv
  reports.xlsx
  features/
    manifest.json
    A/mri_medclip_roi.h5
    V/mpm_dinov2_multiscale.h5
    L/biobert_cls_english.npz
```

`dataset_index.csv` requires `case_id`, `label`, and `mr_sha256`. Labels are
`pik3ca` or `map3k3`; `mr_sha256` is used as the patient/group identifier in
grouped cross-validation.

The feature tensors expected by the released training code are:

| Modality | Encoder | Shape per case |
|---|---|---|
| MRI | MedCLIP ResNet-50 on polygon-masked ROI | `(512,)` |
| MPM | DINOv2-small at three image scales | `(3, 384)` |
| Text | BioBERT CLS embedding | `(768,)` |

`reports.xlsx` is optional when BioBERT features already exist. Its first sheet
must have `case_id` and `english_report` columns in the first row.

