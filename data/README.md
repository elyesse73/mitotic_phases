# Dataset Setup (AMI-Br)

Raw histopathology image patches and metadata are intentionally excluded from version control.

## Source

This project uses the **AMI-Br** (*Atypical and Normal Mitotic Figures in Breast Cancer*) dataset:
> Bertram et al., *"Histologic dataset of normal and atypical mitotic figures on human breast cancer (AMI-Br)"*, Scientific Data, 2025.

## Expected Local Directory Layout

Place the downloaded metadata CSV and extracted patches locally as follows (paths are configurable in `configs/default.yaml`):

```text
data/
├── AMI-BR.csv
└── patches/
    ├── normal/
    │   ├── MIDOG21_6.png
    │   └── ...
    └── atypical/
        ├── MIDOG21_7.png
        └── ...
```

Required columns in `AMI-BR.csv`:
`dataset`, `uid`, `expert1_label`, `expert2_label`, `majority_atypical` (optional: `expert1_atypical`, `expert2_atypical`, `expert3_atypical`).