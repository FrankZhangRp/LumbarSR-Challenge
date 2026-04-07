# LumbarSR: A Paired Clinical CT and Photon-Counting Micro-CT Dataset for Human Lumbar Vertebrae

<p align="center">
  <img src="docs/images/logo_512.png" alt="LumbarSR Logo" width="200">
</p>

<p align="center">
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Samples-30-blue" alt="Samples">
  <img src="https://img.shields.io/badge/Slices-212K-green" alt="Slices">
  <img src="https://img.shields.io/badge/Resolution-105μm-orange" alt="Resolution">
</p>

## Overview

LumbarSR is a paired clinical CT and Micro-PCCT benchmark for lumbar vertebral super-resolution. It contains 30 ex vivo human lumbar samples scanned with multiple clinical CT configurations and matched `105 μm` Micro-PCCT.

The benchmark targets reconstruction of bone microstructure from routine clinical CT into Micro-PCCT-like resolution for musculoskeletal imaging research.

<p align="center">
  <a href="docs/images/showcase.gif">
    <img src="docs/images/showcase_preview.gif" alt="Multi-resolution CT Visualization" width="100%">
  </a>
  <br>
  <em>Tri-planar views and 3D bone rendering across different CT resolutions (<a href="docs/images/showcase.gif">Full resolution</a>)</em>
</p>

## Task Description

Given clinical CT images with **soft tissue kernel** at two FOV settings (`195 μm` and `586 μm` in-plane), reconstruct the corresponding `105 μm` isotropic Micro-PCCT target.

<p align="center">
  <a href="docs/images/flowchart.png">
    <img src="docs/images/flowchart_preview.png" alt="Pipeline Flowchart" width="100%">
  </a>
  <br>
  <em>Pipeline overview (<a href="docs/images/flowchart.png">Full resolution</a>)</em>
</p>

## Super-Resolution Visualization

<p align="center">
  <a href="docs/images/sr_visualization_lumbar26_586um.pdf">
    <img src="docs/images/sr_visualization_lumbar26_586um.jpg" alt="Representative super-resolution comparison on Lumbar_26 at 586um FOV" width="100%">
  </a>
  <br>
  <em>Representative tri-planar super-resolution comparison on <code>Lumbar_26</code> with <code>586X_586Y_1000Z_S</code> input. Columns from left to right: Clinical CT, Nearest, SwinIR, SRCNN, UNet, ESRGAN, and Micro-PCCT target (<a href="docs/images/sr_visualization_lumbar26_586um.pdf">PDF</a>).</em>
</p>

## Dataset

### Download

**[Download Dataset from Zenodo (ZIP parts)](https://doi.org/10.5281/zenodo.19404387)**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19404387.svg)](https://doi.org/10.5281/zenodo.19404387)

The dataset is distributed on Zenodo as split ZIP archives. Please use the Zenodo record for the current file list and download instructions.

### Directory Structure

```
LumbarSR/
├── original_dicom/              # Original DICOM data
│   ├── Lumbar_01/
│   │   ├── clinical_ct/
│   │   │   ├── 195X_195Y_500Z_S/      # Small FOV, soft kernel
│   │   │   ├── 195X_195Y_1000Z_S/     # Small FOV, soft kernel
│   │   │   ├── 586X_586Y_500Z_S/      # Large FOV, soft kernel
│   │   │   └── 586X_586Y_1000Z_S/     # Large FOV, soft kernel
│   │   └── micro_ct/
│   │       └── *.dcm                   # Original Micro-PCCT DICOM
│   ├── Lumbar_02/
│   │   └── ...
│   └── Lumbar_30/
│       └── ...
├── RegisteredData/              # Processed NIfTI data (ready to use)
│   ├── Lumbar_01/
│   │   ├── Lumbar01_ClinicalCT_195X_195Y_500Z_S_registered.nii.gz
│   │   ├── Lumbar01_ClinicalCT_195X_195Y_1000Z_S_registered.nii.gz
│   │   ├── Lumbar01_ClinicalCT_586X_586Y_500Z_S_registered.nii.gz
│   │   ├── Lumbar01_ClinicalCT_586X_586Y_1000Z_S_registered.nii.gz
│   │   └── Lumbar01_MicroPCCT_105um.nii.gz                          # Ground truth
│   ├── Lumbar_02/
│   │   └── ...
│   └── Lumbar_30/
│       └── ...
├── BoneMask/                    # Released binary bone ROI aligned to RegisteredData
│   ├── Lumbar_01/
│   │   ├── Lumbar01_MicroPCCT_105um_BoneMask.nii.gz
│   │   ├── Lumbar01_ClinicalCT_195X_195Y_500Z_B_registered_BoneMask.nii.gz
│   │   ├── Lumbar01_ClinicalCT_195X_195Y_500Z_S_registered_BoneMask.nii.gz
│   │   ├── Lumbar01_ClinicalCT_195X_195Y_1000Z_B_registered_BoneMask.nii.gz
│   │   ├── Lumbar01_ClinicalCT_195X_195Y_1000Z_S_registered_BoneMask.nii.gz
│   │   ├── Lumbar01_ClinicalCT_586X_586Y_500Z_B_registered_BoneMask.nii.gz
│   │   ├── Lumbar01_ClinicalCT_586X_586Y_500Z_S_registered_BoneMask.nii.gz
│   │   ├── Lumbar01_ClinicalCT_586X_586Y_1000Z_B_registered_BoneMask.nii.gz
│   │   └── Lumbar01_ClinicalCT_586X_586Y_1000Z_S_registered_BoneMask.nii.gz
│   ├── Lumbar_02/
│   │   └── ...
│   └── Lumbar_30/
│       └── ...
└── README.md
```

`RegisteredData/` is the recommended starting point. `BoneMask/` provides binary ROIs aligned to the released NIfTI volumes. Each sample contains one `MicroPCCT` mask and eight sequence-level clinical masks.

### Recommended Split

| Usage | Samples |
|-------|---------|
| Training / development | `Lumbar_01` to `Lumbar_25` |
| Reference test set | `Lumbar_26` to `Lumbar_30` |

### Input Data (Clinical CT)

For each sample, we provide 4 sequences with soft tissue kernel:

| Sequence | In-plane Resolution | Slice Thickness | FOV |
|----------|---------------------|-----------------|-----|
| 195X_195Y_500Z_S | 195um | 500um | Small |
| 195X_195Y_1000Z_S | 195um | 1000um | Small |
| 586X_586Y_500Z_S | 586um | 500um | Large |
| 586X_586Y_1000Z_S | 586um | 1000um | Large |

Micro-PCCT targets are released as `int16` NIfTI volumes at `105 μm` isotropic resolution.

## Evaluation Metrics

### CT Window Settings

Image-quality metrics are reported within the released `BoneMask` ROI under three CT windows:

| Window | Window Center (WC) | Window Width (WW) | Description |
|--------|-------------------|-------------------|-------------|
| Raw | - | - | Original HU values (-1024 ~ 3071) |
| Bone | 400 | 1800 | Bone structure visualization |
| Soft Tissue | 40 | 400 | Soft tissue visualization |

## Getting Started

### Environment Setup

```bash
git clone https://github.com/frankzhangrp/LumbarSR-Challenge.git
cd LumbarSR-Challenge

conda create -n lumbarsr python=3.10 -y
conda activate lumbarsr

pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies: PyTorch, MONAI, nibabel, pydicom, SimpleITK, numpy, scipy, and scikit-image.

## Repository Contents

The public repository currently contains four main parts:

| Directory | Status | Description |
|-----------|--------|-------------|
| `baseline/` | Public | Rigid registration baseline based on ANTs, including `register_ants.py` and `batch_register.py` |
| `methods/` | Public | Super-resolution method scripts and usage notes |
| `evaluation/` | Public | Image quality evaluation and trabecular summary scripts |
| `docs/` | Public | GitHub Pages website, visualizations, and benchmark result pages |

The public benchmark reports results for registered clinical CT, nearest interpolation, `SRCNN`, `UNet`, `ESRGAN`, and `SwinIR` under the released `BoneMask`-ROI protocol. Reproducible usage commands are documented in [`methods/README.md`](methods/README.md) and the scripts under [`evaluation/`](evaluation/).

## Registration Baseline and Mask Evaluation

The public repository includes rigid registration code in [`baseline/register_ants.py`](baseline/register_ants.py) and [`baseline/batch_register.py`](baseline/batch_register.py). Public SR evaluation uses the released sequence-level `BoneMask` ROI aligned to each `RegisteredData` volume.

### Registration Evaluation

Registration quality is reported within the released `BoneMask` ROI using `Dice`, `HD95`, and `HD`.

| Method | ROI | Dice ↑ | HD95 ↓ | HD ↓ |
|--------|-----|--------|--------|------|
| ANTs rigid baseline (`195X_195Y_500Z_B`) | BoneMask ROI | `0.9826 ± 0.0017` | `0.2266 ± 0.0131 mm` | `1.1476 ± 0.5960 mm` |
| ANTs rigid baseline (`586X_586Y_500Z_B`) | BoneMask ROI | `0.9811 ± 0.0023` | `0.2887 ± 0.0358 mm` | `0.9095 ± 0.2725 mm` |
| ANTs rigid baseline (overall) | BoneMask ROI | `0.9818 ± 0.0021` | `0.2576 ± 0.0411 mm` | `1.0285 ± 0.4785 mm` |

### Bone Morphometry

The public benchmark reports ROI-based bone morphometry in the released `BoneMask`-defined VOI:

| Category | Metrics |
|----------|---------|
| Core trabecular morphometry | `BV/TV`, `Tb.Th`, `Tb.Sp`, `Tb.N` |
| Released VOI definition | derived from the released `BoneMask` and used consistently for all public morphometry statistics |

Current public subset (`195X_195Y_1000Z_S`):

| Method / Data | BV/TV | Tb.Th (mm) | Tb.Sp (mm) | Tb.N (mm^-1) |
|---------------|-------|------------|------------|--------------|
| Micro-PCCT reference | `0.2208 ± 0.0174` | `0.2573 ± 0.0141` | `0.3647 ± 0.0192` | `0.8608 ± 0.0881` |
| Registered clinical CT baseline | `0.0055 ± 0.0029` | `0.5837 ± 0.1179` | `4.1319 ± 1.1314` | `0.0097 ± 0.0056` |
| SRCNN | `0.0605 ± 0.0212` | `0.7673 ± 0.0865` | `0.9809 ± 0.2448` | `0.0773 ± 0.0202` |
| UNet | `0.0931 ± 0.0235` | `0.4324 ± 0.0550` | `0.4565 ± 0.0574` | `0.2133 ± 0.0346` |
| ESRGAN | `0.1330 ± 0.0239` | `0.2740 ± 0.0112` | `0.3941 ± 0.0249` | `0.4841 ± 0.0773` |
| SwinIR | `0.0415 ± 0.0140` | `0.7371 ± 0.0845` | `1.3594 ± 0.3529` | `0.0549 ± 0.0127` |

Current public subset (`586X_586Y_1000Z_S`):

| Method / Data | BV/TV | Tb.Th (mm) | Tb.Sp (mm) | Tb.N (mm^-1) |
|---------------|-------|------------|------------|--------------|
| Micro-PCCT reference | `0.2208 ± 0.0174` | `0.2573 ± 0.0141` | `0.3647 ± 0.0192` | `0.8608 ± 0.0881` |
| Registered clinical CT baseline | `0.0026 ± 0.0018` | `0.5338 ± 0.1322` | `5.0502 ± 1.8719` | `0.0053 ± 0.0044` |
| SRCNN | `0.1271 ± 0.0204` | `0.6565 ± 0.0739` | `0.4540 ± 0.0425` | `0.1928 ± 0.0104` |
| UNet | `0.1227 ± 0.0254` | `0.4639 ± 0.0467` | `0.4421 ± 0.0459` | `0.2624 ± 0.0300` |
| ESRGAN | `0.1506 ± 0.0209` | `0.2730 ± 0.0089` | `0.3710 ± 0.0160` | `0.5511 ± 0.0702` |
| SwinIR | `0.0486 ± 0.0144` | `0.4010 ± 0.0203` | `0.9216 ± 0.2053` | `0.1198 ± 0.0305` |

### Paired Statistical Comparison vs Micro-PCCT

For the released test subset, we report one-sided exact `Wilcoxon signed-rank` tests against `Micro-PCCT`, paired by case and reported separately for each FOV (`n = 5`; smallest attainable exact one-sided `p = 0.03125`). `BV/TV` and `Tb.N` test `pred < Micro-PCCT`; `Tb.Th` and `Tb.Sp` test `pred > Micro-PCCT`. Each cell below is `signed mean delta / exact one-sided p value`.

Compact paired comparison vs `Micro-PCCT` (`195X_195Y_1000Z_S`):

| Method | BV/TV (Δ / p) | Tb.Th (Δ / p) | Tb.Sp (Δ / p) | Tb.N (Δ / p) |
|---|---:|---:|---:|---:|
| Registered clinical CT baseline | `-0.2125 / 0.03125` | `+0.3761 / 0.03125` | `+3.8776 / 0.03125` | `-0.8398 / 0.03125` |
| Nearest | `-0.2125 / 0.03125` | `+0.3761 / 0.03125` | `+3.8776 / 0.03125` | `-0.8398 / 0.03125` |
| SRCNN | `-0.1556 / 0.03125` | `+0.5861 / 0.03125` | `+0.5512 / 0.03125` | `-0.7762 / 0.03125` |
| UNet | `-0.1217 / 0.03125` | `+0.1907 / 0.03125` | `+0.0951 / 0.03125` | `-0.6357 / 0.03125` |
| ESRGAN | `-0.0878 / 0.03125` | `+0.0167 / 0.03125` | `+0.0293 / 0.09375` | `-0.3767 / 0.03125` |
| SwinIR | `-0.1793 / 0.03125` | `+0.4798 / 0.03125` | `+0.9947 / 0.03125` | `-0.8059 / 0.03125` |

Compact paired comparison vs `Micro-PCCT` (`586X_586Y_1000Z_S`):

| Method | BV/TV (Δ / p) | Tb.Th (Δ / p) | Tb.Sp (Δ / p) | Tb.N (Δ / p) |
|---|---:|---:|---:|---:|
| Registered clinical CT baseline | `-0.2167 / 0.03125` | `+0.2665 / 0.03125` | `+5.0664 / 0.03125` | `-0.8493 / 0.03125` |
| Nearest | `-0.2167 / 0.03125` | `+0.2665 / 0.03125` | `+5.0664 / 0.03125` | `-0.8493 / 0.03125` |
| SRCNN | `-0.0864 / 0.03125` | `+0.4665 / 0.03125` | `+0.0593 / 0.09375` | `-0.6719 / 0.03125` |
| UNet | `-0.0920 / 0.03125` | `+0.2314 / 0.03125` | `+0.0863 / 0.03125` | `-0.5960 / 0.03125` |
| ESRGAN | `-0.0702 / 0.03125` | `+0.0156 / 0.03125` | `+0.0063 / 0.40625` | `-0.3097 / 0.03125` |
| SwinIR | `-0.1722 / 0.03125` | `+0.1437 / 0.03125` | `+0.5568 / 0.03125` | `-0.7410 / 0.03125` |

## Methods and Reproduction

Public benchmark tables currently cover registered clinical CT, `Nearest`, `SRCNN`, `UNet`, `ESRGAN`, and `SwinIR`. Model usage notes are kept in [`methods/README.md`](methods/README.md), and evaluation scripts are provided in [`evaluation/`](evaluation/).

### Baseline Performance

Training uses `Lumbar_01` to `Lumbar_25`; evaluation reports mean ± standard deviation on `Lumbar_26` to `Lumbar_30`.

#### PSNR Results (dB) ↑

| Method | Small FOV (195μm × 195μm) | | | | | | Large FOV (586μm × 586μm) | | | | | |
|--------|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Full** | | | **Masked** | | | **Full** | | | **Masked** | | |
| | Raw | Bone | Soft | Raw | Bone | Soft | Raw | Bone | Soft | Raw | Bone | Soft |
| **Baseline** | 21.94±0.67 | 18.94±0.45 | 17.20±0.38 | 11.86±0.77 | 8.85±0.61 | 7.12±0.61 | 22.01±0.66 | 18.98±0.41 | 17.30±0.34 | 12.04±0.67 | 9.01±0.50 | 7.33±0.51 |
| **UNet** | 21.94±0.67 | 18.94±0.45 | 17.20±0.38 | 11.86±0.77 | 8.85±0.61 | 7.12±0.61 | 22.01±0.66 | 18.98±0.41 | 17.30±0.34 | 12.04±0.67 | 9.01±0.50 | 7.33±0.51 |
| **SRCNN** | **23.64±0.50** | **20.04±0.39** | **18.00±0.35** | **12.87±0.35** | **9.26±0.32** | 7.22±0.30 | **23.99±0.58** | **20.31±0.42** | **18.07±0.37** | **13.18±0.38** | **9.50±0.27** | 7.25±0.21 |
| **ESRGAN** | 21.87±0.63 | 18.78±0.50 | 17.08±0.40 | 9.73±0.45 | 6.60±0.37 | 4.90±0.34 | 22.64±0.48 | 19.22±0.39 | 17.44±0.33 | 10.49±0.29 | 7.04±0.28 | 5.26±0.29 |
| **SwinIR** | 22.82±0.59 | 19.72±0.42 | 17.55±0.34 | 10.75±0.37 | 7.62±0.35 | 5.51±0.28 | 23.31±0.53 | 20.22±0.35 | 17.88±0.32 | 11.21±0.29 | 8.05±0.27 | 5.76±0.22 |
| **Nearest** | 23.74±0.54 | 20.11±0.42 | 18.00±0.36 | 13.04±0.35 | 9.40±0.30 | 7.29±0.26 | 23.71±0.54 | 20.19±0.41 | 18.00±0.36 | 13.03±0.34 | 9.51±0.26 | 7.31±0.23 |

#### SSIM Results ↑

| Method | Small FOV (195μm × 195μm) | | | | | | Large FOV (586μm × 586μm) | | | | | |
|--------|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Full** | | | **Masked** | | | **Full** | | | **Masked** | | |
| | Raw | Bone | Soft | Raw | Bone | Soft | Raw | Bone | Soft | Raw | Bone | Soft |
| **Baseline** | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 |
| **UNet** | 0.93±0.01 | 0.95±0.00 | 0.95±0.00 | 0.93±0.01 | 0.95±0.00 | **0.95±0.00** | **0.94±0.00** | **0.95±0.00** | **0.95±0.00** | **0.94±0.00** | **0.95±0.00** | **0.95±0.00** |
| **SRCNN** | 0.93±0.00 | 0.95±0.00 | 0.95±0.00 | 0.93±0.00 | 0.95±0.00 | 0.95±0.00 | 0.93±0.00 | 0.95±0.00 | 0.95±0.00 | 0.93±0.00 | 0.95±0.00 | 0.95±0.00 |
| **ESRGAN** | 0.93±0.00 | 0.94±0.00 | 0.94±0.00 | 0.82±0.04 | 0.82±0.04 | 0.82±0.04 | 0.93±0.00 | 0.95±0.00 | 0.95±0.00 | 0.82±0.03 | 0.82±0.04 | 0.82±0.03 |
| **SwinIR** | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 |
| **Nearest** | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 | 0.92±0.00 | 0.94±0.00 | 0.95±0.00 |

#### MAE Results ↓

| Method | Small FOV (195μm × 195μm) | | | | | | Large FOV (586μm × 586μm) | | | | | |
|--------|---|---|---|---|---|---|---|---|---|---|---|---|
| | **Full** | | | **Masked** | | | **Full** | | | **Masked** | | |
| | Raw | Bone | Soft | Raw | Bone | Soft | Raw | Bone | Soft | Raw | Bone | Soft |
| **Baseline** | 0.01±0.00 | 0.02±0.00 | 0.02±0.00 | 0.15±0.02 | 0.19±0.03 | 0.21±0.03 | 0.01±0.00 | 0.02±0.00 | 0.02±0.00 | 0.14±0.02 | 0.18±0.02 | 0.20±0.02 |
| **UNet** | **0.01±0.00** | 0.01±0.00 | 0.02±0.00 | 0.14±0.01 | 0.18±0.01 | 0.21±0.01 | **0.01±0.00** | **0.02±0.00** | **0.02±0.00** | **0.14±0.01** | 0.18±0.01 | 0.21±0.01 |
| **SRCNN** | **0.01±0.00** | **0.02±0.00** | **0.02±0.00** | **0.14±0.01** | **0.18±0.01** | **0.21±0.01** | **0.01±0.00** | **0.02±0.00** | **0.02±0.00** | **0.14±0.01** | **0.18±0.01** | **0.21±0.01** |
| **ESRGAN** | 0.01±0.00 | 0.02±0.00 | 0.02±0.00 | 0.22±0.01 | 0.31±0.02 | 0.35±0.03 | 0.01±0.00 | 0.02±0.00 | 0.02±0.00 | 0.21±0.01 | 0.28±0.02 | 0.32±0.02 |
| **SwinIR** | 0.01±0.00 | 0.02±0.00 | 0.02±0.00 | 0.21±0.01 | 0.28±0.02 | 0.32±0.02 | 0.01±0.00 | 0.02±0.00 | 0.02±0.00 | 0.20±0.01 | 0.27±0.01 | 0.31±0.01 |
| **Nearest** | 0.01±0.00 | 0.02±0.00 | 0.02±0.00 | 0.15±0.02 | 0.19±0.03 | 0.21±0.03 | 0.01±0.00 | 0.02±0.00 | 0.02±0.00 | 0.14±0.02 | 0.18±0.02 | 0.20±0.02 |

Detailed commands are provided in [`methods/README.md`](methods/README.md).

## Recommended Usage Policy

1. Use the released LumbarSR training set as the primary development data.
2. External data, pre-trained models, and synthetic data are allowed, but should be disclosed clearly.
3. Do not use hidden ground truth or any form of test leakage.

## Organizers

- **Ping Wang***, Institute of Diagnostic and Interventional Radiology, Shanghai Sixth People's Hospital Affiliated to Shanghai Jiao Tong University School of Medicine
- **Ruipeng Zhang***, Institute of Diagnostic and Interventional Radiology, Shanghai Sixth People's Hospital Affiliated to Shanghai Jiao Tong University School of Medicine
- **Mengfei Wang**, School of Information and Intelligent Science, Donghua University
- **Zhenzhen Cao**, Basic Medical Science, Kunming Medical University
- **Xuefei Hu**, Basic Medical Science, Tarim University School of Medicine
- **Yuehua Li**✉, Institute of Diagnostic and Interventional Radiology, Shanghai Sixth People's Hospital Affiliated to Shanghai Jiao Tong University School of Medicine (liyuehua77@sjtu.edu.cn)

\* Equal contribution

## Citation

If you use this dataset, please cite:

```bibtex
@misc{lumbarsr2026,
  title={LumbarSR: A Paired Clinical CT and Photon-Counting Micro-CT Dataset for Human Lumbar Vertebrae},
  author={Wang, Ping and Zhang, Ruipeng and Wang, Mengfei and Cao, Zhenzhen and Hu, Xuefei and Li, Yuehua},
  year={2026},
  url={https://github.com/frankzhangrp/LumbarSR-Challenge}
}
```

## Contact

- Email: zhangrp@sjtu.edu.cn
- GitHub: [@frankzhangrp](https://github.com/frankzhangrp)

## License

This dataset is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
