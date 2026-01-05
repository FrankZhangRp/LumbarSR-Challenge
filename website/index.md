---
layout: default
title: LumbarSR Challenge
---

# LumbarSR Challenge

## Lumbar Vertebral Super-Resolution Structure Reconstruction

Welcome to the LumbarSR Challenge! This challenge focuses on reconstructing high-resolution Micro-PCCT images from clinical CT scans of lumbar vertebrae.

### Quick Links

- [Dataset Download](#dataset)
- [Evaluation](#evaluation)
- [Submission](#submission)
- [Leaderboard](#leaderboard)

---

## Overview

Super-resolution reconstruction of CT images is crucial for improving diagnostic accuracy in clinical settings. This challenge aims to develop algorithms that can enhance the resolution of clinical CT scans to match the quality of Micro-PCCT imaging.

## Task

**Input**: Clinical CT images (soft tissue kernel) at two FOVs
- Small FOV: 195um in-plane resolution
- Large FOV: 586um in-plane resolution

**Output**: High-resolution Micro-PCCT image (105um isotropic)

---

## Dataset

### Download

[Google Drive](https://drive.google.com/) (Link coming soon)

### Statistics

| Split | Samples | Clinical CT | Micro-PCCT |
|-------|---------|-------------|------------|
| Training | 30 | ✓ | ✓ |
| Validation Phase 1 | 5 | ✓ | ✓ |
| Validation Phase 2 | 5 | ✓ | Hidden |

---

## Evaluation

Metrics computed under bone window (WC=400, WW=1800):
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE (Mean Absolute Error)
- NCC (Normalized Cross Correlation)

---

## Timeline

- **Phase 1**: Training with paired data
- **Phase 2**: Final evaluation

---

## Contact

- Email: zhangrp@sjtu.edu.cn
- GitHub: [@frankzhangrp](https://github.com/frankzhangrp)
