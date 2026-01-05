---
layout: default
title: LumbarSR Challenge
---

# LumbarSR Challenge

## Lumbar Vertebral Super-Resolution Structure Reconstruction

Welcome to the LumbarSR Challenge! This challenge focuses on reconstructing high-resolution Micro-PCCT images from clinical CT scans of lumbar vertebrae.

<div class="quick-links">
  <a href="#overview" class="btn">Overview</a>
  <a href="#dataset" class="btn btn-outline">Dataset</a>
  <a href="#evaluation" class="btn btn-outline">Evaluation</a>
  <a href="#submission" class="btn btn-outline">Submission</a>
</div>

---

## Overview

Super-resolution reconstruction of CT images is crucial for improving diagnostic accuracy in clinical settings. This challenge aims to develop algorithms that can enhance the resolution of clinical CT scans to match the quality of Micro-PCCT imaging.

<div class="info-box">
<strong>Challenge Goal:</strong> Develop algorithms to reconstruct high-resolution Micro-PCCT images (105μm isotropic) from clinical CT scans at different resolutions and fields of view.
</div>

---

## Task

### Input

Clinical CT images reconstructed with **soft tissue kernel** at two different fields of view:

| FOV | In-plane Resolution | Slice Thickness |
|-----|---------------------|-----------------|
| Small FOV | 195μm | 500μm / 1000μm |
| Large FOV | 586μm | 500μm / 1000μm |

### Output

High-resolution **Micro-PCCT** image with **105μm isotropic** resolution.

---

## Dataset

### Download

<a href="https://drive.google.com/" class="btn">Google Drive</a> *(Link coming soon)*

### Data Statistics

| Split | Samples | Clinical CT | Micro-PCCT |
|-------|---------|-------------|------------|
| Training | 30 | Available | Available |
| Validation Phase 1 | 5 | Available | Available |
| Validation Phase 2 | 5 | Available | Hidden |

### Input Sequences

For each sample, we provide 4 clinical CT sequences with soft tissue kernel:

| Sequence | Resolution | Slice Thickness | FOV |
|----------|------------|-----------------|-----|
| 195X_195Y_500Z_S | 195μm | 500μm | Small |
| 195X_195Y_1000Z_S | 195μm | 1000μm | Small |
| 586X_586Y_500Z_S | 586μm | 500μm | Large |
| 586X_586Y_1000Z_S | 586μm | 1000μm | Large |

---

## Evaluation

### Metrics

All metrics are computed under **three different CT window settings**:

| Window | Window Center (WC) | Window Width (WW) | Purpose |
|--------|-------------------|-------------------|---------|
| **Raw** | - | - | Original HU values (-1024 ~ 3071) |
| **Bone** | 400 | 1800 | Bone structure visualization |
| **Soft Tissue** | 40 | 400 | Soft tissue visualization |

### Evaluation Metrics

- **PSNR** - Peak Signal-to-Noise Ratio (dB)
- **SSIM** - Structural Similarity Index
- **MAE** - Mean Absolute Error

<div class="info-box success">
<strong>Final Ranking:</strong> The final ranking will be determined by a weighted combination of metrics across all three window settings.
</div>

---

## Timeline

- **Phase 1**: Training and validation with paired data
- **Phase 2**: Final evaluation on hidden test set

---

## Submission

Submission details will be announced soon.

---

## Contact

- **Email**: [zhangrp@sjtu.edu.cn](mailto:zhangrp@sjtu.edu.cn)
- **GitHub**: [@frankzhangrp](https://github.com/frankzhangrp)

---

## Citation

If you use this dataset, please cite:

```bibtex
@misc{lumbarsr2026,
  title={LumbarSR: Lumbar Vertebral Super-Resolution Challenge},
  year={2026},
  url={https://github.com/frankzhangrp/LumbarSR-Challenge}
}
```
