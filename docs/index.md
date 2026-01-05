---
layout: default
title: LumbarSR Challenge
---

# LumbarSR Challenge

## Lumbar Vertebral Super-Resolution Structure Reconstruction Challenge

<div class="quick-links">
  <a href="#overview" class="btn">Overview</a>
  <a href="#dataset" class="btn btn-outline">Dataset</a>
  <a href="#evaluation" class="btn btn-outline">Evaluation</a>
  <a href="#timeline" class="btn btn-outline">Timeline</a>
</div>

---

## Overview

Medical computed tomography (CT) plays a crucial role in disease diagnosis, with CT technology having undergone significant evolution over the past half-century—from primary CT to energy-integrating detector CT, and now to photon-counting detector CT (PCCT) systems being deployed in clinical centers worldwide. Low back pain is a prevalent symptom affecting approximately 540 million people globally at any given time, with lumbar vertebral microstructure changes being a potential osseous factor contributing to this condition. However, these microstructural morphological changes remain largely undetectable due to current limitations in standard clinical CT equipment resolution.

<div class="info-box">
<strong>Challenge Goal:</strong> Develop algorithms capable of reconstructing high-resolution bone microstructure from paired regular clinical helical CT lumbar vertebral images to match the quality of state-of-the-art Micro-PCCT.
</div>

The LumbarSR challenge provides a unique dataset of **30 paired human dry lumbar vertebrae** scanned with both clinical helical CT (at multiple resolution and reconstruction configurations) and Micro-PCCT at 0.1mm super-resolution. This novel benchmark enables development and evaluation of super-resolution reconstruction algorithms in musculoskeletal imaging, with potential applications in:

- Osteoporosis screening
- Fracture risk assessment
- Low back pain investigation

The technical objective is to develop algorithms that can transform clinical helical CT images (0.5-1.0 mm resolution) into high-resolution images comparable to Micro-PCCT (0.1 mm), representing a **10-200× super-resolution enhancement factor**.

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

### Ground Truth

- **Resolution**: 105μm isotropic
- **Format**: NIfTI (.nii.gz)
- **Data type**: int16 (Hounsfield Units)

---

## Evaluation

### CT Window Settings

All metrics are computed under **three different CT window settings**:

| Window | Window Center (WC) | Window Width (WW) | Purpose |
|--------|-------------------|-------------------|---------|
| **Raw** | - | - | Original HU values (-1024 ~ 3071) |
| **Bone** | 400 | 1800 | Bone structure visualization |
| **Soft Tissue** | 40 | 400 | Soft tissue visualization |

### Metrics

- **PSNR** - Peak Signal-to-Noise Ratio (dB)
- **SSIM** - Structural Similarity Index
- **MAE** - Mean Absolute Error

<div class="info-box success">
<strong>Final Ranking:</strong> The final ranking will be determined by a weighted combination of metrics across all three window settings.
</div>

---

## Training Data Policy

1. **Challenge Data**: Participants are encouraged to use the official LumbarSR training set as the core development data.

2. **External Data (Allowed)**: External datasets are allowed for pre-training and/or supplementary training. Teams must disclose the external data sources in their method description.

3. **Pre-trained / Foundation Models (Allowed)**: Use of any publicly available pre-trained models and foundation models is permitted. Model name/version must be disclosed.

4. **Synthetic Data (Allowed)**: Synthetic data generation (e.g., augmentation, simulation, self-training) is allowed and should be briefly described.

5. **Fair-Play / No Test Leakage**: No use of hidden test ground truth or any form of test-set leakage. Inference must be fully automatic with no case-by-case manual tuning.

---

## Timeline

> **Note**: Schedule is tentative and subject to change.

| Milestone | Date |
|-----------|------|
| Challenge website launch | April 1-10, 2026 |
| Training data release | April 15, 2026 |
| Registration opens | April 15, 2026 |
| Validation data release | May 1, 2026 |
| Submission system opens | June 1, 2026 |
| Public test phase begins | July 1, 2026 |
| Submission deadline | August 15, 2026 |
| Hidden test evaluation | August 16-31, 2026 |
| Results notification | September 5, 2026 |
| Challenge Event | TBD |

---

## Organizers

- **Ruipeng Zhang** - Shanghai Sixth People's Hospital, SJTU
- **Ping Wang** - Shanghai Sixth People's Hospital, SJTU
- **Mengfei Wang** - Donghua University
- **Zhenzhen Cao** - Kunming Medical University
- **Xuefei Hu** - Tarim University School of Medicine
- **Yuehua Li** - Shanghai Sixth People's Hospital, SJTU

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
