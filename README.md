# Lumbar Vertebral Super-Resolution Structure Reconstruction Challenge

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

Medical computed tomography (CT) plays a crucial role in disease diagnosis, with CT technology having undergone significant evolution over the past half-century—from primary CT to energy-integrating detector CT, and now to photon-counting detector CT (PCCT) systems being deployed in clinical centers worldwide. Low back pain is a prevalent symptom affecting approximately 540 million people globally at any given time, with lumbar vertebral microstructure changes being a potential osseous factor contributing to this condition. However, these microstructural morphological changes remain largely undetectable due to current limitations in standard clinical CT equipment resolution.

This challenge addresses the critical need for algorithms capable of reconstructing high-resolution bone microstructure from paired regular clinical helical CT lumbar vertebral images to match the quality of state-of-the-art Micro-PCCT. The LumbarSR challenge provides a unique dataset of 30 paired human dry lumbar vertebrae scanned with both clinical helical CT (at multiple resolution and reconstruction configurations) and Micro-PCCT at 0.1mm super-resolution. This novel benchmark enables development and evaluation of super-resolution reconstruction algorithms in musculoskeletal imaging, with potential applications in osteoporosis screening, fracture risk assessment, and low back pain investigation.

The technical objective is to develop algorithms that can transform clinical helical CT images (0.5-1.0 mm resolution) into high-resolution images comparable to Micro-PCCT (0.1 mm), representing a 10-200× super-resolution enhancement factor. Success in this challenge could transform routine lumbar CT scans into diagnostic tools for bone microstructure assessment, advancing AI-based image reconstruction in musculoskeletal radiology.

<p align="center">
  <a href="docs/images/showcase.gif">
    <img src="docs/images/showcase_preview.gif" alt="Multi-resolution CT Visualization" width="100%">
  </a>
  <br>
  <em>Tri-planar views and 3D bone rendering across different CT resolutions (<a href="docs/images/showcase.gif">Full resolution</a>)</em>
</p>

## Task Description

Given clinical CT images acquired with **soft tissue reconstruction kernel** at two different fields of view (FOV):
- **Small FOV** (195um in-plane resolution)
- **Large FOV** (586um in-plane resolution)

Reconstruct the corresponding high-resolution Micro-PCCT image (105um isotropic resolution).

<p align="center">
  <a href="docs/images/flowchart.png">
    <img src="docs/images/flowchart_preview.png" alt="Challenge Flowchart" width="100%">
  </a>
  <br>
  <em>Challenge pipeline overview (<a href="docs/images/flowchart.png">Full resolution</a>)</em>
</p>

## Dataset

### Download

**[Google Drive Link]** (Coming Soon)

### Data Split

| Split | Samples | Clinical CT | Micro-PCCT |
|-------|---------|-------------|------------|
| Training | 30 | Paired | Paired |
| Validation Phase 1 | 5 | Paired | Paired |
| Validation Phase 2 | 5 | Available | Hidden |

### Input Data (Clinical CT)

For each sample, we provide 4 sequences with soft tissue kernel:

| Sequence | In-plane Resolution | Slice Thickness | FOV |
|----------|---------------------|-----------------|-----|
| 195X_195Y_500Z_S | 195um | 500um | Small |
| 195X_195Y_1000Z_S | 195um | 1000um | Small |
| 586X_586Y_500Z_S | 586um | 500um | Large |
| 586X_586Y_1000Z_S | 586um | 1000um | Large |

### Ground Truth (Micro-PCCT)

- Resolution: 105um isotropic
- Format: NIfTI (.nii.gz)
- Data type: int16 (Hounsfield Units)

### Dataset Statistics

- **Training Set**: 30 samples, 212,805 slices total (avg 7,093.5 slices/case, registered to 105μm isotropic)

## Evaluation Metrics

### Metrics

- **PSNR** (Peak Signal-to-Noise Ratio) - measured in dB
- **SSIM** (Structural Similarity Index)
- **MAE** (Mean Absolute Error)

### CT Window Settings

All metrics are computed under three different CT window settings:

| Window | Window Center (WC) | Window Width (WW) | Description |
|--------|-------------------|-------------------|-------------|
| Raw | - | - | Original HU values (-1024 ~ 3071) |
| Bone | 400 | 1800 | Bone structure visualization |
| Soft Tissue | 40 | 400 | Soft tissue visualization |

## Submission

Details coming soon.

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

## Training Data Policy

1. **Challenge Data**: Participants are encouraged to use the official LumbarSR training set as the core development data.

2. **External Data (Allowed)**: To maximize performance, external datasets are allowed for pre-training and/or supplementary training (public or legally/ethically permitted institutional data). Teams must disclose the external data sources in their method description (name/type and approximate scale).

3. **Pre-trained / Foundation Models (Allowed)**: Use of any publicly available pre-trained models and foundation models is permitted. Model name/version (or checkpoint) must be disclosed.

4. **Synthetic Data (Allowed)**: Synthetic data generation (e.g., augmentation, simulation, self-training) is allowed and should be briefly described.

5. **Fair-Play / No Test Leakage**: No use of hidden test ground truth or any form of test-set leakage. Inference must be fully automatic with no case-by-case manual tuning.

## Organizers

- **Ruipeng Zhang**, Institute of Diagnostic and Interventional Radiology, Shanghai Sixth People's Hospital Affiliated to Shanghai Jiao Tong University School of Medicine
- **Ping Wang**, Institute of Diagnostic and Interventional Radiology, Shanghai Sixth People's Hospital Affiliated to Shanghai Jiao Tong University School of Medicine
- **Mengfei Wang**, School of Information and Intelligent Science, Donghua University
- **Zhenzhen Cao**, Basic Medical Science, Kunming Medical University
- **Xuefei Hu**, Basic Medical Science, Tarim University School of Medicine
- **Yuehua Li**✉, Institute of Diagnostic and Interventional Radiology, Shanghai Sixth People's Hospital Affiliated to Shanghai Jiao Tong University School of Medicine (liyuehua77@sjtu.edu.cn)

## Citation

If you use this dataset, please cite:

```bibtex
@misc{lumbarsr2026,
  title={LumbarSR: Lumbar Vertebral Super-Resolution Structure Reconstruction Challenge},
  year={2026},
  url={https://github.com/frankzhangrp/LumbarSR-Challenge}
}
```

## Contact

- Email: zhangrp@sjtu.edu.cn
- GitHub: [@frankzhangrp](https://github.com/frankzhangrp)

## License

This dataset is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
