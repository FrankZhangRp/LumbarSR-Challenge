# LumbarSR Challenge

**Lumbar Vertebral Super-Resolution Structure Reconstruction Challenge**

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Overview

LumbarSR is a medical imaging challenge focused on super-resolution reconstruction of lumbar vertebral CT images. The goal is to reconstruct high-resolution Photon-Counting CT (PCCT) images from clinical CT scans.

## Task Description

Given clinical CT images acquired with **soft tissue reconstruction kernel** at two different fields of view (FOV):
- **Small FOV** (195um in-plane resolution)
- **Large FOV** (586um in-plane resolution)

Reconstruct the corresponding high-resolution Micro-PCCT image (105um isotropic resolution).

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

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **MAE** (Mean Absolute Error)
- **NCC** (Normalized Cross Correlation)

Metrics are computed under bone window (WC=400, WW=1800).

## Submission

Details coming soon.

## Timeline

- **Phase 1**: Training and validation with paired data
- **Phase 2**: Final evaluation on hidden test set

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
