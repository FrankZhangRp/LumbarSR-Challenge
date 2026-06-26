# Super-Resolution Methods

This directory contains runnable method scripts for the LumbarSR benchmark.

Official paper: [LumbarSR: A Paired Clinical CT and Photon-Counting Micro-CT Dataset for Human Lumbar Vertebrae](https://www.nature.com/articles/s41597-026-07748-5)

## Quick Start

### 1. Setup Environment

```bash
conda create -n lumbarsr python=3.10 -y
conda activate lumbarsr

pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies: PyTorch, MONAI, nibabel, pydicom, SimpleITK, numpy, scipy, and scikit-image.

### 2. Prepare Data

Ensure your data follows this structure:
```
data/
├── RegisteredData/
│   ├── Lumbar_01/
│   │   ├── Lumbar01_ClinicalCT_195X_195Y_1000Z_S_registered.nii.gz
│   │   ├── Lumbar01_ClinicalCT_586X_586Y_1000Z_S_registered.nii.gz
│   │   └── Lumbar01_MicroPCCT_105um.nii.gz
│   ├── Lumbar_02/
│   └── ...
└── BoneMask/
    ├── Lumbar_01/
    │   ├── Lumbar01_MicroPCCT_105um_BoneMask.nii.gz
    │   ├── Lumbar01_ClinicalCT_195X_195Y_1000Z_S_registered_BoneMask.nii.gz
    │   └── Lumbar01_ClinicalCT_586X_586Y_1000Z_S_registered_BoneMask.nii.gz
    └── ...
```

## Methods

Runnable implementations in this directory:

- Interpolation baselines
- SRCNN
- UNet

Public benchmark result tables in the main repository additionally report registered clinical CT, `Nearest`, `ESRGAN`, and `SwinIR` under the same released evaluation protocol.

### 1. Interpolation Baselines

Traditional interpolation methods (fast, no training required).

**Run all interpolation methods:**
```bash
# Nearest neighbor
python methods/interpolation.py --method nearest --data-root data/RegisteredData --output-root results

# Trilinear
python methods/interpolation.py --method trilinear --data-root data/RegisteredData --output-root results

# Bicubic
python methods/interpolation.py --method bicubic --data-root data/RegisteredData --output-root results

# Lanczos
python methods/interpolation.py --method lanczos --data-root data/RegisteredData --output-root results
```

### 2. SRCNN

**Training:**
```bash
# Train SRCNN (single-sequence soft-kernel input)
python methods/train.py \
  --model srcnn \
  --data-root data/RegisteredData \
  --output-dir checkpoints \
  --sequence 195X_195Y_1000Z_S \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4

# Optional dual-kernel training (same FOV: B + S)
python methods/train.py \
  --model srcnn \
  --data-root data/RegisteredData \
  --output-dir checkpoints \
  --sequence 195X_195Y_1000Z_S \
  --dual-channel \
  --paired-sequence 195X_195Y_1000Z_B \
  --epochs 100 \
  --batch-size 4
```

**Inference:**
```bash
# Generate predictions on test set
python methods/inference.py \
  --model srcnn \
  --checkpoint checkpoints/srcnn_195X_195Y_1000Z_S_best.pth \
  --data-root data/RegisteredData \
  --output-root results \
  --sequence 195X_195Y_1000Z_S \
  --batch-size 64
```

### 3. UNet

**Training:**
```bash
# Train UNet (single-sequence soft-kernel input)
python methods/train.py \
  --model unet \
  --data-root data/RegisteredData \
  --output-dir checkpoints \
  --sequence 195X_195Y_1000Z_S \
  --epochs 100 \
  --batch-size 4 \
  --lr 1e-4
```

**Inference:**
```bash
# Generate predictions
python methods/inference.py \
  --model unet \
  --checkpoint checkpoints/unet_195X_195Y_1000Z_S_best.pth \
  --data-root data/RegisteredData \
  --output-root results \
  --sequence 195X_195Y_1000Z_S \
  --batch-size 32
```

Registration baseline code is available separately in [`../baseline/`](../baseline/), while public evaluation code is available in [`../evaluation/`](../evaluation/).

## Model Notes

- `SRCNN`: 2D convolutional baseline with `~60K` parameters
- `UNet`: 2D encoder-decoder baseline with `~30M` parameters
- Default input is the soft-kernel sequence of one FOV; optional dual-channel input uses the matching `B + S` pair from the same FOV

## Training Tips

1. Use the soft-kernel sequence of one FOV as the default input.
2. If dual-channel input is used, pair `B + S` from the same FOV.
3. Training is patch-based by default; inference is full-slice batched.
4. Predictions are saved under `results/<method>/<sample>/LumbarXX_<sequence>_<method>.nii.gz`.

## Evaluation

After generating predictions, use:

```bash
# Evaluate predictions against ground truth inside the released BoneMask ROI
python evaluation/batch_evaluate.py \
  --pred-root results \
  --gt-root data/RegisteredData \
  --bone-mask-root data/BoneMask \
  --methods srcnn unet \
  --output-dir outputs/metrics
```

The public evaluation reports PSNR, SSIM, and MAE under raw, bone, and soft-tissue windows, with BoneMask-ROI metrics as the main reported setting.

## File Structure

```
methods/
├── interpolation.py      # Traditional interpolation baselines
├── srcnn_model.py        # SRCNN model definition
├── unet_model.py         # UNet model definition
├── train.py              # Training script
├── inference.py          # Inference script
└── README.md             # This file
```

## Expected Results

Training uses `Lumbar_01` to `Lumbar_25`; evaluation reports mean ± standard deviation on `Lumbar_26` to `Lumbar_30`.

**Key Performance Summary:**

| Method | PSNR (dB) ↑ | SSIM ↑ | MAE ↓ |
|--------|------------|--------|-------|
| Baseline (registered CT) | 8.85-9.01 | 0.9427-0.9450 | 0.178-0.196 |
| UNet | 8.85-9.50 | **0.9489-0.9500** | 0.179-0.214 |
| SRCNN | **9.26-9.51** | 0.9450-0.9489 | **0.178-0.210** |
| Nearest Interpolation | 8.85-9.51 | 0.9427-0.9450 | 0.178-0.196 |

*Range shows performance across different FOV (Small/Large) and window (Raw/Bone/Soft) configurations in BoneMask-ROI evaluation*

For complete public benchmark tables, see the [full results page](../docs/baseline_results.html) or the main [README](../README.md#baseline-performance).

## Citation

If you use these baseline methods, please cite:

```bibtex
@article{wang2026lumbarsr,
  title={LumbarSR: A Paired Clinical CT and Photon-Counting Micro-CT Dataset for Human Lumbar Vertebrae},
  author={Wang, Ping and Zhang, Ruipeng and Wang, Mengfei and Zong, Shenyan and Zhu, Jinyu and Song, Xinyu and Cao, Zhenzhen and Hu, Xuefei and Wang, Dan and Li, Yuehua},
  journal={Scientific Data},
  year={2026},
  month={jun},
  doi={10.1038/s41597-026-07748-5},
  url={https://www.nature.com/articles/s41597-026-07748-5},
  publisher={Springer Science and Business Media LLC}
}

@misc{lumbarsr2026,
  title={LumbarSR: A Paired Clinical CT and Photon-Counting Micro-CT Dataset for Human Lumbar Vertebrae},
  author={Wang, Ping and Zhang, Ruipeng and Wang, Mengfei and Zong, Shenyan and Zhu, Jinyu and Song, Xinyu and Cao, Zhenzhen and Hu, Xuefei and Wang, Dan and Li, Yuehua},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/FrankZhangRp/LumbarSR-Challenge}
}
```

## Contact

- Email: zhangrp@sjtu.edu.cn
- GitHub: [@frankzhangrp](https://github.com/frankzhangrp)
