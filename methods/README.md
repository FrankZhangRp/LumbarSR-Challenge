# Super-Resolution Methods

This directory contains baseline implementations for the LumbarSR Challenge.

## Quick Start

### 1. Setup Environment

**Option A: Docker (Recommended)**
```bash
# Build Docker image
docker build -t lumbarsr:latest .

# Run container
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/results:/workspace/results \
  lumbarsr:latest

# Or use docker-compose
docker-compose up -d
docker-compose exec lumbarsr bash
```

**Option B: Local Installation**
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure your data follows this structure:
```
data/
├── registered_nifti/
│   ├── Lumbar_01/
│   │   ├── Lumbar01_ClinicalCT_195X_195Y_1000Z_S_registered.nii.gz
│   │   ├── Lumbar01_ClinicalCT_586X_586Y_1000Z_S_registered.nii.gz
│   │   └── Lumbar01_MicroPCCT_105um.nii.gz
│   ├── Lumbar_02/
│   └── ...
```

## Methods

### 1. Interpolation Baselines

Traditional interpolation methods (fast, no training required).

**Run all interpolation methods:**
```bash
# Nearest neighbor
python methods/interpolation.py --method nearest --data-root data/registered_nifti --output-root results

# Trilinear
python methods/interpolation.py --method trilinear --data-root data/registered_nifti --output-root results

# Bicubic
python methods/interpolation.py --method bicubic --data-root data/registered_nifti --output-root results

# Lanczos
python methods/interpolation.py --method lanczos --data-root data/registered_nifti --output-root results
```

### 2. SRCNN (Super-Resolution CNN)

Deep learning approach with patch-based training.

**Training:**
```bash
# Train SRCNN (single-channel input)
python methods/train.py \
  --model srcnn \
  --data-root data/registered_nifti \
  --output-dir checkpoints \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4

# Train SRCNN (dual-channel: small FOV + large FOV)
python methods/train.py \
  --model srcnn \
  --data-root data/registered_nifti \
  --output-dir checkpoints \
  --dual-channel \
  --epochs 100 \
  --batch-size 4
```

**Inference:**
```bash
# Generate predictions on test set
python methods/inference.py \
  --model srcnn \
  --checkpoint checkpoints/srcnn_best.pth \
  --data-root data/registered_nifti \
  --output-root results \
  --patch-size 256
```

### 3. UNet

U-shaped architecture with encoder-decoder and skip connections.

**Training:**
```bash
# Train UNet (dual-channel recommended)
python methods/train.py \
  --model unet \
  --data-root data/registered_nifti \
  --output-dir checkpoints \
  --dual-channel \
  --epochs 100 \
  --batch-size 4 \
  --lr 1e-4
```

**Inference:**
```bash
# Generate predictions
python methods/inference.py \
  --model unet \
  --checkpoint checkpoints/unet_best.pth \
  --data-root data/registered_nifti \
  --output-root results
```

## Model Architecture

### SRCNN
- **Input**: 2-channel (small FOV + large FOV)
- **Layers**:
  - Conv1: 9×9, 64 filters, ReLU
  - Conv2: 1×1, 32 filters, ReLU
  - Conv3: 5×5, 1 filter
- **Parameters**: ~60K
- **Training Loss**: L1

### UNet
- **Input**: 2-channel (small FOV + large FOV)
- **Architecture**: Encoder-decoder with skip connections
  - 4 encoder levels (64 → 512 filters)
  - 4 decoder levels with concatenation
  - Final 1×1 conv to output
- **Parameters**: ~30M
- **Training Loss**: L1

## Training Tips

1. **Dual-channel input**: Using both small and large FOV as input channels improves performance
2. **Patch-based training**: Extract random patches (256×256) from slices for memory efficiency
3. **Data augmentation**: Can add rotation, flipping in `train.py` for better generalization
4. **Batch size**: Adjust based on GPU memory (UNet requires more memory than SRCNN)
5. **Learning rate**: Start with 1e-4, reduce on plateau

## Evaluation

After generating predictions, use the evaluation script:

```bash
# Evaluate predictions against ground truth
python evaluation/batch_evaluate.py \
  --pred-root results \
  --gt-root data/registered_nifti \
  --output-dir outputs/metrics
```

Metrics computed:
- **PSNR** (Peak Signal-to-Noise Ratio, dB)
- **SSIM** (Structural Similarity Index)
- **MAE** (Mean Absolute Error)

Metrics are calculated under:
- Raw HU values
- Bone window (WC=400, WW=1800)
- Soft tissue window (WC=40, WW=400)

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

**Experimental Setup:**
- **Training Set**: Lumbar_01 to Lumbar_25 (25 samples)
- **Test Set**: Lumbar_26 to Lumbar_30 (5 samples)
- **Training**: Deep learning methods trained on 25 samples with dual-channel input
- **Evaluation**: Metrics computed as mean ± standard deviation across test set

**Key Performance Summary:**

| Method | PSNR (dB) ↑ | SSIM ↑ | MAE ↓ |
|--------|------------|--------|-------|
| Baseline (registered CT) | 8.85-9.01 | 0.9427-0.9450 | 0.178-0.196 |
| UNet | 8.85-9.50 | **0.9489-0.9500** | 0.179-0.214 |
| SRCNN | **9.26-9.51** | 0.9450-0.9489 | **0.178-0.210** |
| Nearest Interpolation | 8.85-9.51 | 0.9427-0.9450 | 0.178-0.196 |

*Range shows performance across different FOV (Small/Large) and window (Raw/Bone/Soft) configurations in masked evaluation*

**Key Findings:**
- **SRCNN** achieves the best PSNR with 8-9% improvement over baseline
- **UNet** excels at structural preservation with the highest SSIM scores
- **Masked evaluation** (focusing on anatomical regions) shows more discriminating metrics
- Deep learning methods show consistent improvements in both Small and Large FOV configurations

For complete results tables with all configurations (FOV × Mode × Window), see the [full results page](../docs/baseline_results.html) or the main [README](../README.md#baseline-performance).

## Troubleshooting

**Out of Memory (OOM)**:
- Reduce `--batch-size`
- Reduce `--patch-size`
- Use single-channel instead of dual-channel

**Slow Training**:
- Reduce `--n-patches` (fewer patches per volume)
- Use smaller model (SRCNN instead of UNet)

**Poor Results**:
- Train for more epochs
- Try different learning rates
- Use dual-channel input
- Add data augmentation

## Citation

If you use these baseline methods, please cite:

```bibtex
@misc{lumbarsr2026,
  title={LumbarSR: Lumbar Vertebral Super-Resolution Challenge},
  year={2026},
  url={https://github.com/frankzhangrp/LumbarSR-Challenge}
}
```

## Contact

- Email: zhangrp@sjtu.edu.cn
- GitHub: [@frankzhangrp](https://github.com/frankzhangrp)
