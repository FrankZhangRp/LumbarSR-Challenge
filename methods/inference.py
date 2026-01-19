#!/usr/bin/env python3
"""
Inference script for super-resolution models.

Supports SRCNN and UNet for generating super-resolved CT images.
"""

import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

import torch


def load_model(checkpoint_path, model_type, device):
    """Load trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        model_type: 'srcnn' or 'unet'
        device: torch.device

    Returns:
        Loaded model
    """
    if model_type == 'srcnn':
        from srcnn_model import get_model
        model = get_model('2d', in_channels=2, num_features=64)
    elif model_type == 'unet':
        from unet_model import get_model
        model = get_model('2d', in_channels=2, base_features=64)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")

    return model


def denormalize(x):
    """Convert normalized [0,1] back to HU units."""
    return x * 4095.0 - 1024


def normalize_hu(x):
    """Normalize HU to [0, 1]."""
    x = np.clip(x, -1024, 3071)
    return (x + 1024) / 4095.0


def inference_volume(model, lr_small_path, lr_large_path, gt_path, device, patch_size=256, overlap=32):
    """Run inference on a single volume.

    Args:
        model: Trained SR model
        lr_small_path: Path to small FOV LR
        lr_large_path: Path to large FOV LR
        gt_path: Path to GT (for reference shape)
        device: torch.device
        patch_size: Patch size for inference
        overlap: Overlap between patches

    Returns:
        Super-resolved volume (numpy array)
    """
    # Load volumes
    lr_small = nib.load(lr_small_path).get_fdata().astype(np.float32)
    lr_large = nib.load(lr_large_path).get_fdata().astype(np.float32)
    gt_nii = nib.load(gt_path)

    # Normalize
    lr_small = normalize_hu(lr_small)
    lr_large = normalize_hu(lr_large)

    # Get dimensions
    H, W, D = lr_small.shape

    # Resize large FOV to match small FOV
    from scipy.ndimage import zoom
    scale_h = H / lr_large.shape[0]
    scale_w = W / lr_large.shape[1]
    lr_large = zoom(lr_large, (scale_h, scale_w, 1.0), order=1)

    # Initialize output
    sr_volume = np.zeros_like(lr_small)
    count_map = np.zeros_like(lr_small)

    # Process slice by slice
    with torch.no_grad():
        for d in tqdm(range(D), desc="Processing slices"):
            lr_small_slice = lr_small[:, :, d]
            lr_large_slice = lr_large[:, :, d]

            # Extract patches
            for y in range(0, H, patch_size - overlap):
                for x in range(0, W, patch_size - overlap):
                    # Extract patch
                    y_end = min(y + patch_size, H)
                    x_end = min(x + patch_size, W)

                    lr_small_patch = lr_small_slice[y:y_end, x:x_end]
                    lr_large_patch = lr_large_slice[y:y_end, x:x_end]

                    # Resize to patch_size if needed
                    if lr_small_patch.shape[0] < patch_size or lr_small_patch.shape[1] < patch_size:
                        pad_h = patch_size - lr_small_patch.shape[0]
                        pad_w = patch_size - lr_small_patch.shape[1]
                        lr_small_patch = np.pad(lr_small_patch, ((0, pad_h), (0, pad_w)), mode='constant')
                        lr_large_patch = np.pad(lr_large_patch, ((0, pad_h), (0, pad_w)), mode='constant')

                    # Stack channels
                    lr_input = np.stack([lr_small_patch, lr_large_patch], axis=0)
                    lr_tensor = torch.from_numpy(lr_input).unsqueeze(0).float().to(device)

                    # Predict
                    sr_tensor = model(lr_tensor)

                    # Convert back to numpy
                    sr_patch = sr_tensor.squeeze().cpu().numpy()

                    # Remove padding if added
                    if y_end - y < patch_size or x_end - x < patch_size:
                        sr_patch = sr_patch[:y_end-y, :x_end-x]

                    # Accumulate output
                    sr_volume[y:y_end, x:x_end, d] += sr_patch
                    count_map[y:y_end, x:x_end, d] += 1

    # Average overlapping regions
    sr_volume = sr_volume / (count_map + 1e-8)

    # Denormalize
    sr_volume = denormalize(sr_volume)
    sr_volume = np.clip(sr_volume, -1024, 32767).astype(np.int16)

    return sr_volume, gt_nii.affine


def batch_inference(
    model,
    data_root,
    output_root,
    samples,
    device,
    patch_size=256,
    overlap=32
):
    """Run inference on multiple samples.

    Args:
        model: Trained SR model
        data_root: Root directory of data
        output_root: Output directory
        samples: List of sample names
        device: torch.device
        patch_size: Patch size
        overlap: Overlap between patches
    """
    print(f"\n{'#'*60}")
    print(f"# Running Inference on {len(samples)} samples")
    print(f"{'#'*60}\n")

    for sample in tqdm(samples, desc="Samples"):
        sample_id = sample.split("_")[-1]
        sample_dir = Path(data_root) / sample

        # Paths
        lr_small = sample_dir / f"Lumbar{sample_id}_ClinicalCT_195X_195Y_1000Z_S_registered.nii.gz"
        lr_large = sample_dir / f"Lumbar{sample_id}_ClinicalCT_586X_586Y_1000Z_S_registered.nii.gz"
        gt = sample_dir / f"Lumbar{sample_id}_MicroPCCT_105um.nii.gz"

        if not all([lr_small.exists(), lr_large.exists(), gt.exists()]):
            print(f"  [SKIP] {sample} - missing files")
            continue

        # Run inference
        print(f"\n[{sample}]")
        sr_volume, sr_affine = inference_volume(
            model, str(lr_small), str(lr_large), str(gt),
            device, patch_size, overlap
        )

        # Save result
        output_dir = Path(output_root) / sample
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"Lumbar{sample_id}_SR_pred.nii.gz"
        sr_nii = nib.Nifti1Image(sr_volume, sr_affine)
        nib.save(sr_nii, str(output_path))

        print(f"  Saved: {output_path}")

    print(f"\n{'#'*60}")
    print(f"# Inference complete!")
    print(f"# Results saved to: {output_root}")
    print(f"{'#'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="SR model inference")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--model', type=str, default='srcnn', choices=['srcnn', 'unet'])
    parser.add_argument('--data-root', type=str, default='/data/LumbarSR/registered_nifti')
    parser.add_argument('--output-root', type=str, default='/data/LumbarSR/results')
    parser.add_argument('--samples', nargs='+', default=None, help="Samples to process (default: Lumbar_26-30)")
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Samples
    if args.samples is None:
        samples = [f"Lumbar_{i:02d}" for i in range(26, 31)]
    else:
        samples = args.samples

    print(f"Processing {len(samples)} samples")

    # Load model
    model = load_model(args.checkpoint, args.model, device)

    # Run inference
    batch_inference(
        model,
        args.data_root,
        args.output_root,
        samples,
        device,
        patch_size=args.patch_size,
        overlap=args.overlap
    )


if __name__ == "__main__":
    main()
