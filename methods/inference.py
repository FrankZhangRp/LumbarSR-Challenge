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


def infer_paired_sequence(sequence):
    """Infer the same-FOV paired kernel sequence."""
    if sequence.endswith("_S"):
        return sequence[:-2] + "_B"
    if sequence.endswith("_B"):
        return sequence[:-2] + "_S"
    raise ValueError(f"Cannot infer paired sequence from: {sequence}")


def load_model(checkpoint_path, model_type, device, in_channels=1):
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
        model = get_model('2d', in_channels=in_channels, num_features=64)
    elif model_type == 'unet':
        from unet_model import get_model
        model = get_model('2d', in_channels=in_channels, base_features=64)
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


def inference_volume(model, lr_path, gt_path, device, paired_path=None, patch_size=256, overlap=32):
    """Run inference on a single volume.

    Args:
        model: Trained SR model
        lr_path: Path to primary LR input
        gt_path: Path to GT (for reference shape)
        device: torch.device
        paired_path: Optional path to same-FOV paired kernel input
        patch_size: Patch size for inference
        overlap: Overlap between patches

    Returns:
        Super-resolved volume (numpy array)
    """
    # Load volumes
    lr = nib.load(lr_path).get_fdata().astype(np.float32)
    paired = nib.load(paired_path).get_fdata().astype(np.float32) if paired_path is not None else None
    gt_nii = nib.load(gt_path)

    # Normalize
    lr = normalize_hu(lr)
    if paired is not None:
        paired = normalize_hu(paired)
        if paired.shape != lr.shape:
            raise ValueError(f"Paired sequence shape mismatch: {paired.shape} vs {lr.shape}")

    # Get dimensions
    H, W, D = lr.shape

    # Initialize output
    sr_volume = np.zeros_like(lr)
    count_map = np.zeros_like(lr)

    # Process slice by slice
    with torch.no_grad():
        for d in tqdm(range(D), desc="Processing slices"):
            lr_slice = lr[:, :, d]
            paired_slice = paired[:, :, d] if paired is not None else None

            # Extract patches
            for y in range(0, H, patch_size - overlap):
                for x in range(0, W, patch_size - overlap):
                    # Extract patch
                    y_end = min(y + patch_size, H)
                    x_end = min(x + patch_size, W)

                    lr_patch = lr_slice[y:y_end, x:x_end]
                    paired_patch = paired_slice[y:y_end, x:x_end] if paired_slice is not None else None

                    # Resize to patch_size if needed
                    if lr_patch.shape[0] < patch_size or lr_patch.shape[1] < patch_size:
                        pad_h = patch_size - lr_patch.shape[0]
                        pad_w = patch_size - lr_patch.shape[1]
                        lr_patch = np.pad(lr_patch, ((0, pad_h), (0, pad_w)), mode='constant')
                        if paired_patch is not None:
                            paired_patch = np.pad(paired_patch, ((0, pad_h), (0, pad_w)), mode='constant')

                    if paired_patch is None:
                        lr_input = lr_patch[np.newaxis, ...]
                    else:
                        lr_input = np.stack([lr_patch, paired_patch], axis=0)
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
    sequence,
    dual_channel=False,
    paired_sequence=None,
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
        lr_path = sample_dir / f"Lumbar{sample_id}_ClinicalCT_{sequence}_registered.nii.gz"
        resolved_paired_sequence = paired_sequence or infer_paired_sequence(sequence)
        paired_path = sample_dir / f"Lumbar{sample_id}_ClinicalCT_{resolved_paired_sequence}_registered.nii.gz" if dual_channel else None
        gt = sample_dir / f"Lumbar{sample_id}_MicroPCCT_105um.nii.gz"

        required_paths = [lr_path, gt]
        if paired_path is not None:
            required_paths.append(paired_path)
        if not all(path.exists() for path in required_paths):
            print(f"  [SKIP] {sample} - missing files")
            continue

        # Run inference
        print(f"\n[{sample}]")
        sr_volume, sr_affine = inference_volume(
            model,
            str(lr_path),
            str(gt),
            device,
            paired_path=str(paired_path) if paired_path is not None else None,
            patch_size=patch_size,
            overlap=overlap,
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
    parser.add_argument('--data-root', type=str, default='./data/RegisteredData')
    parser.add_argument('--output-root', type=str, default='./results')
    parser.add_argument('--samples', nargs='+', default=None, help="Samples to process (default: Lumbar_26-30)")
    parser.add_argument('--sequence', type=str, default='195X_195Y_1000Z_S')
    parser.add_argument('--paired-sequence', type=str, default=None)
    parser.add_argument('--dual-channel', action='store_true')
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
    model = load_model(args.checkpoint, args.model, device, in_channels=2 if args.dual_channel else 1)

    # Run inference
    batch_inference(
        model,
        args.data_root,
        args.output_root,
        samples,
        device,
        sequence=args.sequence,
        dual_channel=args.dual_channel,
        paired_sequence=args.paired_sequence,
        patch_size=args.patch_size,
        overlap=args.overlap
    )


if __name__ == "__main__":
    main()
