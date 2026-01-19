#!/usr/bin/env python3
"""
Traditional interpolation baselines for CT super-resolution.

Implements four classical interpolation methods:
- Nearest neighbor (order=0)
- Trilinear (order=1)
- Bicubic (order=3)
- Lanczos-like (order=5)
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
from typing import Tuple, Dict


def get_scale_factor(lr_shape: Tuple[int, ...], hr_shape: Tuple[int, ...]) -> Tuple[float, ...]:
    """Calculate scale factor between LR and HR volumes."""
    return tuple(h / l for l, h in zip(lr_shape, hr_shape))


def interpolate_nearest(lr_data: np.ndarray, hr_shape: Tuple[int, ...]) -> np.ndarray:
    """Nearest neighbor interpolation (order=0)."""
    scale = get_scale_factor(lr_data.shape, hr_shape)
    return ndimage.zoom(lr_data, scale, order=0)


def interpolate_trilinear(lr_data: np.ndarray, hr_shape: Tuple[int, ...]) -> np.ndarray:
    """Trilinear interpolation (order=1)."""
    scale = get_scale_factor(lr_data.shape, hr_shape)
    return ndimage.zoom(lr_data, scale, order=1)


def interpolate_bicubic(lr_data: np.ndarray, hr_shape: Tuple[int, ...]) -> np.ndarray:
    """Bicubic interpolation (order=3)."""
    scale = get_scale_factor(lr_data.shape, hr_shape)
    return ndimage.zoom(lr_data, scale, order=3)


def interpolate_lanczos(lr_data: np.ndarray, hr_shape: Tuple[int, ...]) -> np.ndarray:
    """Lanczos-like interpolation using high-order spline (order=5)."""
    scale = get_scale_factor(lr_data.shape, hr_shape)
    return ndimage.zoom(lr_data, scale, order=5)


INTERPOLATION_METHODS: Dict[str, callable] = {
    'nearest': interpolate_nearest,
    'trilinear': interpolate_trilinear,
    'bicubic': interpolate_bicubic,
    'lanczos': interpolate_lanczos,
}


def upsample_volume(lr_data: np.ndarray, hr_shape: Tuple[int, ...], method: str = 'bicubic') -> np.ndarray:
    """Upsample volume using specified interpolation method.

    Args:
        lr_data: Low-resolution input volume
        hr_shape: Target high-resolution shape
        method: Interpolation method name

    Returns:
        Upsampled volume
    """
    if method not in INTERPOLATION_METHODS:
        raise ValueError(f"Unknown method: {method}. Choose from {list(INTERPOLATION_METHODS.keys())}")

    return INTERPOLATION_METHODS[method](lr_data, hr_shape)


def process_single_case(
    lr_path: str,
    gt_path: str,
    output_path: str,
    method: str = 'bicubic'
) -> np.ndarray:
    """Process a single case with interpolation.

    Args:
        lr_path: Path to low-resolution input
        gt_path: Path to ground truth (for target shape)
        output_path: Path to save upsampled result
        method: Interpolation method

    Returns:
        Upsampled data
    """
    # Load data
    lr_nii = nib.load(lr_path)
    gt_nii = nib.load(gt_path)

    lr_data = lr_nii.get_fdata().astype(np.float32)
    gt_shape = gt_nii.shape

    print(f"  LR shape: {lr_data.shape} -> HR shape: {gt_shape}")

    # Upsample
    hr_pred = upsample_volume(lr_data, gt_shape, method)

    # Save with GT affine
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hr_nii = nib.Nifti1Image(hr_pred.astype(np.int16), gt_nii.affine)
    nib.save(hr_nii, output_path)

    print(f"  Saved: {output_path}")
    return hr_pred


def batch_interpolate(
    data_root: str,
    output_root: str,
    method: str = 'bicubic',
    samples: list = None,
    sequences: list = None
):
    """Batch process all samples with interpolation.

    Args:
        data_root: Root directory of registered data
        output_root: Output directory for upsampled results
        method: Interpolation method
        samples: List of sample names
        sequences: List of sequences to process
    """
    if samples is None:
        samples = [f"Lumbar_{i:02d}" for i in range(1, 31)]

    if sequences is None:
        # Challenge sequences: thick slice + soft kernel
        sequences = ["195X_195Y_1000Z_S", "586X_586Y_1000Z_S"]

    print(f"\n{'#'*60}")
    print(f"# Interpolation Baseline: {method.upper()}")
    print(f"# Samples: {len(samples)}")
    print(f"# Sequences: {sequences}")
    print(f"{'#'*60}\n")

    for sample in samples:
        print(f"\n[{sample}]")
        sample_id = sample.split("_")[-1]
        sample_dir = os.path.join(data_root, sample)

        gt_path = os.path.join(sample_dir, f"Lumbar{sample_id}_MicroPCCT_105um.nii.gz")

        if not os.path.exists(gt_path):
            print(f"  [SKIP] GT not found: {gt_path}")
            continue

        for seq in sequences:
            lr_path = os.path.join(
                sample_dir,
                f"Lumbar{sample_id}_ClinicalCT_{seq}_registered.nii.gz"
            )

            if not os.path.exists(lr_path):
                print(f"  [SKIP] LR not found: {lr_path}")
                continue

            output_path = os.path.join(
                output_root, method, sample,
                f"Lumbar{sample_id}_{seq}_{method}.nii.gz"
            )

            process_single_case(lr_path, gt_path, output_path, method)

    print(f"\n{'#'*60}")
    print(f"# Complete! Results saved to: {output_root}")
    print(f"{'#'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Traditional interpolation baselines for super-resolution"
    )
    parser.add_argument(
        '--method', type=str, default='bicubic',
        choices=list(INTERPOLATION_METHODS.keys()),
        help="Interpolation method (default: bicubic)"
    )
    parser.add_argument(
        '--data-root', type=str,
        default="/data/LumbarSR/registered_nifti",
        help="Root directory of registered NIfTI data"
    )
    parser.add_argument(
        '--output-root', type=str,
        default="/data/LumbarSR/results",
        help="Output directory"
    )
    parser.add_argument(
        '--samples', nargs='+', default=None,
        help="Specific samples (default: all 30)"
    )
    args = parser.parse_args()

    batch_interpolate(
        data_root=args.data_root,
        output_root=args.output_root,
        method=args.method,
        samples=args.samples
    )


if __name__ == "__main__":
    main()
