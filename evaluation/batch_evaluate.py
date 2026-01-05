#!/usr/bin/env python3
"""
Batch evaluation script for registered CT images.

Evaluates all registered Clinical CT sequences against MicroCT ground truth.
Optimized for memory efficiency with large 3D volumes.
"""

import os
import sys
import time
import gc
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import nibabel as nib

from metrics import WINDOWS, apply_window, get_valid_mask
from metrics import compute_psnr, compute_mae, compute_rmse, compute_ncc, compute_nrmse


DATA_ROOT = "/data/wangping_16T/LumbarChallenge2026/RegisteredData"
OUTPUT_DIR = "/data/wangping_16T/LumbarChallenge2026/EvaluationResults"

# Only evaluate key sequences (500Z_B for both FOVs)
SEQUENCES = [
    "195X_195Y_500Z_B",
    "586X_586Y_500Z_B",
]


def compute_metrics_fast(
    gt: np.ndarray,
    pred: np.ndarray,
    window_name: str = 'bone',
    mask_threshold: float = -1000
) -> Dict[str, float]:
    """Compute metrics efficiently without SSIM (too slow for large volumes).

    Args:
        gt: Ground truth in HU
        pred: Prediction in HU
        window_name: CT window to apply
        mask_threshold: HU threshold for valid region

    Returns:
        Dict of metric values
    """
    window = WINDOWS.get(window_name)
    gt_w = apply_window(gt, window)
    pred_w = apply_window(pred, window)
    mask = get_valid_mask(gt, pred, mask_threshold)

    return {
        'psnr': compute_psnr(gt_w, pred_w, mask),
        'mae': compute_mae(gt_w, pred_w, mask),
        'rmse': compute_rmse(gt_w, pred_w, mask),
        'ncc': compute_ncc(gt_w, pred_w, mask),
        'nrmse': compute_nrmse(gt_w, pred_w, mask),
    }


def evaluate_sample(
    sample_name: str,
    data_root: str,
    sequences: List[str] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate all sequences for a single sample.

    Args:
        sample_name: Sample name (e.g., "Lumbar_05")
        data_root: Root directory of registered data
        sequences: List of sequences to evaluate (default: all)

    Returns:
        Dict of sequence_name -> window_name -> metrics
    """
    if sequences is None:
        sequences = SEQUENCES

    sample_id = sample_name.split("_")[-1]
    sample_dir = os.path.join(data_root, sample_name)

    # Load ground truth (MicroCT)
    gt_path = os.path.join(sample_dir, f"Lumbar{sample_id}_MicroPCCT_105um.nii.gz")
    if not os.path.exists(gt_path):
        print(f"  [SKIP] MicroCT not found")
        return {}

    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata().astype(np.float32)

    results = {}

    for seq in sequences:
        pred_path = os.path.join(
            sample_dir,
            f"Lumbar{sample_id}_ClinicalCT_{seq}_registered.nii.gz"
        )

        if not os.path.exists(pred_path):
            print(f"  [SKIP] {seq}: not found")
            continue

        pred_nii = nib.load(pred_path)
        pred_data = pred_nii.get_fdata().astype(np.float32)

        # Compute metrics for bone window only (most relevant)
        metrics = compute_metrics_fast(gt_data, pred_data, 'bone')
        results[seq] = {'bone': metrics}

        # Print brief summary
        print(f"  {seq}: PSNR={metrics['psnr']:.2f}dB, NCC={metrics['ncc']:.4f}, MAE={metrics['mae']:.4f}")

        # Free memory
        del pred_data
        gc.collect()

    # Free GT memory
    del gt_data
    gc.collect()

    return results


def batch_evaluate(
    samples: List[str],
    data_root: str = DATA_ROOT,
    output_dir: str = OUTPUT_DIR
) -> Tuple[Dict, Dict]:
    """Batch evaluate multiple samples.

    Args:
        samples: List of sample names
        data_root: Root directory of registered data
        output_dir: Output directory for results

    Returns:
        Tuple of (all_results, summary)
    """
    print(f"\n{'#'*60}")
    print(f"# Batch Evaluation")
    print(f"# Samples: {len(samples)}")
    print(f"{'#'*60}\n")

    all_results = {}
    t0 = time.time()

    for sample in samples:
        print(f"\n[{sample}]")
        results = evaluate_sample(sample, data_root)
        if results:
            all_results[sample] = results

    total_time = time.time() - t0

    # Aggregate by sequence
    seq_aggregates = {}
    for seq in SEQUENCES:
        seq_metrics = {'psnr': [], 'mae': [], 'rmse': [], 'ncc': [], 'nrmse': []}
        for sample, results in all_results.items():
            if seq in results:
                for metric in seq_metrics:
                    seq_metrics[metric].append(results[seq]['bone'][metric])

        if seq_metrics['psnr']:
            seq_aggregates[seq] = {
                metric: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
                for metric, values in seq_metrics.items()
            }

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Samples evaluated: {len(all_results)}/{len(samples)}")

    # Print per-sequence summary
    print(f"\n[BONE WINDOW - Per Sequence]")
    print(f"{'Sequence':<20} {'PSNR (dB)':<15} {'NCC':<15} {'MAE':<15}")
    print("-" * 65)

    for seq in SEQUENCES:
        if seq in seq_aggregates:
            agg = seq_aggregates[seq]
            print(f"{seq:<20} "
                  f"{agg['psnr']['mean']:.2f}±{agg['psnr']['std']:.2f}      "
                  f"{agg['ncc']['mean']:.4f}±{agg['ncc']['std']:.4f}  "
                  f"{agg['mae']['mean']:.4f}±{agg['mae']['std']:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    import json

    with open(f"{output_dir}/detailed_metrics.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    summary = {
        'n_samples': len(all_results),
        'samples': list(all_results.keys()),
        'sequences': seq_aggregates,
    }
    with open(f"{output_dir}/summary_metrics.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of registered CT")
    parser.add_argument('--samples', nargs='+', default=None,
                        help="Specific samples to evaluate")
    parser.add_argument('--data-root', default=DATA_ROOT,
                        help="Root directory of registered data")
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help="Output directory for results")
    args = parser.parse_args()

    # Default: evaluate the 8 problematic cases
    if args.samples is None:
        samples = [
            "Lumbar_05", "Lumbar_07", "Lumbar_09", "Lumbar_11",
            "Lumbar_18", "Lumbar_24", "Lumbar_28", "Lumbar_30"
        ]
    else:
        samples = args.samples

    batch_evaluate(samples, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
