#!/usr/bin/env python3
"""Evaluate public SR predictions on the released test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from metrics import WINDOWS, apply_window, compute_lbc_comparison, compute_mae, compute_psnr, compute_ssim, get_valid_mask


DEFAULT_GT_ROOT = Path("./data/RegisteredData")
DEFAULT_PRED_ROOT = Path("./results")
DEFAULT_BONE_MASK_ROOT = Path("./data/BoneMask")
DEFAULT_OUTPUT_DIR = Path("./outputs/evaluation")
DEFAULT_METHODS = ["srcnn", "unet"]
DEFAULT_SAMPLES = [f"Lumbar_{i:02d}" for i in range(26, 31)]
DEFAULT_SEQUENCES = ["195X_195Y_1000Z_S", "586X_586Y_1000Z_S"]
EVAL_WINDOWS = ["raw", "bone", "soft_tissue"]
EVAL_MODES = ["full", "non_air", "bone_mask"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate public SR predictions")
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    parser.add_argument('--samples', nargs='+', default=DEFAULT_SAMPLES)
    parser.add_argument('--sequences', nargs='+', default=DEFAULT_SEQUENCES)
    parser.add_argument('--gt-root', type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument('--pred-root', type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument('--bone-mask-root', type=Path, default=DEFAULT_BONE_MASK_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--fast-ssim', action='store_true')
    parser.add_argument('--ssim-workers', type=int, default=8)
    parser.add_argument('--lbc-workers', type=int, default=8)
    return parser.parse_args()


def compute_ssim_with_roi(gt_w, pred_w, mask, n_workers, sample_slices):
    if mask is None:
        return float(compute_ssim(gt_w, pred_w, None, n_workers=n_workers, sample_slices=sample_slices))
    if not np.any(mask):
        return float("nan")
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))
    gt_crop = gt_w[slices].copy()
    pred_crop = pred_w[slices].copy()
    mask_crop = mask[slices]
    gt_crop[~mask_crop] = 0.0
    pred_crop[~mask_crop] = 0.0
    return float(compute_ssim(gt_crop, pred_crop, None, n_workers=n_workers, sample_slices=sample_slices))


def build_pred_path(pred_root: Path, method: str, sample: str, sequence: str) -> Path:
    sample_id = sample.split("_")[-1]
    return pred_root / method / sample / f"Lumbar{sample_id}_{sequence}_{method}.nii.gz"


def build_bone_mask_path(bone_mask_root: Path, sample: str, sequence: str) -> Path:
    sample_id = sample.split("_")[-1]
    return bone_mask_root / sample / f"Lumbar{sample_id}_ClinicalCT_{sequence}_registered_BoneMask.nii.gz"


def evaluate_pair(gt_data, pred_data, bone_mask, fast_ssim, ssim_workers):
    non_air_mask = get_valid_mask(gt_data, pred_data, threshold=-1000)
    sample_slices = int(gt_data.shape[2] * 0.125) if fast_ssim else None
    masks = {
        "full": None,
        "non_air": non_air_mask,
        "bone_mask": bone_mask,
    }
    results = {}
    for mode in EVAL_MODES:
        results[mode] = {}
        mask = masks[mode]
        for window_name in EVAL_WINDOWS:
            window = WINDOWS[window_name]
            gt_w = apply_window(gt_data, window)
            pred_w = apply_window(pred_data, window)
            results[mode][window_name] = {
                "psnr": float(compute_psnr(gt_w, pred_w, mask)),
                "ssim": compute_ssim_with_roi(gt_w, pred_w, mask, ssim_workers, sample_slices),
                "mae": float(compute_mae(gt_w, pred_w, mask)),
            }
    return results


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        key = (row['method'], row['sequence'], row['mode'], row['window'])
        grouped.setdefault(key, {'psnr': [], 'ssim': [], 'mae': []})
        for metric in ('psnr', 'ssim', 'mae'):
            grouped[key][metric].append(float(row[metric]))

    summary_rows = []
    for (method, sequence, mode, window), metrics in sorted(grouped.items()):
        summary_rows.append({
            'method': method,
            'sequence': sequence,
            'mode': mode,
            'window': window,
            'psnr_mean': float(np.mean(metrics['psnr'])),
            'psnr_std': float(np.std(metrics['psnr'])),
            'ssim_mean': float(np.mean(metrics['ssim'])),
            'ssim_std': float(np.std(metrics['ssim'])),
            'mae_mean': float(np.mean(metrics['mae'])),
            'mae_std': float(np.std(metrics['mae'])),
            'n_cases': len(metrics['psnr']),
        })
    return summary_rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detailed_rows = []
    lbc_rows = []
    nested_results = {}

    for method in args.methods:
        nested_results[method] = {}
        for sample in tqdm(args.samples, desc=f"eval:{method}"):
            sample_id = sample.split("_")[-1]
            gt_path = args.gt_root / sample / f"Lumbar{sample_id}_MicroPCCT_105um.nii.gz"
            if not gt_path.exists():
                continue

            gt_data = nib.load(str(gt_path)).get_fdata().astype(np.float32)
            sample_results = {}

            for sequence in args.sequences:
                bone_mask_path = build_bone_mask_path(args.bone_mask_root, sample, sequence)
                pred_path = build_pred_path(args.pred_root, method, sample, sequence)
                if not bone_mask_path.exists() or not pred_path.exists():
                    continue

                pred_data = nib.load(str(pred_path)).get_fdata().astype(np.float32)
                bone_mask = nib.load(str(bone_mask_path)).get_fdata() > 0
                metrics = evaluate_pair(gt_data, pred_data, bone_mask, args.fast_ssim, args.ssim_workers)
                lbc = compute_lbc_comparison(
                    gt_data,
                    pred_data,
                    bone_mask,
                    n_workers=args.lbc_workers if args.lbc_workers > 1 else None,
                )
                metrics['lbc'] = lbc
                sample_results[sequence] = metrics

                for mode in EVAL_MODES:
                    for window_name in EVAL_WINDOWS:
                        values = metrics[mode][window_name]
                        detailed_rows.append({
                            'method': method,
                            'sample': sample,
                            'sequence': sequence,
                            'mode': mode,
                            'window': window_name,
                            'psnr': values['psnr'],
                            'ssim': values['ssim'],
                            'mae': values['mae'],
                        })

                lbc_rows.append({
                    'method': method,
                    'sample': sample,
                    'sequence': sequence,
                    **lbc,
                })

            if sample_results:
                nested_results[method][sample] = sample_results

    detailed_path = args.output_dir / 'detailed_metrics.csv'
    with detailed_path.open('w', newline='') as f:
        if detailed_rows:
            writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detailed_rows)

    summary_rows = aggregate_rows(detailed_rows)
    summary_path = args.output_dir / 'summary_metrics.csv'
    with summary_path.open('w', newline='') as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    lbc_path = args.output_dir / 'lbc_metrics.csv'
    with lbc_path.open('w', newline='') as f:
        if lbc_rows:
            writer = csv.DictWriter(f, fieldnames=list(lbc_rows[0].keys()))
            writer.writeheader()
            writer.writerows(lbc_rows)

    with (args.output_dir / 'evaluation_results.json').open('w', encoding='utf-8') as f:
        json.dump(nested_results, f, indent=2)

    print(f"Detailed metrics: {detailed_path}")
    print(f"Summary metrics:  {summary_path}")
    print(f"LBC metrics:      {lbc_path}")


if __name__ == "__main__":
    main()
