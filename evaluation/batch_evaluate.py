#!/usr/bin/env python3
import os
import sys
import time
import gc
import json
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import nibabel as nib

from metrics import WINDOWS, EVAL_WINDOWS, apply_window, get_valid_mask
from metrics import compute_psnr, compute_ssim, compute_mae, compute_rmse, compute_ncc, compute_nrmse
from metrics import compute_lbc_comparison


DATA_ROOT = "./data/registered_nifti"
OUTPUT_DIR = "./outputs/evaluation"

SEQUENCES = [
    "195X_195Y_500Z_B",
    "195X_195Y_500Z_S",
    "195X_195Y_1000Z_B",
    "195X_195Y_1000Z_S",
    "586X_586Y_500Z_B",
    "586X_586Y_500Z_S",
    "586X_586Y_1000Z_B",
    "586X_586Y_1000Z_S",
]


def compute_metrics_fast(
    gt: np.ndarray,
    pred: np.ndarray,
    window_name: str = 'bone',
    mask_threshold: float = -1000
) -> Dict[str, float]:
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


def compute_metrics_full(
    gt: np.ndarray,
    pred: np.ndarray,
    window_name: str = 'bone',
    mask_threshold: float = -1000,
    ssim_workers: int = None
) -> Dict[str, float]:
    window = WINDOWS.get(window_name)
    gt_w = apply_window(gt, window)
    pred_w = apply_window(pred, window)
    mask = get_valid_mask(gt, pred, mask_threshold)

    return {
        'psnr': compute_psnr(gt_w, pred_w, mask),
        'ssim': compute_ssim(gt_w, pred_w, mask, n_workers=ssim_workers),
        'mae': compute_mae(gt_w, pred_w, mask),
        'rmse': compute_rmse(gt_w, pred_w, mask),
        'ncc': compute_ncc(gt_w, pred_w, mask),
        'nrmse': compute_nrmse(gt_w, pred_w, mask),
    }


def evaluate_sample(
    sample_name: str,
    data_root: str,
    sequences: List[str] = None,
    compute_ssim: bool = True,
    ssim_workers: int = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if sequences is None:
        sequences = SEQUENCES

    sample_id = sample_name.split("_")[-1]
    sample_dir = os.path.join(data_root, sample_name)

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

        seq_results = {}
        for window_name in EVAL_WINDOWS:
            if compute_ssim:
                metrics = compute_metrics_full(gt_data, pred_data, window_name, ssim_workers=ssim_workers)
            else:
                metrics = compute_metrics_fast(gt_data, pred_data, window_name)
            seq_results[window_name] = metrics

        # Compute LBC on raw HU
        bone_mask = get_valid_mask(gt_data, pred_data, threshold=-500)
        seq_results['lbc'] = compute_lbc_comparison(gt_data, pred_data, bone_mask)

        results[seq] = seq_results

        m = seq_results['bone']
        ssim_str = f", SSIM={m.get('ssim', 0):.4f}" if compute_ssim else ""
        lbc_ratio = seq_results['lbc'].get('lbc_ratio', 0)
        print(f"  {seq}: PSNR={m['psnr']:.2f}dB{ssim_str}, NCC={m['ncc']:.4f}, LBC_ratio={lbc_ratio:.4f}")

        del pred_data
        gc.collect()

    del gt_data
    gc.collect()

    return results


def batch_evaluate(
    samples: List[str],
    data_root: str = DATA_ROOT,
    output_dir: str = OUTPUT_DIR,
    compute_ssim_flag: bool = True,
    ssim_workers: int = None
) -> Tuple[Dict, Dict]:
    print(f"\n{'#'*60}")
    print(f"# Batch Evaluation")
    print(f"# Samples: {len(samples)}")
    print(f"# Compute SSIM: {compute_ssim_flag}")
    print(f"{'#'*60}\n")

    all_results = {}
    t0 = time.time()

    for sample in samples:
        print(f"\n[{sample}]")
        results = evaluate_sample(
            sample, data_root, SEQUENCES,
            compute_ssim=compute_ssim_flag,
            ssim_workers=ssim_workers
        )
        if results:
            all_results[sample] = results

    total_time = time.time() - t0

    # Aggregate by sequence and window
    seq_aggregates = {}
    metric_names = ['psnr', 'ssim', 'mae', 'rmse', 'ncc', 'nrmse'] if compute_ssim_flag else ['psnr', 'mae', 'rmse', 'ncc', 'nrmse']

    for window_name in EVAL_WINDOWS:
        seq_aggregates[window_name] = {}
        for seq in SEQUENCES:
            seq_metrics = {m: [] for m in metric_names}
            for sample, results in all_results.items():
                if seq in results and window_name in results[seq]:
                    for metric in metric_names:
                        if metric in results[seq][window_name]:
                            seq_metrics[metric].append(results[seq][window_name][metric])

            if seq_metrics['psnr']:
                seq_aggregates[window_name][seq] = {
                    metric: {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
                    for metric, values in seq_metrics.items() if values
                }

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Samples evaluated: {len(all_results)}/{len(samples)}")

    for window_name in EVAL_WINDOWS:
        print(f"\n[{window_name.upper()}]")
        header = f"{'Sequence':<20} {'PSNR (dB)':<15} {'NCC':<15}"
        if compute_ssim_flag:
            header += f" {'SSIM':<15}"
        print(header)
        print("-" * 65)

        for seq in SEQUENCES:
            if seq in seq_aggregates[window_name]:
                agg = seq_aggregates[window_name][seq]
                line = f"{seq:<20} "
                line += f"{agg['psnr']['mean']:.2f}±{agg['psnr']['std']:.2f}      "
                line += f"{agg['ncc']['mean']:.4f}±{agg['ncc']['std']:.4f}  "
                if compute_ssim_flag and 'ssim' in agg:
                    line += f"{agg['ssim']['mean']:.4f}±{agg['ssim']['std']:.4f}"
                print(line)

    # Aggregate LBC by sequence
    lbc_aggregates = {}
    lbc_keys = ['lbc_gt_mean', 'lbc_gt_db', 'lbc_pred_mean', 'lbc_pred_db',
                'lbc_ratio', 'lbc_diff', 'lbc_diff_db']
    for seq in SEQUENCES:
        lbc_lists = {k: [] for k in lbc_keys}
        for sample, results in all_results.items():
            if seq in results and 'lbc' in results[seq]:
                lbc = results[seq]['lbc']
                for k in lbc_keys:
                    val = lbc.get(k)
                    if val is not None and np.isfinite(val):
                        lbc_lists[k].append(val)
        if lbc_lists['lbc_gt_mean']:
            lbc_aggregates[seq] = {
                k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                for k, v in lbc_lists.items() if v
            }

    # Print LBC summary
    print(f"\n[LOCAL BONE CONTRAST]")
    print(f"{'Sequence':<20} {'GT (HU)':<15} {'Pred (HU)':<15} {'Ratio':<15} {'Diff_dB':<15}")
    print("-" * 80)
    for seq in SEQUENCES:
        if seq in lbc_aggregates:
            a = lbc_aggregates[seq]
            gt_m = a.get('lbc_gt_mean', {}).get('mean', 0)
            pred_m = a.get('lbc_pred_mean', {}).get('mean', 0)
            ratio = a.get('lbc_ratio', {}).get('mean', 0)
            diff_db = a.get('lbc_diff_db', {}).get('mean', 0)
            print(f"{seq:<20} {gt_m:<15.1f} {pred_m:<15.1f} {ratio:<15.4f} {diff_db:<15.2f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    import json

    with open(f"{output_dir}/detailed_metrics.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    summary = {
        'n_samples': len(all_results),
        'samples': list(all_results.keys()),
        'sequences': seq_aggregates,
        'lbc': lbc_aggregates,
    }
    with open(f"{output_dir}/summary_metrics.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of registered CT")
    parser.add_argument('--samples', nargs='+', default=None,
                        help="Specific samples to evaluate (default: all 30)")
    parser.add_argument('--data-root', default=DATA_ROOT,
                        help="Root directory of registered data")
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument('--fast', action='store_true',
                        help="Fast mode: skip SSIM computation")
    parser.add_argument('--ssim-workers', type=int, default=None,
                        help="Number of workers for SSIM (default: auto)")
    args = parser.parse_args()

    if args.samples is None:
        samples = [f"Lumbar_{i:02d}" for i in range(1, 31)]
    else:
        samples = args.samples

    batch_evaluate(
        samples, args.data_root, args.output_dir,
        compute_ssim_flag=not args.fast,
        ssim_workers=args.ssim_workers
    )


if __name__ == "__main__":
    main()
