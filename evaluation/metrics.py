#!/usr/bin/env python3
"""
Super-resolution evaluation metrics for CT images.

Computes metrics under different CT windows:
- Raw HU values
- Bone window (WC=400, WW=1800)
- Soft tissue window (WC=40, WW=400)

Metrics computed:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- NCC (Normalized Cross Correlation)
- NRMSE (Normalized RMSE)
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim_func


# CT Window presets
WINDOWS = {
    'raw': None,  # No windowing, use original HU
    'bone': {'center': 400, 'width': 1800},      # Bone window
    'soft_tissue': {'center': 40, 'width': 400}, # Soft tissue window
}

# Windows to evaluate (bone and soft tissue only)
EVAL_WINDOWS = ['bone', 'soft_tissue']


@dataclass
class MetricResult:
    """Single metric result with value and metadata."""
    value: float
    window: str
    metric_name: str


@dataclass
class CaseMetrics:
    """All metrics for a single case."""
    case_id: str
    psnr: Dict[str, float] = field(default_factory=dict)
    ssim: Dict[str, float] = field(default_factory=dict)
    mae: Dict[str, float] = field(default_factory=dict)
    rmse: Dict[str, float] = field(default_factory=dict)
    ncc: Dict[str, float] = field(default_factory=dict)
    nrmse: Dict[str, float] = field(default_factory=dict)
    lbc: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DatasetMetrics:
    """Aggregated metrics for entire dataset."""
    n_cases: int
    metrics: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # Structure: {metric_name: {window: {mean, std}}}
    lbc: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Aggregated LBC: {metric_name: {mean, std}}

    def to_dict(self) -> Dict:
        return asdict(self)


def apply_window(data: np.ndarray, window: Optional[Dict]) -> np.ndarray:
    """Apply CT window to data.

    Args:
        data: CT data in HU
        window: Dict with 'center' and 'width', or None for raw HU

    Returns:
        Windowed data normalized to [0, 1]
    """
    if window is None:
        # For raw HU, clip to typical CT range and normalize
        data_clipped = np.clip(data, -1024, 3071)
        return (data_clipped + 1024) / 4095.0

    center = window['center']
    width = window['width']
    min_val = center - width / 2
    max_val = center + width / 2

    windowed = np.clip(data, min_val, max_val)
    return (windowed - min_val) / width


def get_valid_mask(gt: np.ndarray, pred: np.ndarray,
                   threshold: float = -1000) -> np.ndarray:
    """Get mask of valid voxels (non-air regions).

    Args:
        gt: Ground truth array
        pred: Prediction array
        threshold: HU threshold for valid region

    Returns:
        Boolean mask of valid voxels
    """
    return (gt > threshold) | (pred > threshold)


def compute_psnr(gt: np.ndarray, pred: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> float:
    """Compute PSNR between ground truth and prediction.

    Args:
        gt: Ground truth (normalized to [0, 1])
        pred: Prediction (normalized to [0, 1])
        mask: Optional mask for valid region

    Returns:
        PSNR value in dB
    """
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]

    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-10:
        return 100.0  # Perfect match
    return 10 * np.log10(1.0 / mse)


def _compute_ssim_slice(args: Tuple[np.ndarray, np.ndarray, int]) -> float:
    gt_slice, pred_slice, win_size = args
    return ssim_func(gt_slice, pred_slice, data_range=1.0, win_size=win_size)


def compute_ssim_3d(gt: np.ndarray, pred: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> float:
    win_size = min(7, min(gt.shape) // 2 * 2 - 1)
    if win_size < 3:
        win_size = 3
    return ssim_func(gt, pred, data_range=1.0, win_size=win_size)


def compute_ssim(gt: np.ndarray, pred: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 n_workers: int = None,
                 sample_slices: int = None,
                 use_3d: bool = False) -> float:
    if use_3d:
        return compute_ssim_3d(gt, pred, mask)

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 64)

    min_dim = min(gt.shape[0], gt.shape[1])
    win_size = min(7, min_dim // 2 * 2 - 1)
    if win_size < 3:
        win_size = 3

    n_slices = gt.shape[2]

    if sample_slices is not None and sample_slices < n_slices:
        indices = np.linspace(0, n_slices - 1, sample_slices, dtype=int)
    else:
        indices = range(n_slices)

    slice_args = [(gt[:, :, i].copy(), pred[:, :, i].copy(), win_size)
                  for i in indices]

    if n_workers == 1:
        ssim_values = [_compute_ssim_slice(args) for args in slice_args]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            ssim_values = list(executor.map(_compute_ssim_slice, slice_args))

    return float(np.mean(ssim_values))


def compute_mae(gt: np.ndarray, pred: np.ndarray,
                mask: Optional[np.ndarray] = None) -> float:
    """Compute Mean Absolute Error."""
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]
    return np.mean(np.abs(gt - pred))


def compute_rmse(gt: np.ndarray, pred: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> float:
    """Compute Root Mean Square Error."""
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]
    return np.sqrt(np.mean((gt - pred) ** 2))


def compute_nrmse(gt: np.ndarray, pred: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> float:
    """Compute Normalized RMSE (normalized by data range)."""
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    data_range = gt.max() - gt.min()
    if data_range < 1e-10:
        return 0.0
    return rmse / data_range


def compute_ncc(gt: np.ndarray, pred: np.ndarray,
                mask: Optional[np.ndarray] = None) -> float:
    """Compute Normalized Cross Correlation."""
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]

    gt_mean = gt.mean()
    pred_mean = pred.mean()
    gt_std = gt.std()
    pred_std = pred.std()

    if gt_std < 1e-10 or pred_std < 1e-10:
        return 0.0

    return np.mean((gt - gt_mean) * (pred - pred_mean)) / (gt_std * pred_std)


# ============================================================================
# Local Bone Contrast (LBC) - Microstructure resolution metric
# ============================================================================

LBC_WINDOW_SIZE = 16        # Sliding window size (pixels)
LBC_STRIDE = 8              # Sliding window stride (50% overlap)
LBC_MIN_MASK_RATIO = 0.05   # Min fraction of mask voxels in window


def _compute_lbc_slice(
    slice_data: np.ndarray,
    bone_mask_slice: np.ndarray,
    window_size: int = LBC_WINDOW_SIZE,
    stride: int = LBC_STRIDE,
    min_mask_ratio: float = LBC_MIN_MASK_RATIO,
) -> List[float]:
    """Compute local bone contrast for a single 2D slice.

    Slides a window across the bone mask region and computes the
    percentile dynamic range (P95 - P5) within each window.

    Args:
        slice_data: 2D HU data (H, W)
        bone_mask_slice: 2D boolean mask of bone region (H, W)
        window_size: Sliding window size in pixels
        stride: Sliding window stride
        min_mask_ratio: Min fraction of mask voxels required in window

    Returns:
        List of P95-P5 dynamic range values (HU) for valid windows
    """
    h, w = slice_data.shape
    min_voxels = int(window_size * window_size * min_mask_ratio)
    contrasts = []

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            mask_patch = bone_mask_slice[y:y + window_size, x:x + window_size]
            mask_count = mask_patch.sum()
            if mask_count < max(min_voxels, 10):
                continue

            values = slice_data[y:y + window_size, x:x + window_size][mask_patch]
            p95 = np.percentile(values, 95)
            p5 = np.percentile(values, 5)
            contrasts.append(float(p95 - p5))

    return contrasts


def _compute_lbc_slice_args(args: Tuple) -> List[float]:
    """Wrapper for multiprocessing."""
    slice_data, bone_mask_slice, window_size, stride, min_mask_ratio = args
    return _compute_lbc_slice(slice_data, bone_mask_slice,
                              window_size, stride, min_mask_ratio)


def compute_lbc(
    data: np.ndarray,
    bone_mask: np.ndarray,
    window_size: int = LBC_WINDOW_SIZE,
    stride: int = LBC_STRIDE,
    n_workers: int = None,
    sample_ratio: float = None,
) -> Dict[str, float]:
    """Compute Local Bone Contrast for a 3D volume.

    Measures the percentile dynamic range (P95 - P5) within sliding
    windows across the bone mask region. Higher values indicate better
    microstructure visibility.

    Args:
        data: 3D CT data in HU (H, W, D)
        bone_mask: 3D boolean mask of bone region (H, W, D)
        window_size: Sliding window size in pixels
        stride: Sliding window stride
        n_workers: Number of parallel workers
        sample_ratio: If set, sample this fraction of slices

    Returns:
        Dict with lbc_mean, lbc_std, lbc_db, n_windows
    """
    n_slices = data.shape[2]

    if sample_ratio and 0 < sample_ratio < 1.0:
        n_sample = max(int(n_slices * sample_ratio), 10)
        indices = np.linspace(0, n_slices - 1, n_sample, dtype=int)
    else:
        indices = range(n_slices)

    valid_indices = [z for z in indices if bone_mask[:, :, z].any()]

    if not valid_indices:
        return {'lbc_mean': 0.0, 'lbc_std': 0.0, 'lbc_db': -np.inf,
                'n_windows': 0}

    if n_workers and n_workers > 1:
        args_list = [
            (data[:, :, z].copy(), bone_mask[:, :, z].copy(),
             window_size, stride, LBC_MIN_MASK_RATIO)
            for z in valid_indices
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            slice_results = list(executor.map(
                _compute_lbc_slice_args, args_list))
        all_contrasts = []
        for r in slice_results:
            all_contrasts.extend(r)
    else:
        all_contrasts = []
        for z in valid_indices:
            contrasts = _compute_lbc_slice(
                data[:, :, z], bone_mask[:, :, z],
                window_size=window_size, stride=stride,
            )
            all_contrasts.extend(contrasts)

    if not all_contrasts:
        return {'lbc_mean': 0.0, 'lbc_std': 0.0, 'lbc_db': -np.inf,
                'n_windows': 0}

    arr = np.array(all_contrasts)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    lbc_db = float(20 * np.log10(max(mean_val, 1e-10)))

    return {
        'lbc_mean': mean_val,
        'lbc_std': std_val,
        'lbc_db': lbc_db,
        'n_windows': len(all_contrasts),
    }


def compute_lbc_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    bone_mask: np.ndarray,
    window_size: int = LBC_WINDOW_SIZE,
    stride: int = LBC_STRIDE,
    n_workers: int = None,
    sample_ratio: float = None,
    gt_lbc_cache: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute LBC for both GT and prediction, return comparison metrics.

    Args:
        gt: Ground truth 3D CT data in HU
        pred: Prediction 3D CT data in HU
        bone_mask: 3D boolean mask of bone region
        window_size: Sliding window size
        stride: Sliding window stride
        n_workers: Number of parallel workers
        sample_ratio: Fraction of slices to sample
        gt_lbc_cache: Pre-computed GT LBC to avoid redundant computation

    Returns:
        Dict with lbc_gt_mean, lbc_pred_mean, lbc_ratio, lbc_diff_db, etc.
    """
    gt_lbc = gt_lbc_cache if gt_lbc_cache else compute_lbc(
        gt, bone_mask, window_size, stride, n_workers, sample_ratio)
    pred_lbc = compute_lbc(
        pred, bone_mask, window_size, stride, n_workers, sample_ratio)

    gt_mean = gt_lbc['lbc_mean']
    pred_mean = pred_lbc['lbc_mean']

    if gt_mean > 1e-10:
        ratio = pred_mean / gt_mean
    else:
        ratio = 0.0

    return {
        'lbc_gt_mean': gt_lbc['lbc_mean'],
        'lbc_gt_std': gt_lbc['lbc_std'],
        'lbc_gt_db': gt_lbc['lbc_db'],
        'lbc_gt_n_windows': gt_lbc['n_windows'],
        'lbc_pred_mean': pred_lbc['lbc_mean'],
        'lbc_pred_std': pred_lbc['lbc_std'],
        'lbc_pred_db': pred_lbc['lbc_db'],
        'lbc_pred_n_windows': pred_lbc['n_windows'],
        'lbc_ratio': float(ratio),
        'lbc_diff': float(abs(pred_mean - gt_mean)),
        'lbc_diff_db': float(pred_lbc['lbc_db'] - gt_lbc['lbc_db']),
    }


class MetricsCalculator:
    """Calculator for super-resolution evaluation metrics."""

    def __init__(self, use_mask: bool = True, mask_threshold: float = -1000):
        """Initialize calculator.

        Args:
            use_mask: Whether to use valid region mask
            mask_threshold: HU threshold for valid region
        """
        self.use_mask = use_mask
        self.mask_threshold = mask_threshold

    def compute_all_metrics(self, gt: np.ndarray, pred: np.ndarray,
                            window_name: str = 'raw') -> Dict[str, float]:
        """Compute all metrics for a single window.

        Args:
            gt: Ground truth in HU
            pred: Prediction in HU
            window_name: Name of CT window to apply

        Returns:
            Dict of metric_name -> value
        """
        window = WINDOWS.get(window_name)
        gt_w = apply_window(gt, window)
        pred_w = apply_window(pred, window)

        mask = None
        if self.use_mask:
            mask = get_valid_mask(gt, pred, self.mask_threshold)

        return {
            'psnr': compute_psnr(gt_w, pred_w, mask),
            'ssim': compute_ssim(gt_w, pred_w, mask),
            'mae': compute_mae(gt_w, pred_w, mask),
            'rmse': compute_rmse(gt_w, pred_w, mask),
            'ncc': compute_ncc(gt_w, pred_w, mask),
            'nrmse': compute_nrmse(gt_w, pred_w, mask),
        }

    def evaluate_case(self, gt: np.ndarray, pred: np.ndarray,
                      case_id: str,
                      compute_lbc_flag: bool = True) -> CaseMetrics:
        """Evaluate a single case across bone and soft tissue windows.

        Args:
            gt: Ground truth in HU
            pred: Prediction in HU
            case_id: Case identifier
            compute_lbc_flag: Whether to compute LBC metrics

        Returns:
            CaseMetrics with all metrics
        """
        result = CaseMetrics(case_id=case_id)

        for window_name in EVAL_WINDOWS:
            metrics = self.compute_all_metrics(gt, pred, window_name)
            result.psnr[window_name] = metrics['psnr']
            result.ssim[window_name] = metrics['ssim']
            result.mae[window_name] = metrics['mae']
            result.rmse[window_name] = metrics['rmse']
            result.ncc[window_name] = metrics['ncc']
            result.nrmse[window_name] = metrics['nrmse']

        if compute_lbc_flag:
            bone_mask = get_valid_mask(gt, pred, threshold=-500)
            result.lbc = compute_lbc_comparison(gt, pred, bone_mask)

        return result

    @staticmethod
    def aggregate_results(case_results: List[CaseMetrics]) -> DatasetMetrics:
        """Aggregate case results into dataset-level statistics.

        Args:
            case_results: List of CaseMetrics

        Returns:
            DatasetMetrics with mean and std for each metric/window
        """
        n_cases = len(case_results)
        metric_names = ['psnr', 'ssim', 'mae', 'rmse', 'ncc', 'nrmse']

        dataset = DatasetMetrics(n_cases=n_cases)

        for metric_name in metric_names:
            dataset.metrics[metric_name] = {}
            for window_name in EVAL_WINDOWS:
                values = [getattr(c, metric_name)[window_name]
                          for c in case_results]
                dataset.metrics[metric_name][window_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }

        # Aggregate LBC metrics
        lbc_keys = [
            'lbc_gt_mean', 'lbc_gt_std', 'lbc_gt_db',
            'lbc_pred_mean', 'lbc_pred_std', 'lbc_pred_db',
            'lbc_ratio', 'lbc_diff', 'lbc_diff_db',
        ]
        cases_with_lbc = [c for c in case_results if c.lbc]
        if cases_with_lbc:
            for key in lbc_keys:
                values = [c.lbc[key] for c in cases_with_lbc
                          if key in c.lbc and np.isfinite(c.lbc[key])]
                if values:
                    dataset.lbc[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                    }

        return dataset


def format_case_report(case: CaseMetrics) -> str:
    """Format single case metrics as text report."""
    lines = [f"\n{'='*60}", f"Case: {case.case_id}", '='*60]

    for window in EVAL_WINDOWS:
        window_label = "BONE" if window == "bone" else "SOFT TISSUE"
        lines.append(f"\n[{window_label}]")
        lines.append(f"  PSNR:  {case.psnr[window]:.2f} dB")
        lines.append(f"  SSIM:  {case.ssim[window]:.4f}")
        lines.append(f"  MAE:   {case.mae[window]:.4f}")
        lines.append(f"  RMSE:  {case.rmse[window]:.4f}")
        lines.append(f"  NCC:   {case.ncc[window]:.4f}")
        lines.append(f"  NRMSE: {case.nrmse[window]:.4f}")

    if case.lbc:
        lbc = case.lbc
        lines.append(f"\n[LOCAL BONE CONTRAST]")
        lines.append(f"  GT:   LBC={lbc.get('lbc_gt_mean', 0):.1f} HU "
                     f"({lbc.get('lbc_gt_db', 0):.2f} dB)")
        lines.append(f"  Pred: LBC={lbc.get('lbc_pred_mean', 0):.1f} HU "
                     f"({lbc.get('lbc_pred_db', 0):.2f} dB)")
        lines.append(f"  Ratio={lbc.get('lbc_ratio', 0):.4f}  "
                     f"Diff_dB={lbc.get('lbc_diff_db', 0):.2f}")

    return '\n'.join(lines)


def format_dataset_report(dataset: DatasetMetrics) -> str:
    """Format dataset-level metrics as text report."""
    lines = [
        f"\n{'#'*60}",
        f"# DATASET SUMMARY (n={dataset.n_cases})",
        '#'*60
    ]

    metric_names = ['psnr', 'ssim', 'mae', 'rmse', 'ncc', 'nrmse']
    units = {'psnr': 'dB', 'ssim': '', 'mae': '', 'rmse': '', 'ncc': '', 'nrmse': ''}

    for window in EVAL_WINDOWS:
        window_label = "BONE" if window == "bone" else "SOFT TISSUE"
        lines.append(f"\n[{window_label}]")
        for metric in metric_names:
            stats = dataset.metrics[metric][window]
            unit = units[metric]
            lines.append(
                f"  {metric.upper():6s}: {stats['mean']:.4f} +/- {stats['std']:.4f} {unit}"
            )

    if dataset.lbc:
        lines.append(f"\n[LOCAL BONE CONTRAST]")
        lbc = dataset.lbc
        for key, label in [
            ('lbc_gt_mean', 'GT LBC (HU)'),
            ('lbc_pred_mean', 'Pred LBC (HU)'),
            ('lbc_ratio', 'LBC Ratio'),
            ('lbc_diff_db', 'LBC Diff (dB)'),
        ]:
            if key in lbc:
                stats = lbc[key]
                lines.append(f"  {label:<16}: "
                             f"{stats['mean']:.4f} +/- {stats['std']:.4f}")

    return '\n'.join(lines)


def save_results(case_results: List[CaseMetrics],
                 dataset_metrics: DatasetMetrics,
                 output_dir: str):
    """Save evaluation results to files.

    Args:
        case_results: List of per-case metrics
        dataset_metrics: Aggregated dataset metrics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save per-case results as JSON
    cases_data = [c.to_dict() for c in case_results]
    with open(f"{output_dir}/case_metrics.json", 'w') as f:
        json.dump(cases_data, f, indent=2)

    # Save dataset summary as JSON
    with open(f"{output_dir}/dataset_metrics.json", 'w') as f:
        json.dump(dataset_metrics.to_dict(), f, indent=2)

    # Save text report
    with open(f"{output_dir}/report.txt", 'w') as f:
        for case in case_results:
            f.write(format_case_report(case))
        f.write(format_dataset_report(dataset_metrics))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics")
    parser.add_argument('--data-root', type=str, default='./data/registered_nifti',
                        help='Root directory with registered data')
    parser.add_argument('--sample', type=str, default='Lumbar_01',
                        help='Sample name to evaluate')
    parser.add_argument('--pred-seq', type=str, default='195X_195Y_500Z_B_registered',
                        help='Prediction sequence name')
    args = parser.parse_args()

    calculator = MetricsCalculator(use_mask=True)

    # Demo: evaluate single case
    sample = args.sample
    sample_id = sample.split("_")[-1]

    gt_path = f"{args.data_root}/{sample}/Lumbar{sample_id}_MicroPCCT_105um.nii.gz"
    pred_path = f"{args.data_root}/{sample}/Lumbar{sample_id}_ClinicalCT_{args.pred_seq}.nii.gz"

    print(f"Loading {sample}...")
    gt = nib.load(gt_path).get_fdata()
    pred = nib.load(pred_path).get_fdata()

    print("Computing metrics...")
    case_metrics = calculator.evaluate_case(gt, pred, sample)
    print(format_case_report(case_metrics))
