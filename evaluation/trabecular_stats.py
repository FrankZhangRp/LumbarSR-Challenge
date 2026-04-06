#!/usr/bin/env python3
"""Summarize trabecular morphometry differences and exact Wilcoxon p-values."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path


METHOD_ORDER = [
    "baseline_registered",
    "nearest",
    "trilinear",
    "bicubic",
    "lanczos",
    "srcnn_npz",
    "unet_npz",
    "esrgan_npz_gan",
    "swinir_npz",
]

METHOD_LABELS = {
    "baseline_registered": "Registered clinical CT baseline",
    "nearest": "Nearest",
    "trilinear": "Trilinear",
    "bicubic": "Bicubic",
    "lanczos": "Lanczos",
    "srcnn_npz": "SRCNN",
    "unet_npz": "UNet",
    "esrgan_npz_gan": "ESRGAN",
    "swinir_npz": "SwinIR",
}

METRICS = [
    ("BV/TV", "pred_BV_TV", "gt_BV_TV"),
    ("Tb.Th", "pred_Tb_Th_mm", "gt_Tb_Th_mm"),
    ("Tb.Sp", "pred_Tb_Sp_mm", "gt_Tb_Sp_mm"),
    ("Tb.N", "pred_Tb_N_per_mm", "gt_Tb_N_per_mm"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detail-csv",
        nargs="+",
        type=Path,
        required=True,
        help="One or more trabecular_metrics_detail.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save summary CSV files.",
    )
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    deduped: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                deduped[(row["method"], row["sample"], row["sequence"])] = row
    return list(deduped.values())


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    rank = 1
    while start < len(order):
        end = start
        while end + 1 < len(order) and abs(values[order[end + 1]] - values[order[start]]) < 1e-12:
            end += 1
        avg_rank = (rank + rank + (end - start)) / 2.0
        for pos in range(start, end + 1):
            ranks[order[pos]] = avg_rank
        rank += end - start + 1
        start = end + 1
    return ranks


def exact_wilcoxon_two_sided(differences: list[float]) -> float:
    non_zero = [diff for diff in differences if abs(diff) > 1e-12]
    if not non_zero:
        return 1.0
    abs_diffs = [abs(diff) for diff in non_zero]
    ranks = average_ranks(abs_diffs)
    observed_w_pos = sum(rank for diff, rank in zip(non_zero, ranks) if diff > 0)
    total_rank = sum(ranks)
    observed_w_min = min(observed_w_pos, total_rank - observed_w_pos)

    outcomes = 0
    extreme = 0
    for signs in itertools.product([0, 1], repeat=len(non_zero)):
        w_pos = sum(rank for sign, rank in zip(signs, ranks) if sign)
        w_min = min(w_pos, total_rank - w_pos)
        outcomes += 1
        if w_min <= observed_w_min + 1e-12:
            extreme += 1
    return extreme / outcomes


def summarize(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    delta_rows: list[dict[str, str]] = []
    pvalue_rows: list[dict[str, str]] = []

    sequences = sorted({row["sequence"] for row in rows})
    for sequence in sequences:
        for method in METHOD_ORDER:
            group = [row for row in rows if row["sequence"] == sequence and row["method"] == method]
            if not group:
                continue

            delta_row = {
                "sequence": sequence,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_cases": str(len(group)),
            }
            pvalue_row = {
                "sequence": sequence,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_cases": str(len(group)),
            }
            for metric_name, pred_col, gt_col in METRICS:
                diffs = [float(row[pred_col]) - float(row[gt_col]) for row in group]
                delta_row[metric_name] = f"{sum(diffs) / len(diffs):+.6f}"
                pvalue_row[metric_name] = f"{exact_wilcoxon_two_sided(diffs):.6f}"
            delta_rows.append(delta_row)
            pvalue_rows.append(pvalue_row)
    return delta_rows, pvalue_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.detail_csv)
    delta_rows, pvalue_rows = summarize(rows)
    delta_path = args.output_dir / "trabecular_delta_summary.csv"
    pvalue_path = args.output_dir / "trabecular_pvalue_summary.csv"
    write_csv(delta_path, delta_rows)
    write_csv(pvalue_path, pvalue_rows)
    print(f"Delta summary:   {delta_path}")
    print(f"P-value summary: {pvalue_path}")


if __name__ == "__main__":
    main()
