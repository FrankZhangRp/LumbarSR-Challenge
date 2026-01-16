#!/usr/bin/env python3
import os
import sys
import time
import argparse
from multiprocessing import Pool
from typing import List, Tuple

from register_ants import register_sample


INPUT_ROOT = "/data/wangping_16T/LumbarChallenge2026/CroppedData"
OUTPUT_ROOT = "/data/wangping_16T/LumbarChallenge2026/RegisteredData"
ALL_SAMPLES = [f"Lumbar_{i:02d}" for i in range(1, 31)]

TRY_ROTATIONS = False


def register_worker(sample_name: str) -> Tuple[str, bool, List]:
    return register_sample(sample_name, INPUT_ROOT, OUTPUT_ROOT, try_rotations=TRY_ROTATIONS)


def main():
    global TRY_ROTATIONS
    parser = argparse.ArgumentParser(description="Batch registration of Clinical CT to MicroCT")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--samples', nargs='+', default=None)
    parser.add_argument('--try-rotations', action='store_true')
    args = parser.parse_args()

    TRY_ROTATIONS = args.try_rotations
    samples = args.samples if args.samples else ALL_SAMPLES

    print(f"\n{'#'*60}")
    print(f"# Batch Registration")
    print(f"# Samples: {len(samples)}, Workers: {args.workers}")
    print(f"{'#'*60}\n")

    t0 = time.time()

    with Pool(processes=args.workers) as pool:
        results = pool.map(register_worker, samples)

    total_time = time.time() - t0

    success_count = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Success: {success_count}/{len(samples)} samples")

    print(f"\nDice scores:")
    for sample_name, ok, dice_results in results:
        if ok and dice_results:
            scores = ", ".join([f"{fov}={d:.3f}" for fov, d in dice_results])
            print(f"  {sample_name}: {scores}")
        elif not ok:
            print(f"  {sample_name}: FAILED")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
