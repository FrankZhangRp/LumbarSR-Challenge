#!/usr/bin/env python3
import os
import shutil
from typing import Optional, Dict, List, Tuple
import numpy as np
import nibabel as nib
import ants
from scipy.ndimage import zoom


def center_origin(shape: Tuple, spacing: Tuple) -> Tuple:
    return tuple(-s * sp / 2 for s, sp in zip(shape, spacing))


def rotate_xy(arr: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return arr
    return np.rot90(arr, k=rotation, axes=(0, 1))


def dice_coefficient(arr1: np.ndarray, arr2: np.ndarray, threshold: int = 200) -> float:
    mask1 = arr1 > threshold
    mask2 = arr2 > threshold
    intersection = np.sum(mask1 & mask2)
    return 2 * intersection / (np.sum(mask1) + np.sum(mask2) + 1e-8)


Y_TRANSFORMS = [
    (0, "original"),
    (1, "rot180"),
    (2, "flipY"),
]


def apply_y_transform(arr: np.ndarray, transform: int) -> np.ndarray:
    if transform == 0:
        return arr
    elif transform == 1:
        return np.rot90(arr, k=2, axes=(0, 1))
    elif transform == 2:
        return np.flip(arr, axis=1)
    return arr


def register_fov(
    fixed_nii: nib.Nifti1Image,
    ref_path: str,
    all_seq_paths: List[str],
    output_dir: str,
    sample_id: str,
    fov: str,
    downsample_factor: int = 4,
    rotation: int = 0,
    y_transform: int = 0
) -> List[Tuple[str, float]]:
    results = []

    fixed_arr = fixed_nii.get_fdata()
    fixed_affine = fixed_nii.affine
    fixed_spacing = tuple(np.abs(np.diag(fixed_affine)[:3]))

    fixed_ds = zoom(fixed_arr, 1.0 / downsample_factor, order=1)
    fixed_ds_spacing = tuple(s * downsample_factor for s in fixed_spacing)

    fixed_ants_ds = ants.from_numpy(
        fixed_ds.astype(np.float32),
        spacing=fixed_ds_spacing,
        origin=center_origin(fixed_ds.shape, fixed_ds_spacing)
    )
    fixed_ants_orig = ants.from_numpy(
        fixed_arr.astype(np.float32),
        spacing=fixed_spacing,
        origin=center_origin(fixed_arr.shape, fixed_spacing)
    )

    ref_nii = nib.load(ref_path)
    ref_arr = ref_nii.get_fdata()
    ref_spacing = tuple(np.abs(np.diag(ref_nii.affine)[:3]))

    ref_flipped = np.flip(np.flip(ref_arr, 2), 0)
    ref_rotated = rotate_xy(ref_flipped, rotation)
    ref_transformed = apply_y_transform(ref_rotated, y_transform)

    if rotation in [1, 3]:
        ref_spacing = (ref_spacing[1], ref_spacing[0], ref_spacing[2])

    ref_ants = ants.from_numpy(
        ref_transformed.astype(np.float32),
        spacing=ref_spacing,
        origin=center_origin(ref_transformed.shape, ref_spacing)
    )

    print(f"    Registering {fov}um FOV (500Z_B)...")
    reg_result = ants.registration(
        fixed=fixed_ants_ds,
        moving=ref_ants,
        type_of_transform='Rigid',
        aff_metric='mattes',
        aff_sampling=32,
        aff_iterations=(3000, 2000, 1000, 500),
        aff_shrink_factors=(8, 6, 4, 2),
        aff_smoothing_sigmas=(4, 3, 2, 1),
        verbose=False
    )
    transforms = reg_result['fwdtransforms']

    for seq_path in all_seq_paths:
        seq_name = os.path.basename(seq_path)
        seq_suffix = seq_name.replace(f"Lumbar{sample_id}_ClinicalCT_", "").replace(".nii.gz", "")

        seq_nii = nib.load(seq_path)
        seq_arr = seq_nii.get_fdata()
        seq_spacing = tuple(np.abs(np.diag(seq_nii.affine)[:3]))

        seq_flipped = np.flip(np.flip(seq_arr, 2), 0)
        seq_rotated = rotate_xy(seq_flipped, rotation)
        seq_transformed = apply_y_transform(seq_rotated, y_transform)

        if rotation in [1, 3]:
            seq_spacing = (seq_spacing[1], seq_spacing[0], seq_spacing[2])

        seq_ants = ants.from_numpy(
            seq_transformed.astype(np.float32),
            spacing=seq_spacing,
            origin=center_origin(seq_transformed.shape, seq_spacing)
        )

        warped = ants.apply_transforms(
            fixed=fixed_ants_orig,
            moving=seq_ants,
            transformlist=transforms,
            interpolator='linear'
        )

        warped_arr = warped.numpy()
        warped_arr[warped_arr == 0] = -1024
        warped_arr = np.clip(warped_arr, -1024, 32767).astype(np.int16)

        out_name = f"Lumbar{sample_id}_ClinicalCT_{seq_suffix}_registered.nii.gz"
        out_path = os.path.join(output_dir, out_name)
        nib.save(nib.Nifti1Image(warped_arr, fixed_affine), out_path)

        if "500Z_B" in seq_suffix:
            dice = dice_coefficient(fixed_arr, warped_arr)
            results.append((f"{fov}um", dice))
            print(f"      {seq_suffix}: Dice={dice:.3f}")
        else:
            print(f"      {seq_suffix}: saved")

    return results


def register_fov_with_best_rotation(
    fixed_nii: nib.Nifti1Image,
    ref_path: str,
    all_seq_paths: List[str],
    output_dir: str,
    sample_id: str,
    fov: str,
    downsample_factor: int = 4
) -> Tuple[List[Tuple[str, float]], int, str]:
    fixed_arr = fixed_nii.get_fdata()
    fixed_affine = fixed_nii.affine
    fixed_spacing = tuple(np.abs(np.diag(fixed_affine)[:3]))

    fixed_ds = zoom(fixed_arr, 1.0 / downsample_factor, order=1)
    fixed_ds_spacing = tuple(s * downsample_factor for s in fixed_spacing)

    fixed_ants_ds = ants.from_numpy(
        fixed_ds.astype(np.float32),
        spacing=fixed_ds_spacing,
        origin=center_origin(fixed_ds.shape, fixed_ds_spacing)
    )

    ref_nii = nib.load(ref_path)
    ref_arr = ref_nii.get_fdata()
    ref_spacing = tuple(np.abs(np.diag(ref_nii.affine)[:3]))

    ref_flipped = np.flip(np.flip(ref_arr, 2), 0)

    best_dice = -1
    best_transform = 0
    best_transform_name = "original"

    print(f"    Testing Y transforms for {fov}um FOV...")

    for y_t, y_name in Y_TRANSFORMS:
        ref_transformed = apply_y_transform(ref_flipped, y_t)

        ref_ants = ants.from_numpy(
            ref_transformed.astype(np.float32),
            spacing=ref_spacing,
            origin=center_origin(ref_transformed.shape, ref_spacing)
        )

        reg_result = ants.registration(
            fixed=fixed_ants_ds,
            moving=ref_ants,
            type_of_transform='Rigid',
            aff_metric='mattes',
            aff_sampling=32,
            aff_iterations=(1000, 500, 250),
            aff_shrink_factors=(8, 4, 2),
            aff_smoothing_sigmas=(3, 2, 1),
            verbose=False
        )

        warped = reg_result['warpedmovout']
        warped_arr = warped.numpy()
        dice = dice_coefficient(fixed_ds, warped_arr)

        print(f"      {y_name}: Dice={dice:.3f}")

        if dice > best_dice:
            best_dice = dice
            best_transform = y_t
            best_transform_name = y_name

    print(f"    Best Y transform: {best_transform_name} (Dice={best_dice:.3f})")

    results = register_fov(
        fixed_nii, ref_path, all_seq_paths,
        output_dir, sample_id, fov,
        downsample_factor=downsample_factor,
        rotation=0,
        y_transform=best_transform
    )

    return results, best_transform, best_transform_name


def register_sample(
    sample_name: str,
    input_root: str,
    output_root: str,
    try_rotations: bool = False
) -> Tuple[str, bool, List[Tuple[str, float]]]:
    sample_id = sample_name.split("_")[-1]
    input_dir = os.path.join(input_root, sample_name)
    output_dir = os.path.join(output_root, sample_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Registering: {sample_name}")

    try:
        fixed_path = os.path.join(input_dir, f"Lumbar{sample_id}_MicroPCCT_105um.nii.gz")
        if not os.path.exists(fixed_path):
            print(f"  Error: MicroCT not found")
            return (sample_name, False, [])

        fixed_nii = nib.load(fixed_path)

        shutil.copy2(fixed_path, os.path.join(output_dir, f"Lumbar{sample_id}_MicroPCCT_105um.nii.gz"))

        all_results = []

        for fov in ["195", "586"]:
            ref_seq = f"{fov}X_{fov}Y_500Z_B"
            ref_path = os.path.join(input_dir, f"Lumbar{sample_id}_ClinicalCT_{ref_seq}.nii.gz")

            if not os.path.exists(ref_path):
                print(f"  Skipping {fov}um FOV (reference not found)")
                continue

            all_seqs = [
                f"{fov}X_{fov}Y_500Z_B",
                f"{fov}X_{fov}Y_500Z_S",
                f"{fov}X_{fov}Y_1000Z_B",
                f"{fov}X_{fov}Y_1000Z_S",
            ]
            all_seq_paths = []
            for seq in all_seqs:
                seq_path = os.path.join(input_dir, f"Lumbar{sample_id}_ClinicalCT_{seq}.nii.gz")
                if os.path.exists(seq_path):
                    all_seq_paths.append(seq_path)

            if try_rotations:
                results, best_rot, rot_name = register_fov_with_best_rotation(
                    fixed_nii, ref_path, all_seq_paths,
                    output_dir, sample_id, fov
                )
            else:
                results = register_fov(
                    fixed_nii, ref_path, all_seq_paths,
                    output_dir, sample_id, fov
                )
            all_results.extend(results)

        print(f"  {sample_name}: {all_results}")
        return (sample_name, True, all_results)

    except Exception as e:
        print(f"  Error {sample_name}: {e}")
        import traceback
        traceback.print_exc()
        return (sample_name, False, [])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Register clinical CT to MicroPCCT")
    parser.add_argument('--input-root', type=str, default='./data/original_dicom',
                        help='Input directory with cropped data')
    parser.add_argument('--output-root', type=str, default='./data/registered_nifti',
                        help='Output directory for registered data')
    parser.add_argument('--sample', type=str, default='Lumbar_01',
                        help='Sample name to process')
    args = parser.parse_args()

    sample = args.sample
    print(f"\n{'='*60}")
    print(f"Registering Clinical CT to MicroCT: {sample}")
    print(f"{'='*60}")

    sample_name, success, results = register_sample(sample, args.input_root, args.output_root)

    print(f"\n{'='*60}")
    print(f"Result: {'OK' if success else 'FAILED'}")
    if results:
        print(f"Dice scores: {results}")
    print(f"{'='*60}")
