#!/usr/bin/env python3
"""Inference script for public 2D SR baselines.

Runs one model load per job and processes the full case as a batch of 2D slices.
"""

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
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    if any(key.startswith('module.') for key in state_dict):
        state_dict = {
            (key[7:] if key.startswith('module.') else key): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    if isinstance(checkpoint, dict):
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        loss = checkpoint.get('loss')
        if loss is not None:
            print(f"  Loss: {loss:.4f}")

    return model


def denormalize(x):
    """Convert normalized [0,1] back to HU units."""
    return x * 4095.0 - 1024


def normalize_hu(x):
    """Normalize HU to [0, 1]."""
    x = np.clip(x, -1024, 3071)
    return (x + 1024) / 4095.0


def inference_volume(model, lr_path, gt_path, device, paired_path=None, batch_size=64):
    """Run full-slice batched inference on one registered volume."""
    lr = nib.load(lr_path).get_fdata().astype(np.float32)
    paired = nib.load(paired_path).get_fdata().astype(np.float32) if paired_path is not None else None
    gt_nii = nib.load(gt_path)

    lr = normalize_hu(lr)
    if paired is not None:
        paired = normalize_hu(paired)
        if paired.shape != lr.shape:
            raise ValueError(f"Paired sequence shape mismatch: {paired.shape} vs {lr.shape}")

    channels = [lr] if paired is None else [lr, paired]
    slices = np.ascontiguousarray(
        np.stack([np.transpose(volume, (2, 0, 1)) for volume in channels], axis=1).astype(np.float32, copy=False)
    )
    d = slices.shape[0]
    preds = np.zeros((d, slices.shape[2], slices.shape[3]), dtype=np.float32)

    with torch.inference_mode():
        for z_start in tqdm(range(0, d, batch_size), desc="Processing slice batches"):
            z_end = min(z_start + batch_size, d)
            batch = torch.from_numpy(slices[z_start:z_end]).float().to(device, non_blocking=True)
            pred = model(batch).squeeze(1).cpu().numpy().astype(np.float32)
            preds[z_start:z_end] = pred

    sr_volume = denormalize(np.transpose(preds, (1, 2, 0)))
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
    batch_size=64,
    model_tag=None,
):
    """Run inference on multiple samples."""
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
        resolved_tag = model_tag or model.__class__.__name__.lower()
        sr_volume, sr_affine = inference_volume(
            model,
            str(lr_path),
            str(gt),
            device,
            paired_path=str(paired_path) if paired_path is not None else None,
            batch_size=batch_size,
        )

        # Save result
        output_dir = Path(output_root) / resolved_tag / sample
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"Lumbar{sample_id}_{sequence}_{resolved_tag}.nii.gz"
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
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--model-tag', type=str, default=None,
                        help="Output filename tag; defaults to the model name")
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
        batch_size=args.batch_size,
        model_tag=args.model_tag or args.model,
    )


if __name__ == "__main__":
    main()
