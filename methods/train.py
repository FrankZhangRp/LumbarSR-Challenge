#!/usr/bin/env python3
"""
Training script for super-resolution models.

Supports both SRCNN and UNet architectures.
"""

import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from srcnn_model import SRCNN, CombinedLoss
from unet_model import UNet


class LumbarSRDataset(Dataset):
    """Dataset for Lumbar Super-Resolution."""

    def __init__(self, data_root, samples, sequences, patch_size=None, n_patches_per_volume=10):
        """Initialize dataset.

        Args:
            data_root: Root directory of registered NIfTI data
            samples: List of sample names (e.g., ['Lumbar_01', 'Lumbar_02'])
            sequences: List of sequences to use (e.g., ['195X_195Y_1000Z_S'])
            patch_size: Size of 2D patches (H, W). None for full slices.
            n_patches_per_volume: Number of patches to extract per volume
        """
        self.data_root = Path(data_root)
        self.samples = samples
        self.sequences = sequences
        self.patch_size = patch_size
        self.n_patches_per_volume = n_patches_per_volume

        # Build file list
        self.pairs = []
        for sample in samples:
            sample_id = sample.split("_")[-1]
            sample_dir = self.data_root / sample

            gt_path = sample_dir / f"Lumbar{sample_id}_MicroPCCT_105um.nii.gz"

            if not gt_path.exists():
                continue

            for seq in sequences:
                lr_path = sample_dir / f"Lumbar{sample_id}_ClinicalCT_{seq}_registered.nii.gz"

                if lr_path.exists():
                    self.pairs.append((str(lr_path), str(gt_path)))

        print(f"Found {len(self.pairs)} LR-HR pairs")

    def __len__(self):
        return len(self.pairs) * self.n_patches_per_volume

    def _load_volume(self, lr_path, gt_path):
        """Load LR and HR volumes."""
        lr_nii = nib.load(lr_path)
        gt_nii = nib.load(gt_path)

        lr_vol = lr_nii.get_fdata().astype(np.float32)
        gt_vol = gt_nii.get_fdata().astype(np.float32)

        return lr_vol, gt_vol

    def _extract_patch(self, lr_vol, gt_vol):
        """Extract random 2D patch from volume."""
        # Random slice
        d = np.random.randint(0, lr_vol.shape[2])

        lr_slice = lr_vol[:, :, d]
        gt_slice = gt_vol[:, :, d]

        # Random crop if patch_size specified
        if self.patch_size is not None:
            h, w = self.patch_size
            H, W = lr_slice.shape

            if H > h and W > w:
                y = np.random.randint(0, H - h)
                x = np.random.randint(0, W - w)
                lr_slice = lr_slice[y:y+h, x:x+w]
                gt_slice = gt_slice[y:y+h, x:x+w]

        return lr_slice, gt_slice

    def __getitem__(self, idx):
        """Get a single training sample."""
        # Get volume pair
        vol_idx = idx // self.n_patches_per_volume
        lr_path, gt_path = self.pairs[vol_idx]

        # Load volumes
        lr_vol, gt_vol = self._load_volume(lr_path, gt_path)

        # Extract patch
        lr_patch, gt_patch = self._extract_patch(lr_vol, gt_vol)

        # Normalize to [0, 1]
        lr_patch = np.clip(lr_patch, -1024, 3071)
        lr_patch = (lr_patch + 1024) / 4095.0

        gt_patch = np.clip(gt_patch, -1024, 3071)
        gt_patch = (gt_patch + 1024) / 4095.0

        # Convert to tensor
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(gt_patch).unsqueeze(0).float()

        return lr_tensor, gt_tensor


class DualChannelDataset(Dataset):
    """Dataset with dual-channel input (small FOV + large FOV)."""

    def __init__(self, data_root, samples, patch_size=(256, 256), n_patches_per_volume=10):
        """Initialize dual-channel dataset.

        Args:
            data_root: Root directory of registered NIfTI data
            samples: List of sample names
            patch_size: Size of patches (H, W)
            n_patches_per_volume: Number of patches per volume
        """
        self.data_root = Path(data_root)
        self.samples = samples
        self.patch_size = patch_size
        self.n_patches_per_volume = n_patches_per_volume

        # Build file list
        self.pairs = []
        for sample in samples:
            sample_id = sample.split("_")[-1]
            sample_dir = self.data_root / sample

            gt_path = sample_dir / f"Lumbar{sample_id}_MicroPCCT_105um.nii.gz"
            lr_small = sample_dir / f"Lumbar{sample_id}_ClinicalCT_195X_195Y_1000Z_S_registered.nii.gz"
            lr_large = sample_dir / f"Lumbar{sample_id}_ClinicalCT_586X_586Y_1000Z_S_registered.nii.gz"

            if gt_path.exists() and lr_small.exists() and lr_large.exists():
                self.pairs.append((str(lr_small), str(lr_large), str(gt_path)))

        print(f"Found {len(self.pairs)} dual-channel volume pairs")

    def __len__(self):
        return len(self.pairs) * self.n_patches_per_volume

    def __getitem__(self, idx):
        """Get dual-channel sample."""
        vol_idx = idx // self.n_patches_per_volume
        lr_small_path, lr_large_path, gt_path = self.pairs[vol_idx]

        # Load volumes
        lr_small = nib.load(lr_small_path).get_fdata().astype(np.float32)
        lr_large = nib.load(lr_large_path).get_fdata().astype(np.float32)
        gt = nib.load(gt_path).get_fdata().astype(np.float32)

        # Random slice and crop
        d = np.random.randint(0, lr_small.shape[2])
        h, w = self.patch_size

        lr_small_slice = lr_small[:, :, d]
        lr_large_slice = lr_large[:, :, d]
        gt_slice = gt[:, :, d]

        # Resize large FOV to match small FOV
        from scipy.ndimage import zoom
        scale_h = h / lr_large_slice.shape[0]
        scale_w = w / lr_large_slice.shape[1]
        lr_large_slice = zoom(lr_large_slice, (scale_h, scale_w), order=1)

        # Random crop
        H, W = lr_small_slice.shape
        if H > h and W > w:
            y = np.random.randint(0, H - h)
            x = np.random.randint(0, W - w)
            lr_small_slice = lr_small_slice[y:y+h, x:x+w]
            gt_slice = gt_slice[y:y+h, x:x+w]

        # Normalize
        def normalize(x):
            x = np.clip(x, -1024, 3071)
            return (x + 1024) / 4095.0

        lr_small_slice = normalize(lr_small_slice)
        lr_large_slice = normalize(lr_large_slice)
        gt_slice = normalize(gt_slice)

        # Stack channels
        lr_input = np.stack([lr_small_slice, lr_large_slice], axis=0)
        lr_tensor = torch.from_numpy(lr_input).float()
        gt_tensor = torch.from_numpy(gt_slice).unsqueeze(0).float()

        return lr_tensor, gt_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for lr, gt in pbar:
        lr = lr.to(device)
        gt = gt.to(device)

        # Forward
        optimizer.zero_grad()
        pred = model(lr)

        # Loss
        loss = criterion(pred, gt)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for lr, gt in tqdm(dataloader, desc="Validation"):
            lr = lr.to(device)
            gt = gt.to(device)

            pred = model(lr)
            loss = criterion(pred, gt)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train SR model")
    parser.add_argument('--model', type=str, default='srcnn', choices=['srcnn', 'unet'])
    parser.add_argument('--data-root', type=str, default='/data/LumbarSR/registered_nifti')
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch-size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--n-patches', type=int, default=10)
    parser.add_argument('--dual-channel', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Training samples (Lumbar_01 to Lumbar_25)
    train_samples = [f"Lumbar_{i:02d}" for i in range(1, 26)]

    # Dataset
    if args.dual_channel:
        dataset = DualChannelDataset(
            data_root=args.data_root,
            samples=train_samples,
            patch_size=tuple(args.patch_size),
            n_patches_per_volume=args.n_patches
        )
        in_channels = 2
    else:
        dataset = LumbarSRDataset(
            data_root=args.data_root,
            samples=train_samples,
            sequences=["195X_195Y_1000Z_S"],
            patch_size=tuple(args.patch_size),
            n_patches_per_volume=args.n_patches
        )
        in_channels = 1

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model
    if args.model == 'srcnn':
        from srcnn_model import get_model
        model = get_model('2d', in_channels=in_channels, num_features=64)
    else:
        from unet_model import get_model
        model = get_model('2d', in_channels=in_channels, base_features=64)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()}, Parameters: {n_params:,}")

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        scheduler.step(train_loss)

        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(args.output_dir, f"{args.model}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"Saved best model: {checkpoint_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
