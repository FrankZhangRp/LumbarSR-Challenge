#!/usr/bin/env python3
"""
SRCNN model for CT super-resolution.

Simple implementation of Super-Resolution Convolutional Neural Network.
Reference: Dong et al. "Image Super-Resolution Using Deep Convolutional Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network.

    Architecture:
    1. Patch extraction and representation: f1=9x9, n1=64, ReLU
    2. Non-linear mapping: f2=1x1, n2=32, ReLU
    3. Reconstruction: f3=5x5, n3=1

    For 2D slice-based super-resolution.
    """

    def __init__(self, in_channels=1, num_features=64):
        """Initialize SRCNN.

        Args:
            in_channels: Number of input channels (1=grayscale, 2=dual-channel)
            num_features: Number of feature maps in first layer
        """
        super(SRCNN, self).__init__()

        # Layer 1: Patch extraction (9x9 conv)
        self.conv1 = nn.Conv2d(
            in_channels, num_features,
            kernel_size=9, padding=4, bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)

        # Layer 2: Non-linear mapping (1x1 conv)
        self.conv2 = nn.Conv2d(
            num_features, num_features // 2,
            kernel_size=1, padding=0, bias=True
        )
        self.relu2 = nn.ReLU(inplace=True)

        # Layer 3: Reconstruction (5x5 conv)
        self.conv3 = nn.Conv2d(
            num_features // 2, 1,
            kernel_size=5, padding=2, bias=True
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Super-resolved output [B, 1, H, W]
        """
        # Patch extraction
        x = self.relu1(self.conv1(x))

        # Non-linear mapping
        x = self.relu2(self.conv2(x))

        # Reconstruction
        x = self.conv3(x)

        return x


class SRCNN3D(nn.Module):
    """3D version of SRCNN for volumetric data.

    Similar architecture but uses 3D convolutions.
    More memory-intensive but processes volume as whole.
    """

    def __init__(self, in_channels=1, num_features=64):
        """Initialize 3D SRCNN.

        Args:
            in_channels: Number of input channels
            num_features: Number of feature maps
        """
        super(SRCNN3D, self).__init__()

        # Layer 1: Patch extraction (3D conv)
        self.conv1 = nn.Conv3d(
            in_channels, num_features,
            kernel_size=(3, 9, 9), padding=(1, 4, 4), bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)

        # Layer 2: Non-linear mapping
        self.conv2 = nn.Conv3d(
            num_features, num_features // 2,
            kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=True
        )
        self.relu2 = nn.ReLU(inplace=True)

        # Layer 3: Reconstruction
        self.conv3 = nn.Conv3d(
            num_features // 2, 1,
            kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=True
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Super-resolved output [B, 1, D, H, W]
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


# Loss functions
class CombinedLoss(nn.Module):
    """Combined loss: L1 + L2 + SSIM."""

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1):
        """Initialize combined loss.

        Args:
            alpha: Weight for L1 loss
            beta: Weight for L2 loss
            gamma: Weight for (1 - SSIM)
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def ssim(self, pred, target, window_size=11):
        """Calculate SSIM for 3D volumes."""
        # Simplified SSIM for 3D
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)

        sigma1 = torch.mean((pred - mu1) ** 2)
        sigma2 = torch.mean((target - mu2) ** 2)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))

        ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

        return ssim_val

    def forward(self, pred, target):
        """Calculate combined loss."""
        l1 = self.l1_loss(pred, target)
        l2 = self.mse_loss(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)

        return self.alpha * l1 + self.beta * l2 + self.gamma * ssim_loss


def get_model(model_type='2d', in_channels=1, num_features=64):
    """Get SRCNN model.

    Args:
        model_type: '2d' for slice-based, '3d' for volume-based
        in_channels: Number of input channels
        num_features: Number of feature maps

    Returns:
        SRCNN model
    """
    if model_type == '2d':
        return SRCNN(in_channels=in_channels, num_features=num_features)
    elif model_type == '3d':
        return SRCNN3D(in_channels=in_channels, num_features=num_features)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    # Test model
    model = SRCNN(in_channels=2, num_features=64)

    # Forward pass
    x = torch.randn(2, 2, 256, 256)
    out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
