#!/usr/bin/env python3
"""
UNet model for CT super-resolution.

U-shaped architecture with encoder-decoder and skip connections.
Originally designed for segmentation, effective for SR tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (Conv -> ReLU) x 2."""

    def __init__(self, in_channels, out_channels):
        """Initialize double convolution block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net architecture for super-resolution.

    Encoder: Contracting path with max pooling
    Decoder: Expanding path with upsampling
    Skip connections: Concatenate encoder features to decoder
    """

    def __init__(self, in_channels=1, base_features=64):
        """Initialize UNet.

        Args:
            in_channels: Number of input channels
            base_features: Base number of feature maps (doubles/downs each level)
        """
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_features)
        self.enc2 = DoubleConv(base_features, base_features * 2)
        self.enc3 = DoubleConv(base_features * 2, base_features * 4)
        self.enc4 = DoubleConv(base_features * 4, base_features * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(base_features * 8, base_features * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_features * 16, base_features * 8)

        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features * 2, base_features)

        # Final output layer
        self.final = nn.Conv2d(base_features, 1, kernel_size=1)

        # Max pooling for encoder
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _crop_and_concat(self, upsampled, encoder_feat):
        """Crop encoder feature to match upsampled size and concatenate."""
        _, _, h_up, w_up = upsampled.size()
        _, _, h_enc, w_enc = encoder_feat.size()

        # Crop if needed
        if h_enc != h_up or w_enc != w_up:
            dh = (h_enc - h_up) // 2
            dw = (w_enc - w_up) // 2
            encoder_feat = encoder_feat[:, :, dh:dh+h_up, dw:dw+w_up]

        return torch.cat([upsampled, encoder_feat], dim=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Super-resolved output [B, 1, H, W]
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        up4 = self.up4(bottleneck)
        up4 = self._crop_and_concat(up4, enc4)
        dec4 = self.dec4(up4)

        up3 = self.up3(dec4)
        up3 = self._crop_and_concat(up3, enc3)
        dec3 = self.dec3(up3)

        up2 = self.up2(dec3)
        up2 = self._crop_and_concat(up2, enc2)
        dec2 = self.dec2(up2)

        up1 = self.up1(dec2)
        up1 = self._crop_and_concat(up1, enc1)
        dec1 = self.dec1(up1)

        # Output
        return self.final(dec1)


class UNet3D(nn.Module):
    """3D U-Net for volumetric super-resolution."""

    def __init__(self, in_channels=1, base_features=32):
        """Initialize 3D UNet.

        Args:
            in_channels: Number of input channels
            base_features: Base number of feature maps
        """
        super(UNet3D, self).__init__()

        # Encoder
        self.enc1 = self._make_conv_block(in_channels, base_features)
        self.enc2 = self._make_conv_block(base_features, base_features * 2)
        self.enc3 = self._make_conv_block(base_features * 2, base_features * 4)

        # Bottleneck
        self.bottleneck = self._make_conv_block(base_features * 4, base_features * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = self._make_conv_block(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = self._make_conv_block(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = self._make_conv_block(base_features * 2, base_features)

        # Final output
        self.final = nn.Conv3d(base_features, 1, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def _make_conv_block(self, in_ch, out_ch):
        """Create 3D double conv block."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def _crop_and_concat(self, upsampled, encoder_feat):
        """Crop encoder feature and concatenate."""
        _, _, d_up, h_up, w_up = upsampled.size()
        _, _, d_enc, h_enc, w_enc = encoder_feat.size()

        if d_enc != d_up or h_enc != h_up or w_enc != w_up:
            dd = (d_enc - d_up) // 2
            dh = (h_enc - h_up) // 2
            dw = (w_enc - w_up) // 2
            encoder_feat = encoder_feat[:, :, dd:dd+d_up, dh:dh+h_up, dw:dw+w_up]

        return torch.cat([upsampled, encoder_feat], dim=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input [B, C, D, H, W]

        Returns:
            Output [B, 1, D, H, W]
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder
        up3 = self.up3(bottleneck)
        up3 = self._crop_and_concat(up3, enc3)
        dec3 = self.dec3(up3)

        up2 = self.up2(dec3)
        up2 = self._crop_and_concat(up2, enc2)
        dec2 = self.dec2(up2)

        up1 = self.up1(dec2)
        up1 = self._crop_and_concat(up1, enc1)
        dec1 = self.dec1(up1)

        return self.final(dec1)


def get_model(model_type='2d', in_channels=1, base_features=64):
    """Get UNet model.

    Args:
        model_type: '2d' for slice-based, '3d' for volume-based
        in_channels: Number of input channels
        base_features: Base number of feature maps

    Returns:
        UNet model
    """
    if model_type == '2d':
        return UNet(in_channels=in_channels, base_features=base_features)
    elif model_type == '3d':
        return UNet3D(in_channels=in_channels, base_features=base_features)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    # Test 2D model
    model_2d = UNet(in_channels=2, base_features=64)
    x_2d = torch.randn(2, 2, 256, 256)
    out_2d = model_2d(x_2d)

    print(f"2D UNet:")
    print(f"  Input shape: {x_2d.shape}")
    print(f"  Output shape: {out_2d.shape}")
    n_params_2d = sum(p.numel() for p in model_2d.parameters())
    print(f"  Parameters: {n_params_2d:,}")

    # Test 3D model
    model_3d = UNet3D(in_channels=2, base_features=32)
    x_3d = torch.randn(1, 2, 64, 256, 256)
    out_3d = model_3d(x_3d)

    print(f"\n3D UNet:")
    print(f"  Input shape: {x_3d.shape}")
    print(f"  Output shape: {out_3d.shape}")
    n_params_3d = sum(p.numel() for p in model_3d.parameters())
    print(f"  Parameters: {n_params_3d:,}")
