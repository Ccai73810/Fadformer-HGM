# -*- coding: utf-8 -*-
"""
SS-RF: Spectral-Spatial Rectified Flow Model
A Mathematically Constrained Generative Restoration Model for FADformer-HGM++
"""

import torch
import torch.nn as nn
from models.FADformer import FADBackbone, get_residue, FADformer_HGM_mini, FADformer_HGM

class RepConv2d(nn.Module):
    """
    Structural Re-parameterization Conv Layer.
    Multi-branch structure during training; Fused into a single Conv2d layer for inference.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.deploy = False
        
        # Training branches
        self.branch_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
        self.branch_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)
        if in_channels == out_channels and stride == 1:
            self.branch_identity = nn.Identity()
        else:
            self.branch_identity = None
            
        # Reparameterized single branch (deployment)
        self.rbr_reparam = None

    def forward(self, x):
        if self.deploy:
            return self.rbr_reparam(x)
            
        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.branch_identity is not None:
            out = out + self.branch_identity(x)
        return out
        
    def switch_to_deploy(self):
        if self.deploy:
            return
            
        # Retrieve weights and biases
        w_3x3 = self.branch_3x3.weight
        b_3x3 = self.branch_3x3.bias
        w_1x1 = self.branch_1x1.weight
        b_1x1 = self.branch_1x1.bias
        
        # Pad 1x1 weight from shape (out, in, 1, 1) to (out, in, 3, 3)
        w_1x1_padded = nn.functional.pad(w_1x1, [1, 1, 1, 1])
        
        # Identity mapping to 3x3 weight
        w_id = 0
        b_id = 0
        if self.branch_identity is not None:
            input_dim = self.in_channels
            w_id = torch.zeros((input_dim, input_dim, 3, 3), device=w_3x3.device)
            for i in range(input_dim):
                w_id[i, i, 1, 1] = 1.0
            b_id = torch.zeros(input_dim, device=w_3x3.device)
            
        # Mathematically fuse weights and biases
        fused_weight = w_3x3 + w_1x1_padded + w_id
        fused_bias = b_3x3 + b_1x1 + b_id
        
        # Construct deployment single Conv2d layer
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )
        self.rbr_reparam.weight.data.copy_(fused_weight)
        self.rbr_reparam.bias.data.copy_(fused_bias)
        
        # Flip deploy switch and delete multi-branch variables
        self.deploy = True
        del self.branch_3x3
        del self.branch_1x1
        if self.branch_identity is not None:
            del self.branch_identity


class RepDegradationEncoder(nn.Module):
    """
    High-capacity degradation encoder using Structural Re-parameterization (RepConv).
    Zero inference overhead after reparameterization.
    """
    def __init__(self, in_channels=3, latent_dim=64):
        super().__init__()
        self.enc1 = RepConv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.enc2 = RepConv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.enc3 = RepConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.enc4 = RepConv2d(64, latent_dim, kernel_size=3, stride=2, padding=1)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.act1(self.enc1(x))
        x = self.act2(self.enc2(x))
        x = self.act3(self.enc3(x))
        x = self.act4(self.enc4(x))
        z = self.pool(x)
        return z.view(z.size(0), -1)
        
    def switch_to_deploy(self):
        self.enc1.switch_to_deploy()
        self.enc2.switch_to_deploy()
        self.enc3.switch_to_deploy()
        self.enc4.switch_to_deploy()


class SpectralDriftModulator(nn.Module):
    """
    SDM (Spectral Drift Modulator): Directly modulates features in complex Fourier Space.
    Specifically modulates Amplitude and Phase based on the degradation prior z.
    """
    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.fc_amp = nn.Sequential(
            nn.Linear(latent_dim, feature_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.fc_phase = nn.Sequential(
            nn.Linear(latent_dim, feature_dim * 2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, z):
        # x: B, C, H, W (Spatial Feature map)
        # z: B, latent_dim (Degradation prior)
        B, C, H, W = x.shape
        
        # 1. Fourier Transform into Complex Frequency Domain
        fft_x = torch.fft.rfft2(x, norm='ortho')
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        # 2. Compute Amplitude Modulation (scale + shift)
        amp_params = self.fc_amp(z) # B, 2*C
        scale_a, shift_a = amp_params.chunk(2, dim=1)
        scale_a = torch.sigmoid(scale_a).view(B, C, 1, 1) # bound scale to prevent explosion
        shift_a = shift_a.view(B, C, 1, 1)
        
        # 3. Compute Phase Modulation (scale + shift)
        phase_params = self.fc_phase(z) # B, 2*C
        scale_p, shift_p = phase_params.chunk(2, dim=1)
        scale_p = torch.tanh(scale_p).view(B, C, 1, 1) # bound phase scale
        shift_p = shift_p.view(B, C, 1, 1)
        
        # 4. Apply Modulations to Amplitude and Phase
        amp_mod = amp * (1.0 + scale_a) + shift_a
        phase_mod = phase * (1.0 + scale_p) + shift_p
        
        # 5. Reconstruct Complex Spectrum and Perform Inverse RFFT2D
        fft_x_mod = torch.polar(amp_mod, phase_mod)
        x_mod = torch.fft.irfft2(fft_x_mod, s=(H, W), norm='ortho')
        
        return x_mod


class SS_RF_Model(nn.Module):
    """
    SS-RF (Spectral-Spatial Rectified Flow Model):
    Integrates FADformer-HGM backbone with mathematically constrained flow matching components.
    """
    def __init__(self, backbone, latent_dim=64):
        super().__init__()
        self.backbone = backbone
        self.deg_encoder = RepDegradationEncoder(in_channels=3, latent_dim=latent_dim)
        
        # Determine feature dimensions dynamically for bottleneck (layer3) and output (layer5)
        dim3 = self.backbone.layer3.blocks[0].dim
        dim5 = self.backbone.layer5.blocks[0].dim
        
        self.sdm3 = SpectralDriftModulator(latent_dim, dim3)
        self.sdm5 = SpectralDriftModulator(latent_dim, dim5)
        
    def forward(self, x, return_z=False):
        # 1. Extract Degradation Prior
        z = self.deg_encoder(x) # B, latent_dim
        
        # 2. Forward pass through HGM Backbone with Spectral Drift Modulation
        b = self.backbone
        copy0 = x
        mask = get_residue(x)
        
        out = b.patch_embed(x)
        out, mask = b.layer1((out, mask))
        copy1 = out
        
        out = b.downsample1(out)
        mask = b.down_rcp1(mask)
        
        out, mask = b.layer2((out, mask))
        copy2 = out
        
        out = b.downsample2(out)
        mask = b.down_rcp2(mask)
        
        # Stage 3 (Bottleneck)
        out, mask = b.layer3((out, mask))
        # Inject Fourier domain drift modulation
        out = self.sdm3(out, z)
        
        out = b.upsample1(out)
        mask = b.up_rcp1(mask)
        
        out = b.skip2(torch.cat([out, copy2], dim=1))
        out, mask = b.layer4((out, mask))
        
        out = b.upsample2(out)
        mask = b.up_rcp2(mask)
        
        out = b.skip1(torch.cat([out, copy1], dim=1))
        
        # Stage 5 (Output)
        out, mask = b.layer5((out, mask))
        # Inject Fourier domain drift modulation
        out = self.sdm5(out, z)
        
        out = b.patch_unembed(out)
        out = copy0 + out
        
        if return_z:
            return out, z
        return out

    def switch_to_deploy(self):
        """
        Merge RepConv branches in the degradation encoder for zero-overhead inference.
        """
        self.deg_encoder.switch_to_deploy()


def SS_RF_mini(latent_dim=64, window_size=8, num_heads=8, fusion_mode='gate'):
    """Factory function for ultra-lightweight SS-RF model"""
    backbone = FADformer_HGM_mini(window_size=window_size, num_heads=num_heads, fusion_mode=fusion_mode)
    return SS_RF_Model(backbone, latent_dim=latent_dim)


def SS_RF_full(latent_dim=64, window_size=8, num_heads=8, fusion_mode='gate', use_checkpoint=True):
    """Factory function for full-parameter SS-RF model"""
    backbone = FADformer_HGM(window_size=window_size, num_heads=num_heads, fusion_mode=fusion_mode, use_checkpoint=use_checkpoint)
    return SS_RF_Model(backbone, latent_dim=latent_dim)
