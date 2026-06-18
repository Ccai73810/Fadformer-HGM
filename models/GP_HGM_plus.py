import torch
import torch.nn as nn
from models.FADformer import FADBackbone, get_residue, FADformer_HGM_mini, FADformer_HGM, FADformer_mini, FADformer

class DegradationEncoder(nn.Module):
    """
    Lightweight encoder to extract degradation prior (latent z) from the input image.
    Used for implicit generative prior modeling.
    """
    def __init__(self, in_channels=3, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        # x: B, C, H, W
        z = self.encoder(x) # B, latent_dim, 1, 1
        return z.view(z.size(0), -1)

class FiLM_Layer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Modulates backbone features using the degradation latent vector z.
    """
    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, feature_dim * 2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, z):
        # z: B, latent_dim
        # x: B, C, H, W
        scale_shift = self.fc(z) # B, 2*C
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.view(*scale.shape, 1, 1)
        shift = shift.view(*shift.shape, 1, 1)
        return x * (1 + scale) + shift

class GP_HGM_Model(nn.Module):
    """
    GP-HGM++: Generative Prior + Hybrid Global Mixer
    Combines the HGM backbone with a lightweight generative prior (Degradation Encoder + FiLM).
    """
    def __init__(self, backbone, latent_dim=64):
        super().__init__()
        self.backbone = backbone
        self.deg_encoder = DegradationEncoder(in_channels=3, latent_dim=latent_dim)
        
        # Determine feature dims for stage 3 (bottleneck) and stage 5 (output)
        dim3 = self.backbone.layer3.blocks[0].dim
        dim5 = self.backbone.layer5.blocks[0].dim
        
        self.film3 = FiLM_Layer(latent_dim, dim3)
        self.film5 = FiLM_Layer(latent_dim, dim5)
        
    def forward(self, x, return_z=False):
        # 1. Encode Degradation Prior
        z = self.deg_encoder(x) # B, latent_dim
        
        # 2. Forward through modified backbone
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
        # Inject FiLM
        out = self.film3(out, z)
        
        out = b.upsample1(out)
        mask = b.up_rcp1(mask)
        
        out = b.skip2(torch.cat([out, copy2], dim=1))
        out, mask = b.layer4((out, mask))
        
        out = b.upsample2(out)
        mask = b.up_rcp2(mask)
        
        out = b.skip1(torch.cat([out, copy1], dim=1))
        
        # Stage 5 (Output)
        out, mask = b.layer5((out, mask))
        # Inject FiLM
        out = self.film5(out, z)
        
        out = b.patch_unembed(out)
        out = copy0 + out
        
        if return_z:
            return out, z
        return out

def GP_HGM_mini(latent_dim=64, window_size=8, num_heads=8, fusion_mode='gate'):
    backbone = FADformer_HGM_mini(window_size=window_size, num_heads=num_heads, fusion_mode=fusion_mode)
    return GP_HGM_Model(backbone, latent_dim=latent_dim)

def GP_HGM_full(latent_dim=64, window_size=8, num_heads=8, fusion_mode='gate', use_checkpoint=True):
    backbone = FADformer_HGM(window_size=window_size, num_heads=num_heads, fusion_mode=fusion_mode, use_checkpoint=use_checkpoint)
    return GP_HGM_Model(backbone, latent_dim=latent_dim)
