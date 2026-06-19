# -*- coding: utf-8 -*-
"""
GP-HGM++ Original Size Evaluation Script
Evaluates the trained GP-HGM++ model on the test dataset at original resolution.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Setup path
sys.path.insert(0, os.path.abspath('.'))

try:
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

class RainDatasetOriginalSize(Dataset):
    def __init__(self, root, split='test'):
        self.input_dir = os.path.join(root, split, 'input')
        self.target_dir = os.path.join(root, split, 'target')
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Directory not found: {self.input_dir}")
        self.filenames = sorted([f for f in os.listdir(self.input_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        inp = Image.open(os.path.join(self.input_dir, fname)).convert('RGB')
        tgt = Image.open(os.path.join(self.target_dir, fname)).convert('RGB')
        
        inp_tensor = torch.from_numpy(np.array(inp).astype(np.float32).transpose(2, 0, 1) / 255.0)
        tgt_tensor = torch.from_numpy(np.array(tgt).astype(np.float32).transpose(2, 0, 1) / 255.0)
        return {'source': inp_tensor, 'target': tgt_tensor, 'filename': fname}

def rgb_to_y(img):
    r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    y = 0.256789 * r + 0.504129 * g + 0.097906 * b + 16.0 / 255.0
    return y

def calc_psnr_y(img1, img2):
    y1 = rgb_to_y(img1)
    y2 = rgb_to_y(img2)
    mse = torch.mean((y1 - y2) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()

def calc_ssim_y(img1, img2):
    if not HAS_SKIMAGE:
        return 0.0
    y1 = rgb_to_y(img1).squeeze(0).squeeze(0).cpu().numpy() * 255.0
    y2 = rgb_to_y(img2).squeeze(0).squeeze(0).cpu().numpy() * 255.0
    score = ssim_func(y1, y2, data_range=255.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return score

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation running on: {device}")
    
    # Paths
    dataset_dir = './Rain200L'
    ckpt_path = './saved_models/rain200l_gphgm_plus/GP_HGM_plus_best_mini.pth'
    
    if not os.path.exists(ckpt_path):
        # Check if full scale ckpt exists
        ckpt_path = './saved_models/rain200l_gphgm_plus/GP_HGM_plus_best_full.pth'
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found at standard paths.")
            sys.exit(1)
            
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Loading dataset from: {dataset_dir}")
    
    # Import GP-HGM++ Model
    from models.GP_HGM_plus import GP_HGM_mini, GP_HGM_full
    
    # Detect model scale from checkpoint path
    if 'mini' in ckpt_path:
        model = GP_HGM_mini(latent_dim=64, window_size=8, num_heads=4, fusion_mode='gate')
        print("Model configuration: GP-HGM++ Mini (2.45M)")
    else:
        model = GP_HGM_full(latent_dim=64, window_size=8, num_heads=4, fusion_mode='gate', use_checkpoint=False)
        print("Model configuration: GP-HGM++ Full (9.95M)")
        
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    new_state = {}
    for k, v in state.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state[name] = v
        
    model.load_state_dict(new_state)
    model = model.to(device)
    model.eval()
    
    dataset = RainDatasetOriginalSize(dataset_dir, 'test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    img_multiple_of = 32
    
    with torch.no_grad():
        for batch in loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            h, w = source.shape[2], source.shape[3]
            H = ((h + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
            W = ((w + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
            padh = H - h if h % img_multiple_of != 0 else 0
            padw = W - w if w % img_multiple_of != 0 else 0
            
            source_padded = F.pad(source, (0, padw, 0, padh), 'reflect')
            output_padded = model(source_padded).clamp_(0, 1)
            output = output_padded[:, :, :h, :w]
            
            psnr_val = calc_psnr_y(output, target)
            ssim_val = calc_ssim_y(output, target)
            
            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1
            
            if count % 20 == 0 or count == len(loader):
                print(f"Processed {count:3d}/{len(loader):3d} images...")
                
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print("\n" + "="*50)
    print(f" GP-HGM++ Original Size Test Results:")
    print("-"*50)
    print(f" Average PSNR (Y): {avg_psnr:.4f} dB")
    print(f" Average SSIM (Y): {avg_ssim:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
