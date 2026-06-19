# -*- coding: utf-8 -*-
"""
GP-HGM++ Visualization and Comparison Script
Loads the trained model weights and saves side-by-side deraining visual comparisons.
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Setup path
sys.path.insert(0, os.path.abspath('.'))

def find_dataset_dir(dataset_name):
    candidates = [
        # Local Windows Baidu Download paths
        os.path.join(r'D:\BaiduNetdiskDownload', dataset_name, dataset_name),
        os.path.join(r'D:\BaiduNetdiskDownload', dataset_name),
        # Relative paths
        os.path.join('.', dataset_name),
        os.path.join('..', dataset_name),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'test', 'input')):
            return os.path.abspath(path)
    return None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Visualization script running on: {device}")
    
    # Paths
    dataset_dir = find_dataset_dir('Rain200L')
    ckpt_path = './saved_models/rain200l_gphgm_plus/GP_HGM_plus_best_mini.pth'
    output_dir = './comparison_results'
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Trained checkpoint not found at {ckpt_path}.")
        print("Please make sure you have successfully downloaded the weights from the server using SCP.")
        sys.exit(1)
        
    if dataset_dir is None:
        print(f"Error: Rain200L test dataset directory not found in candidate paths.")
        sys.exit(1)
            
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    from models.GP_HGM_plus import GP_HGM_mini
    model = GP_HGM_mini(latent_dim=64, window_size=8, num_heads=4, fusion_mode='gate')
    
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    new_state = {}
    for k, v in state.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        # Map old sequential film layers to new linear layers
        name = name.replace('fc.0.weight', 'fc.weight')
        name = name.replace('fc.0.bias', 'fc.bias')
        new_state[name] = v
        
    model.load_state_dict(new_state)
    model = model.to(device)
    model.eval()
    
    # Load first 5 sample images from test set
    input_folder = os.path.join(dataset_dir, 'test', 'input')
    target_folder = os.path.join(dataset_dir, 'test', 'target')
    filenames = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))])[:5]
    
    print(f"Found {len(filenames)} sample images for visualization.")
    img_multiple_of = 32
    
    for fname in filenames:
        print(f"Processing image: {fname}...")
        inp_img = Image.open(os.path.join(input_folder, fname)).convert('RGB')
        tgt_img = Image.open(os.path.join(target_folder, fname)).convert('RGB')
        
        # Preprocess to tensor
        inp_tensor = torch.from_numpy(np.array(inp_img).astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        
        h, w = inp_tensor.shape[2], inp_tensor.shape[3]
        H = ((h + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        W = ((w + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        
        with torch.no_grad():
            source_padded = F.pad(inp_tensor, (0, padw, 0, padh), 'reflect')
            output_padded = model(source_padded).clamp_(0, 1)
            output = output_padded[:, :, :h, :w]
            
        # Convert output back to image
        out_np = (output.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        out_img = Image.fromarray(out_np)
        
        # Merge side-by-side: Input | GP-HGM++ Output | Ground Truth
        width, height = inp_img.size
        combined = Image.new('RGB', (width * 3, height))
        combined.paste(inp_img, (0, 0))
        combined.paste(out_img, (width, 0))
        combined.paste(tgt_img, (width * 2, 0))
        
        # Save comparison visual
        save_path = os.path.join(output_dir, f"comparison_{os.path.splitext(fname)[0]}.png")
        combined.save(save_path)
        print(f"Saved side-by-side comparison to: {save_path}")
        
    print("\n" + "="*50)
    print(" Visualizations completed successfully!")
    print(f" Results are saved in the directory: {output_dir}")
    print("="*50)

if __name__ == '__main__':
    main()
