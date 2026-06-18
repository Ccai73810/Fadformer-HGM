import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from PIL import Image
from models.FADformer import FADformer
from torch.amp import autocast

DATA_DIR = r'D:\BaiduNetdiskDownload\Rain200H\Rain200H'
PRETRAINED = './pretrain_weights/rain200H/FADformer_Rain200H.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()

# Load model
model = FADformer()
ckpt = torch.load(PRETRAINED, map_location='cpu')
state = ckpt.get('state_dict', ckpt)
new_state = {}
for k, v in state.items():
    name = k.replace('module.', '') if k.startswith('module.') else k
    new_state[name] = v
model.load_state_dict(new_state)
model = model.to(device)
model.eval()

# Let's test on the first 5 images of the test set
test_input_dir = os.path.join(DATA_DIR, 'test', 'input')
test_target_dir = os.path.join(DATA_DIR, 'test', 'target')
filenames = sorted([f for f in os.listdir(test_input_dir) if f.endswith('.png')])[:5]

for mode in ['FP32 (no autocast)', 'FP16 (autocast)']:
    psnrs = []
    
    for fname in filenames:
        inp = Image.open(os.path.join(test_input_dir, fname)).convert('RGB')
        tgt = Image.open(os.path.join(test_target_dir, fname)).convert('RGB')
        
        # Resize to 256x256
        inp = inp.resize((256, 256), Image.BICUBIC)
        tgt = tgt.resize((256, 256), Image.BICUBIC)
        
        # Convert to tensor
        inp_t = torch.from_numpy(np.array(inp).astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        tgt_t = torch.from_numpy(np.array(tgt).astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if 'autocast' in mode and device.type == 'cuda':
                with autocast('cuda'):
                    out_t = model(inp_t).clamp_(0, 1)
            else:
                out_t = model(inp_t).clamp_(0, 1)
                
        psnrs.append(calc_psnr(out_t, tgt_t))
        
    print(f"{mode} Model PSNR: {np.mean(psnrs):.2f} dB")
