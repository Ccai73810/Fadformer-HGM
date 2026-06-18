import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from PIL import Image
import random
from models.FADformer import FADformer

DATA_DIR = r'D:\BaiduNetdiskDownload\Rain200H\Rain200H'
PRETRAINED = './pretrain_weights/rain200H/FADformer_Rain200H.pth'

device = torch.device('cpu')

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

print("=== Evaluation Modes ===")

for mode in ['resize', 'crop', 'original']:
    psnrs = []
    direct_psnrs = []
    
    for fname in filenames:
        inp = Image.open(os.path.join(test_input_dir, fname)).convert('RGB')
        tgt = Image.open(os.path.join(test_target_dir, fname)).convert('RGB')
        
        if mode == 'resize':
            inp = inp.resize((256, 256), Image.BICUBIC)
            tgt = tgt.resize((256, 256), Image.BICUBIC)
        elif mode == 'crop':
            w, h = inp.size
            x = (w - 256) // 2
            y = (h - 256) // 2
            inp = inp.crop((x, y, x + 256, y + 256))
            tgt = tgt.crop((x, y, x + 256, y + 256))
        
        # Convert to tensor
        inp_t = torch.from_numpy(np.array(inp).astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        tgt_t = torch.from_numpy(np.array(tgt).astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        
        # Pad if original size (must be multiple of 4)
        if mode == 'original':
            h, w = inp_t.shape[2], inp_t.shape[3]
            H = ((h + 3) // 4) * 4
            W = ((w + 3) // 4) * 4
            padh = H - h
            padw = W - w
            import torch.nn.functional as F
            inp_t_pad = F.pad(inp_t, (0, padw, 0, padh), 'reflect')
            with torch.no_grad():
                out_t_pad = model(inp_t_pad).clamp_(0, 1)
            out_t = out_t_pad[:, :, :h, :w]
        else:
            with torch.no_grad():
                out_t = model(inp_t).clamp_(0, 1)
                
        psnrs.append(calc_psnr(out_t, tgt_t))
        direct_psnrs.append(calc_psnr(inp_t, tgt_t))
        
    print(f"\nMode: {mode}")
    print(f"  Direct PSNR: {np.mean(direct_psnrs):.2f} dB")
    print(f"  Model PSNR:  {np.mean(psnrs):.2f} dB")
