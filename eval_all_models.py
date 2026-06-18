import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
try:
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Try to insert current directory to path
sys.path.insert(0, os.path.abspath('.'))

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
        
        # Convert to tensor at original size
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
    # gaussian_weights=True, sigma=1.5, use_sample_covariance=False matches Matlab SSIM
    score = ssim_func(y1, y2, data_range=255.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return score

# Import models
from models.FADformer import FADformer, FADformer_mini, FADformer_HGM, FADformer_HGM_mini

def find_dataset_dir(dataset_name):
    # Try different possible paths for the dataset (both Windows local and Linux server)
    candidates = [
        # Local Windows Baidu Download paths
        os.path.join(r'D:\BaiduNetdiskDownload', dataset_name, dataset_name),
        os.path.join(r'D:\BaiduNetdiskDownload', dataset_name),
        # Relative paths
        os.path.join('.', dataset_name),
        os.path.join('..', dataset_name),
        # Linux Server paths
        os.path.join('/root/HGM_Training_Package', dataset_name),
        os.path.join('/root', dataset_name),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'test', 'input')):
            return os.path.abspath(path)
    return None

def evaluate_single_model(model_name, model_type, ckpt_path, dataset_dir, device):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at: {ckpt_path} -> Skipping {model_name}")
        return None, None
        
    print(f"\nEvaluating {model_name}...")
    print(f" - Checkpoint: {ckpt_path}")
    print(f" - Dataset: {dataset_dir}")
    
    # Initialize model
    if model_type == 'baseline_mini':
        model = FADformer_mini()
    elif model_type == 'hgm_mini':
        model = FADformer_HGM_mini(window_size=8, num_heads=4, fusion_mode='gate')
    elif model_type == 'baseline_full':
        model = FADformer()
    elif model_type == 'hgm_full':
        model = FADformer_HGM(window_size=8, num_heads=4, fusion_mode='gate', use_checkpoint=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Load state dict
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'params' in ckpt:
            state = ckpt['params']
        else:
            state = ckpt
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state[name] = v
        
    model.load_state_dict(new_state)
    model = model.to(device)
    model.eval()
    
    # Setup dataset
    try:
        dataset = RainDatasetOriginalSize(dataset_dir, 'test')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None
        
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    # Pad multiple
    img_multiple_of = 32 if 'hgm' in model_type else 4
    
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
            
            # Progress print
            if count % 20 == 0 or count == len(loader):
                print(f"   -> Processed {count:3d}/{len(loader):3d} images...")
                
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"Result for {model_name}: PSNR = {avg_psnr:.4f} dB, SSIM = {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation of all FADformer models")
    parser.add_argument('--dataset', type=str, default='Rain200H', choices=['Rain200H', 'Rain200L'], help="Dataset to evaluate on")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'baseline_mini', 'hgm_mini', 'baseline_full', 'hgm_full'], help="Model type to evaluate")
    parser.add_argument('--device', type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Device auto-detect
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Evaluation Device: {device}")
    if not HAS_SKIMAGE:
        print("Warning: scikit-image is not installed. SSIM calculation will be skipped (value set to 0.0).")
    
    # Dataset path resolution
    dataset_dir = find_dataset_dir(args.dataset)
    if dataset_dir is None:
        print(f"Error: Could not find dataset {args.dataset} directory in standard candidate paths.")
        sys.exit(1)
    print(f"Found dataset {args.dataset} at: {dataset_dir}")
    
    # Checkpoint paths mapped by dataset and model
    ckpt_mappings = {
        'Rain200H': {
            'baseline_mini': ('FADformer-mini (Baseline)', 'baseline_mini', './saved_models/rain200h_real/FADformer_orig_best.pth'),
            'hgm_mini': ('FADformer-HGM-mini (Ours)', 'hgm_mini', './saved_models/rain200h_real/FADformer_HGM_best_server.pth'),
            'baseline_full': ('FADformer-full (Baseline)', 'baseline_full', './pretrain_weights/rain200H/FADformer_Rain200H.pth'),
            'hgm_full': ('FADformer-HGM-full (Ours)', 'hgm_full', './saved_models/rain200h_real/FADformer_HGM_latest_server_full.pth'),
        },
        'Rain200L': {
            'baseline_mini': ('FADformer-mini (Baseline)', 'baseline_mini', './saved_models/rain200l_real/FADformer_orig_best.pth'),
            # If ours mini is trained on Rain200L, it would be downloaded here:
            'hgm_mini': ('FADformer-HGM-mini (Ours)', 'hgm_mini', './saved_models/rain200l_real/FADformer_HGM_best_server.pth'),
            'baseline_full': ('FADformer-full (Baseline)', 'baseline_full', './pretrain_weights/rain200L/FADformer_Rain200L.pth'),
            # No HGM full on Rain200L trained/planned yet, but map it just in case
            'hgm_full': ('FADformer-HGM-full (Ours)', 'hgm_full', './saved_models/rain200l_real/FADformer_HGM_latest_server_full.pth'),
        }
    }
    
    selected_models = {}
    if args.model == 'all':
        selected_models = ckpt_mappings[args.dataset]
    else:
        selected_models = {args.model: ckpt_mappings[args.dataset][args.model]}
        
    results = {}
    for key, (model_name, model_type, ckpt_path) in selected_models.items():
        # Fallbacks for mini ckpt path if server prefix is missing locally or saved as _mini.pth
        if key == 'hgm_mini' and not os.path.exists(ckpt_path):
            for alt in [ckpt_path.replace('_server', '_mini'), ckpt_path.replace('_server', '')]:
                if os.path.exists(alt):
                    ckpt_path = alt
                    break
                
        # Fallbacks for full ckpt path (best vs latest)
        if key == 'hgm_full' and not os.path.exists(ckpt_path):
            alternative_path = ckpt_path.replace('latest', 'best')
            if os.path.exists(alternative_path):
                ckpt_path = alternative_path
                
        psnr, ssim = evaluate_single_model(model_name, model_type, ckpt_path, dataset_dir, device)
        if psnr is not None:
            results[model_name] = (psnr, ssim)
            
    # Print summary table
    print("\n" + "="*70)
    print(f" Summary of Evaluation on {args.dataset}:")
    print("-"*70)
    print(f" {'Model Name':<35} | {'PSNR (dB)':<12} | {'SSIM':<10}")
    print("-"*70)
    for model_name, (psnr, ssim) in results.items():
        print(f" {model_name:<35} | {psnr:<12.4f} | {ssim:<10.4f}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
