# -*- coding: utf-8 -*-
"""
FADformer_HGM Rain200L Training and Fine-tuning Script
Supports training from scratch and fine-tuning with pretrained baseline weights.
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Setup paths (add workspace directory to path)
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

try:
    from torch.amp import autocast, GradScaler
    AMP_NEW = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_NEW = False

# Wrap autocast for backward compatibility in PyTorch versions (e.g. 2.0.1 on server)
# where torch.cuda.amp.autocast is used and doesn't accept device_type as first positional arg
if not AMP_NEW:
    _autocast = autocast
    class autocast_compat(_autocast):
        def __init__(self, device_type=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
    autocast = autocast_compat

from models.FADformer import FADformer, FADformer_mini, FADformer_HGM, FADformer_HGM_mini

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Fine-tune FADformer / FADformer_HGM on Rain200L")
    
    # Dataset and paths
    parser.add_argument('--data_dir', type=str, default='./Rain200L',
                        help='Path to the Rain200L dataset (default: ./Rain200L)')
    parser.add_argument('--save_dir', type=str, default='./saved_models/rain200l_real',
                        help='Directory to save checkpoints (default: ./saved_models/rain200l_real)')
    parser.add_argument('--pretrained', type=str, default='./pretrain_weights/rain200L/FADformer_Rain200L.pth',
                        help='Path to pretrained FADformer weights (default: ./pretrain_weights/rain200L/FADformer_Rain200L.pth)')
    
    # Model configuration
    parser.add_argument('--model_scale', type=str, default='mini', choices=['mini', 'full'],
                        help='Model scale configuration (default: mini)')
    parser.add_argument('--use_hgm', action='store_true', default=True,
                        help='Use Hybrid Global Mixer (HGM) architecture (default: True)')
    parser.add_argument('--no_hgm', action='store_false', dest='use_hgm',
                        help='Disable HGM, use original baseline backbone')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size for HGM sparse attention (default: 8)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of heads for HGM sparse attention (default: 4)')
    parser.add_argument('--fusion_mode', type=str, default='gate', choices=['gate', 'sum', 'learnable'],
                        help='Feature fusion mode for HGM (default: gate)')
    
    # Training hyper-parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total training epochs (default: 100 for fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size per GPU (default: 2 to prevent OOM)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5, recommended for fine-tuning)')
    parser.add_argument('--accumulation', type=int, default=8,
                        help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Cropped image size for training (default: 256)')
    
    # Execution modes
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train from scratch instead of loading pretrained weights')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not resume training from the latest checkpoint if it exists')
    parser.add_argument('--seed', type=int, default=8001,
                        help='Random seed (default: 8001)')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Evaluation frequency in epochs (default: 10)')
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RainDataset(Dataset):
    def __init__(self, root, split='train', img_size=256):
        self.input_dir = os.path.join(root, split, 'input')
        self.target_dir = os.path.join(root, split, 'target')
        self.img_size = img_size
        
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
            
        self.filenames = sorted([f for f in os.listdir(self.input_dir) if f.endswith(('.png', '.jpg', '.bmp'))])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        inp = Image.open(os.path.join(self.input_dir, fname)).convert('RGB')
        tgt = Image.open(os.path.join(self.target_dir, fname)).convert('RGB')
        
        # Random Crop for training, Center Crop/Resize for testing
        w, h = inp.size
        if w < self.img_size or h < self.img_size:
            scale = max(self.img_size / w, self.img_size / h)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            inp = inp.resize((new_w, new_h), Image.BICUBIC)
            tgt = tgt.resize((new_w, new_h), Image.BICUBIC)
            w, h = inp.size
            
        x = random.randint(0, w - self.img_size)
        y = random.randint(0, h - self.img_size)
        
        inp = inp.crop((x, y, x + self.img_size, y + self.img_size))
        tgt = tgt.crop((x, y, x + self.img_size, y + self.img_size))
        
        inp = torch.from_numpy(np.array(inp).astype(np.float32).transpose(2, 0, 1) / 255.0)
        tgt = torch.from_numpy(np.array(tgt).astype(np.float32).transpose(2, 0, 1) / 255.0)
        return {'source': inp, 'target': tgt, 'filename': fname}

def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()

def validate(model, loader, device):
    model.eval()
    total_psnr = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            with autocast(enabled=torch.cuda.is_available()):
                output = model(source).clamp_(0, 1)
            total_psnr += calc_psnr(output, target)
            count += 1
    return total_psnr / count if count > 0 else 0

def load_pretrained(model, path, logger):
    logger(f"Loading weights from checkpoint: {path}")
    if not os.path.exists(path):
        logger(f"Warning: checkpoint path {path} does not exist. Model will start from scratch!")
        return model

    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'params' in ckpt:
            state = ckpt['params']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
    else:
        state = ckpt

    new_state = {}
    model_state = model.state_dict()
    mismatched_keys = []
    
    for k, v in state.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        if name in model_state:
            # Check shape compatibility
            if model_state[name].shape == v.shape:
                new_state[name] = v
            else:
                mismatched_keys.append((name, list(v.shape), list(model_state[name].shape)))
        else:
            # Try matching without prefix if name doesn't match directly
            new_state[name] = v

    if mismatched_keys:
        logger(f"Shape Mismatch: Filtered out {len(mismatched_keys)} keys due to shape difference (e.g. {mismatched_keys[:2]})")

    result = model.load_state_dict(new_state, strict=False)
    
    # Filter missing keys to only report structural backbone ones
    missing = [k for k in result.missing_keys if not any(x in k.lower() for x in ['hgm', 'sparse', 'window', 'gate', 'relative_position', 'ca_conv'])]
    unexpected = result.unexpected_keys
    
    if missing:
        logger(f"Warning: missing backbone keys in checkpoint: {missing[:5]}...")
    if unexpected:
        logger(f"Warning: unexpected keys in checkpoint: {unexpected[:5]}...")
        
    logger(f"Successfully loaded {len(new_state)}/{len(model_state)} parameters from pretrained weights (strict=False)")
    return model

def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    progress_path = os.path.join(args.save_dir, 'progress.txt')
    log_path = os.path.join(args.save_dir, 'HGM_train_log.txt')
    
    def write_progress(msg):
        print(msg)
        with open(progress_path, 'w', encoding='utf-8') as f:
            f.write(msg + '\n')
            f.flush()
            os.fsync(f.fileno())

    def write_log(msg):
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
            f.flush()
            os.fsync(f.fileno())

    write_progress(f"=== FADformer HGM Rain200L Training Script ===")
    write_progress(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_progress(f"Device: {device}")
    if torch.cuda.is_available():
        write_progress(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)")
    
    # 1. Dataset Loading
    write_progress(f"Loading Rain200L dataset from {args.data_dir}...")
    try:
        train_dataset = RainDataset(args.data_dir, 'train', args.img_size)
        test_dataset = RainDataset(args.data_dir, 'test', args.img_size)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=0, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                 num_workers=0, pin_memory=True)
        write_progress(f"Dataset loaded: {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    except Exception as e:
        write_progress(f"ERROR loading dataset: {e}")
        write_progress("Please ensure --data_dir matches the path to the Rain200L dataset.")
        return

    # 2. Instantiate and Setup Models
    write_progress(f"Configuring models (scale={args.model_scale}, use_hgm={args.use_hgm})...")
    
    # Base Original Model (for baseline comparison)
    if args.model_scale == 'mini':
        model_orig = FADformer_mini()
    else:
        model_orig = FADformer()
        
    params_orig = sum(p.numel() for p in model_orig.parameters())
    write_progress(f"Baseline FADformer size: {params_orig/1e6:.2f}M params")
    
    # HGM / Target model
    if args.use_hgm:
        if args.model_scale == 'mini':
            model_hgm = FADformer_HGM_mini(window_size=args.window_size, num_heads=args.num_heads, fusion_mode=args.fusion_mode)
        else:
            model_hgm = FADformer_HGM(window_size=args.window_size, num_heads=args.num_heads, fusion_mode=args.fusion_mode)
    else:
        if args.model_scale == 'mini':
            model_hgm = FADformer_mini()
        else:
            model_hgm = FADformer()
            
    params_hgm = sum(p.numel() for p in model_hgm.parameters())
    write_progress(f"Target HGM model size: {params_hgm/1e6:.2f}M params")

    # 3. Evaluate baseline (unless checkpoint already has it)
    baseline_psnr = 0.0
    latest_ckpt_path = os.path.join(args.save_dir, f'FADformer_HGM_latest_{args.model_scale}.pth')
    resume_training = not args.no_resume
    
    has_checkpoint = False
    if resume_training and os.path.exists(latest_ckpt_path):
        try:
            ckpt = torch.load(latest_ckpt_path, map_location='cpu')
            if 'baseline_psnr' in ckpt:
                baseline_psnr = ckpt['baseline_psnr']
                has_checkpoint = True
                write_progress(f"Found latest checkpoint. Loaded baseline PSNR: {baseline_psnr:.2f} dB")
        except Exception as e:
            write_progress(f"Failed to read checkpoint for baseline score: {e}")

    if not has_checkpoint:
        if not args.from_scratch:
            # Determine correct baseline weights path based on model scale
            if args.model_scale == 'mini':
                baseline_path = './saved_models/rain200l_real/FADformer_orig_best.pth'
            else:
                baseline_path = './pretrain_weights/rain200L/FADformer_Rain200L.pth'
                
            if os.path.exists(baseline_path):
                write_progress(f"Evaluating baseline model (Original FADformer pretrained from {baseline_path})...")
                model_orig = load_pretrained(model_orig, baseline_path, write_progress)
                model_orig = model_orig.to(device)
                baseline_psnr = validate(model_orig, test_loader, device)
                write_progress(f"Pretrained FADformer baseline PSNR on Rain200L: {baseline_psnr:.2f} dB")
            else:
                write_progress(f"Baseline checkpoint {baseline_path} not found. Skipping baseline evaluation.")
                # Fallback to hardcoded reference values if files are not present
                baseline_psnr = 36.61 if args.model_scale == 'mini' else 41.69
                write_progress(f"Using reference baseline PSNR: {baseline_psnr:.2f} dB")
                
            del model_orig
            torch.cuda.empty_cache()
        else:
            write_progress("Training from scratch. Baseline PSNR set to 0.0.")
            baseline_psnr = 0.0

    # 4. Load Pretrained weights into target model if fine-tuning
    if not args.from_scratch and not has_checkpoint:
        write_progress("Initializing target model with pretrained weights for fine-tuning...")
        model_hgm = load_pretrained(model_hgm, args.pretrained, write_progress)

    model_hgm = model_hgm.to(device)
    
    # Setup optimizer, scheduler, criterion, scaler
    optimizer = optim.AdamW(model_hgm.parameters(), lr=args.lr, weight_decay=1e-2)
    WARMUP_EPOCHS = 3
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - WARMUP_EPOCHS, eta_min=1e-6)
    criterion = nn.L1Loss()
    scaler = GradScaler('cuda') if AMP_NEW and torch.cuda.is_available() else None

    best_psnr = 0.0
    start_epoch = 0
    train_start = time.time()

    # 5. Resume from checkpoint if exists
    if resume_training and os.path.exists(latest_ckpt_path):
        try:
            write_progress(f"Resuming training from latest checkpoint: {latest_ckpt_path}...")
            ckpt = torch.load(latest_ckpt_path, map_location='cpu')
            model_hgm.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            best_psnr = ckpt.get('psnr', 0.0)
            baseline_psnr = ckpt.get('baseline_psnr', baseline_psnr)
            write_progress(f"Resumed from epoch {start_epoch} with best validation PSNR: {best_psnr:.2f} dB")
        except Exception as e:
            write_progress(f"Failed to resume from checkpoint: {e}. Starting from epoch 0.")

    if start_epoch == 0:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Rain200L HGM Training Log\n")
            f.write(f"Baseline PSNR: {baseline_psnr:.2f} dB\n")
            f.write(f"Model params: {params_hgm/1e6:.2f}M\n")
            f.write(f"Fine-tuning Configuration: lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}\n\n")
    else:
        write_log(f"\n--- Resumed training from epoch {start_epoch} ---")

    # 6. Training Loop
    total_batches = len(train_loader)
    write_progress(f"Starting training loop: {args.epochs} epochs, {total_batches} batches/epoch")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model_hgm.train()
        epoch_loss = 0
        optimizer.zero_grad()
        epoch_start = time.time()

        # Warmup for first WARMUP_EPOCHS epochs if starting from epoch 0
        if epoch <= WARMUP_EPOCHS and start_epoch == 0:
            warmup_lr = (epoch / WARMUP_EPOCHS) * args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        for batch_idx, batch in enumerate(train_loader):
            source = batch['source'].to(device)
            target = batch['target'].to(device)

            if scaler is not None:
                with autocast(enabled=torch.cuda.is_available()):
                    output = model_hgm(source)
                    loss = criterion(output, target) / args.accumulation
            else:
                output = model_hgm(source)
                loss = criterion(output, target) / args.accumulation

            if torch.isnan(loss):
                write_progress(f"Warning: NaN loss detected at E{epoch} batch {batch_idx+1}. Skipping batch.")
                optimizer.zero_grad()
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.accumulation == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_hgm.parameters(), 0.01)
                    
                    # Detect NaN gradient to protect weights
                    has_nan_grad = False
                    for param in model_hgm.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        write_progress(f"Warning: NaN gradient detected at E{epoch} batch {batch_idx+1}. Skipping step.")
                        optimizer.zero_grad()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(model_hgm.parameters(), 0.01)
                    has_nan_grad = False
                    for param in model_hgm.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            break
                    if has_nan_grad:
                        write_progress(f"Warning: NaN gradient detected at E{epoch} batch {batch_idx+1}. Skipping step.")
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

            epoch_loss += loss.item() * args.accumulation

            if (batch_idx + 1) % 50 == 0:
                elapsed_b = time.time() - epoch_start
                write_progress(f"E{epoch} batch {batch_idx+1}/{total_batches} loss={loss.item()*args.accumulation:.4f} {elapsed_b:.0f}s")

        # Step scheduler
        if epoch > WARMUP_EPOCHS or start_epoch > 0:
            scheduler.step()
            
        avg_loss = epoch_loss / total_batches
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        write_progress(f"E{epoch} completed: loss={avg_loss:.4f} lr={current_lr:.1e} time={epoch_time:.0f}s")

        # Validation and saving checkpoints
        if epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs:
            write_progress(f"Validating epoch {epoch}...")
            val_psnr = validate(model_hgm, test_loader, device)
            improved = ""
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_hgm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'psnr': best_psnr,
                    'loss': avg_loss,
                    'baseline_psnr': baseline_psnr,
                }, os.path.join(args.save_dir, f'FADformer_HGM_best_{args.model_scale}.pth'))
                improved = " *BEST*"

            elapsed = time.time() - train_start
            mem = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
            delta = val_psnr - baseline_psnr
            msg = (f"HGM E{epoch:3d}/{args.epochs} | Loss:{avg_loss:.4f} | "
                   f"PSNR:{val_psnr:.2f} (Best:{best_psnr:.2f}, Base:{baseline_psnr:.2f}, dP:{delta:+.2f}) | "
                   f"LR:{current_lr:.1e} | {elapsed/60:.1f}min | GPU:{mem:.2f}GB{improved}")
            write_progress(msg)
            write_log(msg)

        # Save latest checkpoint for resumption at the end of each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_hgm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'psnr': best_psnr,
            'loss': avg_loss,
            'baseline_psnr': baseline_psnr,
        }, os.path.join(args.save_dir, f'FADformer_HGM_latest_{args.model_scale}.pth'))

        # Periodically save epochs
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_hgm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'psnr': best_psnr,
                'loss': avg_loss,
                'baseline_psnr': baseline_psnr,
            }, os.path.join(args.save_dir, f'FADformer_HGM_ep{epoch}_{args.model_scale}.pth'))

    total_time = time.time() - train_start
    write_progress(f"HGM training completed! Time: {total_time/60:.1f}min | Best PSNR: {best_psnr:.2f} dB")

    # 7. Final Evaluation and Comparison
    write_progress("Starting final comparison...")
    
    # Reload original baseline
    if args.model_scale == 'mini':
        model_orig = FADformer_mini()
    else:
        model_orig = FADformer()
    if not args.from_scratch:
        model_orig = load_pretrained(model_orig, args.pretrained, write_progress)
    model_orig = model_orig.to(device)
    model_orig.eval()

    # Load best trained HGM model
    ckpt_hgm = torch.load(os.path.join(args.save_dir, f'FADformer_HGM_best_{args.model_scale}.pth'), map_location=device)
    model_hgm.load_state_dict(ckpt_hgm['model_state_dict'])
    model_hgm.eval()

    orig_psnrs, hgm_psnrs = [], []
    orig_times, hgm_times = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            source = batch['source'].to(device)
            target = batch['target'].to(device)

            t0 = time.time()
            with autocast(enabled=torch.cuda.is_available()):
                out_orig = model_orig(source).clamp_(0, 1)
            orig_times.append(time.time() - t0)

            t0 = time.time()
            with autocast(enabled=torch.cuda.is_available()):
                out_hgm = model_hgm(source).clamp_(0, 1)
            hgm_times.append(time.time() - t0)

            orig_psnrs.append(calc_psnr(out_orig, target))
            hgm_psnrs.append(calc_psnr(out_hgm, target))

            if (i + 1) % 50 == 0:
                write_progress(f"Compared {i+1}/{len(test_loader)} images")

    avg_psnr_orig = np.mean(orig_psnrs)
    avg_psnr_hgm = np.mean(hgm_psnrs)
    delta_psnr = avg_psnr_hgm - avg_psnr_orig
    avg_time_orig = np.mean(orig_times) * 1000
    avg_time_hgm = np.mean(hgm_times) * 1000

    report = (
        f"\n{'Metric':<25} {'Original':>12} {'HGM':>12} {'Delta':>12}\n"
        f"{'-'*65}\n"
        f"{'Parameters (M)':<25} {params_orig/1e6:>12.2f} {params_hgm/1e6:>12.2f} {(params_hgm-params_orig)/params_orig*100:>+11.1f}%\n"
        f"{'Test PSNR (dB)':<25} {avg_psnr_orig:>12.2f} {avg_psnr_hgm:>12.2f} {delta_psnr:>+11.2f}\n"
        f"{'Inference Time (ms)':<25} {avg_time_orig:>12.1f} {avg_time_hgm:>12.1f} {(avg_time_hgm/avg_time_orig-1)*100:>+11.1f}%\n"
    )
    write_progress(report)

    if delta_psnr > 0:
        write_progress(f"HGM PSNR improvement on Rain200L: +{delta_psnr:.2f} dB")
    else:
        write_progress(f"HGM PSNR change: {delta_psnr:+.2f} dB")

    report_path = os.path.join(args.save_dir, 'comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Rain200L Training Comparison Report\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: Rain200L ({len(train_dataset)} train, {len(test_dataset)} test)\n")
        f.write(f"Baseline: pretrained FADformer (PSNR={baseline_psnr:.2f} dB)\n")
        f.write(f"HGM: trained {args.epochs} epochs\n\n")
        f.write(report)

    write_progress(f"Report saved: {report_path}")
    write_progress("ALL DONE!")

if __name__ == '__main__':
    main()
