# -*- coding: utf-8 -*-
"""
SS-RF: Spectral-Spatial Rectified Flow Training and Fine-tuning Script
Implements the Mathematically Constrained Flow Restoration Pipeline on Rain200L
"""

import os
import sys
import time
import random
import argparse
import copy
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

if not AMP_NEW:
    _autocast = autocast
    class autocast_compat(_autocast):
        def __init__(self, device_type=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
    autocast = autocast_compat

from models.SS_RF_Model import SS_RF_mini, SS_RF_full

def parse_args():
    parser = argparse.ArgumentParser(description="Train SS-RF on Rain200L")
    
    # Dataset and paths
    parser.add_argument('--data_dir', type=str, default='./Rain200L',
                        help='Path to the Rain200L dataset (default: ./Rain200L)')
    parser.add_argument('--save_dir', type=str, default='./saved_models/rain200l_ss_rf',
                        help='Directory to save checkpoints')
    parser.add_argument('--pretrained', type=str, default='./saved_models/rain200l_real/FADformer_HGM_best_mini.pth',
                        help='Path to pretrained FADformer backbone weights')
    
    # Model configuration
    parser.add_argument('--model_scale', type=str, default='mini', choices=['mini', 'full'],
                        help='Model scale configuration (default: mini)')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of the degradation latent vector z')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size for HGM sparse attention (default: 8)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of heads for HGM sparse attention (default: 4)')
    parser.add_argument('--fusion_mode', type=str, default='gate', choices=['gate', 'sum', 'learnable'],
                        help='Feature fusion mode for HGM (default: gate)')
    
    # Loss Hyper-parameters (SS-RF Constraints)
    parser.add_argument('--lambda1', type=float, default=0.5,
                        help='Weight for spatial flow alignment loss (default: 0.5)')
    parser.add_argument('--lambda2', type=float, default=0.1,
                        help='Weight for spectral flow alignment loss (default: 0.1)')
    parser.add_argument('--lambda3', type=float, default=0.1,
                        help='Weight for prior trajectory invariance loss (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Weight for spectral reconstruction loss (default: 0.1)')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Weight for phase flow difference wrapping factor (default: 0.1)')
    
    # Training hyper-parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size per GPU (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
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

def compute_spec_flow_loss(v_theta, target_u, beta=0.1):
    """
    Computes L1 spectral loss on both Amplitude and phase-wrapped angle domain.
    """
    fft_v = torch.fft.rfft2(v_theta, norm='ortho')
    fft_u = torch.fft.rfft2(target_u, norm='ortho')
    
    amp_v = torch.abs(fft_v)
    amp_u = torch.abs(fft_u)
    loss_amp = torch.mean(torch.abs(amp_v - amp_u))
    
    # Phase difference wrapping onto [-pi, pi] using circular distance
    phase_v = torch.angle(fft_v)
    phase_u = torch.angle(fft_u)
    phase_diff = torch.atan2(torch.sin(phase_v - phase_u), torch.cos(phase_v - phase_u))
    loss_phase = torch.mean(torch.abs(phase_diff))
    
    return loss_amp + beta * loss_phase

def compute_recon_loss(pred_clean, target_clean, gamma=0.1):
    """
    Joint Spatial and Spectral Reconstruction loss.
    """
    loss_spatial = torch.mean(torch.abs(pred_clean - target_clean))
    
    fft_pred = torch.fft.rfft2(pred_clean, norm='ortho')
    fft_target = torch.fft.rfft2(target_clean, norm='ortho')
    loss_spectral = torch.mean(torch.abs(torch.abs(fft_pred) - torch.abs(fft_target)))
    
    return loss_spatial + gamma * loss_spectral

def validate(model, loader, device):
    """
    Validation step using the exact mathematical reparameterization (switch_to_deploy)
    to verify actual inference-time performance.
    """
    fused_model = copy.deepcopy(model)
    fused_model.switch_to_deploy()
    fused_model.eval()
    
    total_psnr = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            with autocast(enabled=torch.cuda.is_available()):
                output = fused_model(source).clamp_(0, 1)
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
        state = ckpt.get('params', ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))
    else:
        state = ckpt

    new_state = {}
    model_state = model.state_dict()
    mismatched_keys = []
    
    for k, v in state.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        if not name.startswith('backbone.') and not name.startswith('deg_encoder') and not name.startswith('sdm'):
            mapped_name = f'backbone.{name}'
        else:
            mapped_name = name
            
        if mapped_name in model_state:
            if model_state[mapped_name].shape == v.shape:
                new_state[mapped_name] = v
            else:
                mismatched_keys.append((mapped_name, list(v.shape), list(model_state[mapped_name].shape)))

    if mismatched_keys:
        logger(f"Shape Mismatch: Filtered out {len(mismatched_keys)} keys due to shape difference")

    result = model.load_state_dict(new_state, strict=False)
    missing = [k for k in result.missing_keys if not any(x in k.lower() for x in ['hgm', 'sparse', 'window', 'deg_encoder', 'sdm', 'gate'])]
    if missing:
        logger(f"Warning: missing backbone keys in checkpoint: {missing[:5]}...")
        
    logger(f"Successfully loaded {len(new_state)}/{len(model_state)} parameters from pretrained weights")
    return model

def print_reparam_stats(model, logger):
    """Prints size statistics before and after structural reparameterization"""
    params_train = sum(p.numel() for p in model.parameters())
    
    model_copy = copy.deepcopy(model)
    model_copy.switch_to_deploy()
    params_deploy = sum(p.numel() for p in model_copy.parameters())
    
    logger(f"Reparameterization Statistics:")
    logger(f"  - Training (multi-branch) params: {params_train/1e6:.4f}M")
    logger(f"  - Deployment (fused) params:     {params_deploy/1e6:.4f}M")
    logger(f"  - Parameter reduction:            {(params_train - params_deploy)/1e3:.2f}K ({100*(params_train - params_deploy)/params_train:.2f}%)")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    progress_path = os.path.join(args.save_dir, 'progress.txt')
    log_path = os.path.join(args.save_dir, 'SS_RF_train_log.txt')
    
    def write_progress(msg):
        print(msg)
        with open(progress_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
            f.flush()
            os.fsync(f.fileno())

    def write_log(msg):
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
            f.flush()
            os.fsync(f.fileno())

    with open(progress_path, 'w', encoding='utf-8') as f:
        f.write("")

    write_progress(f"=== SS-RF (Spectral-Spatial Rectified Flow) Training ===")
    write_progress(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_progress(f"Device: {device}")
    
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
        return

    # 2. Instantiate Target Model
    write_progress(f"Configuring SS-RF model (scale={args.model_scale}, latent_dim={args.latent_dim})...")
    if args.model_scale == 'mini':
        model = SS_RF_mini(latent_dim=args.latent_dim, window_size=args.window_size, num_heads=args.num_heads, fusion_mode=args.fusion_mode)
    else:
        model = SS_RF_full(latent_dim=args.latent_dim, window_size=args.window_size, num_heads=args.num_heads, fusion_mode=args.fusion_mode)
    
    print_reparam_stats(model, write_progress)

    baseline_psnr = 0.0
    latest_ckpt_path = os.path.join(args.save_dir, f'SS_RF_latest_{args.model_scale}.pth')
    resume_training = not args.no_resume
    
    has_checkpoint = False
    if resume_training and os.path.exists(latest_ckpt_path):
        try:
            ckpt = torch.load(latest_ckpt_path, map_location='cpu')
            if 'baseline_psnr' in ckpt:
                baseline_psnr = ckpt['baseline_psnr']
                has_checkpoint = True
        except Exception as e:
            pass

    if not has_checkpoint:
        if not args.from_scratch:
            write_progress("Initializing target model backbone with pretrained weights...")
            model = load_pretrained(model, args.pretrained, write_progress)
        else:
            write_progress("Training from scratch. Baseline PSNR set to 0.0.")
            baseline_psnr = 0.0

    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    WARMUP_EPOCHS = 3
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - WARMUP_EPOCHS, eta_min=1e-6)
    scaler = GradScaler('cuda') if AMP_NEW and torch.cuda.is_available() else None

    best_psnr = 0.0
    start_epoch = 0
    train_start = time.time()

    if resume_training and os.path.exists(latest_ckpt_path):
        try:
            write_progress(f"Resuming training from latest checkpoint: {latest_ckpt_path}...")
            ckpt = torch.load(latest_ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
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
            f.write(f"Rain200L SS-RF (Rectified Flow) Training Log\n")
            f.write(f"Configuration: lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}\n\n")

    total_batches = len(train_loader)
    write_progress(f"Starting 3-stage training loop: {args.epochs} epochs, {total_batches} batches/epoch")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        epoch_start = time.time()

        # 3-Stage Training Logic for Freezing/Unfreezing Parameters
        stage_desc = ""
        if epoch <= 10:
            stage_desc = "[Stage 1: Train Prior Encoder (Rep-DE) + Modulator (SDM) (Backbone Frozen)]"
            for name, param in model.named_parameters():
                if 'deg_encoder' in name or 'sdm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif epoch <= 20:
            stage_desc = "[Stage 2: Unfreeze Layer 3/5 (Deep Modulation Stages)]"
            for name, param in model.named_parameters():
                if 'deg_encoder' in name or 'sdm' in name or 'layer3' in name or 'layer5' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            stage_desc = "[Stage 3: Full Unfreeze + Dual-Spectral Flow Matching Constraints]"
            for param in model.parameters():
                param.requires_grad = True

        write_progress(f"--- Epoch {epoch} {stage_desc} ---")

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
            B = source.size(0)

            # Sample trajectory factors t1, t2 ~ U(0,1) for trajectory flow constraints
            t1 = torch.rand(B, 1, 1, 1, device=device)
            t2 = torch.rand(B, 1, 1, 1, device=device)
            
            # Construct probability flow intermediate points
            X_t1 = (1.0 - t1) * source + t1 * target
            X_t2 = (1.0 - t2) * source + t2 * target
            target_u = target - source  # Constant drift vector field

            if scaler is not None:
                with autocast(enabled=torch.cuda.is_available()):
                    # Forward intermediate steps
                    v_theta_t1, z_t1 = model(X_t1, return_z=True)
                    v_theta_t2, z_t2 = model(X_t2, return_z=True)
                    
                    # Compute constraints
                    loss_flow_align = nn.L1Loss()(v_theta_t1, target_u)
                    loss_spec_flow = compute_spec_flow_loss(v_theta_t1, target_u, beta=args.beta)
                    loss_prior_inv = nn.MSELoss()(z_t1, z_t2)
                    
                    # Absolute endpoint reconstruction (t=0 prediction)
                    pred_clean = model(source)
                    loss_recon = compute_recon_loss(pred_clean, target, gamma=args.gamma)
                    
                    # Combined Loss
                    loss = (loss_recon + 
                            args.lambda1 * loss_flow_align + 
                            args.lambda2 * loss_spec_flow + 
                            args.lambda3 * loss_prior_inv) / args.accumulation
            else:
                v_theta_t1, z_t1 = model(X_t1, return_z=True)
                v_theta_t2, z_t2 = model(X_t2, return_z=True)
                
                loss_flow_align = nn.L1Loss()(v_theta_t1, target_u)
                loss_spec_flow = compute_spec_flow_loss(v_theta_t1, target_u, beta=args.beta)
                loss_prior_inv = nn.MSELoss()(z_t1, z_t2)
                
                pred_clean = model(source)
                loss_recon = compute_recon_loss(pred_clean, target, gamma=args.gamma)
                
                loss = (loss_recon + 
                        args.lambda1 * loss_flow_align + 
                        args.lambda2 * loss_spec_flow + 
                        args.lambda3 * loss_prior_inv) / args.accumulation

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.accumulation == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * args.accumulation

            if (batch_idx + 1) % 50 == 0:
                elapsed_b = time.time() - epoch_start
                write_progress(f"E{epoch} batch {batch_idx+1}/{total_batches} loss={loss.item()*args.accumulation:.4f} {elapsed_b:.0f}s")

        if epoch > WARMUP_EPOCHS or start_epoch > 0:
            scheduler.step()
            
        avg_loss = epoch_loss / total_batches
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        write_progress(f"E{epoch} completed: loss={avg_loss:.4f} lr={current_lr:.1e} time={epoch_time:.0f}s")

        # Validation step
        if epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs:
            write_progress(f"Validating epoch {epoch} (using reparameterized fused model)...")
            val_psnr = validate(model, test_loader, device)
            improved = ""
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'psnr': best_psnr,
                    'loss': avg_loss,
                    'baseline_psnr': baseline_psnr,
                }, os.path.join(args.save_dir, f'SS_RF_best_{args.model_scale}.pth'))
                improved = " *BEST*"

            elapsed = time.time() - train_start
            mem = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
            msg = (f"SS-RF E{epoch:3d}/{args.epochs} | Loss:{avg_loss:.4f} | "
                   f"PSNR:{val_psnr:.2f} (Best:{best_psnr:.2f}) | "
                   f"LR:{current_lr:.1e} | {elapsed/60:.1f}min | GPU:{mem:.2f}GB{improved}")
            write_progress(msg)
            write_log(msg)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'psnr': best_psnr,
            'loss': avg_loss,
            'baseline_psnr': baseline_psnr,
        }, os.path.join(args.save_dir, f'SS_RF_latest_{args.model_scale}.pth'))

    total_time = time.time() - train_start
    write_progress(f"SS-RF training completed! Time: {total_time/60:.1f}min | Best PSNR: {best_psnr:.2f} dB")
    write_progress("ALL DONE!")

if __name__ == '__main__':
    main()
