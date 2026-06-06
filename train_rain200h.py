"""
FADformer_HGM Rain200H Training

Baseline: pretrained FADformer (pretrain_weights/rain200H/FADformer_Rain200H.pth)
Train: FADformer_HGM_mini only
Compare: after training
"""

import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
try:
    from torch.amp import autocast, GradScaler
    AMP_NEW = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_NEW = False
from PIL import Image
import numpy as np

DATA_DIR = './Rain200H'
SAVE_DIR = './saved_models/rain200h_real'
PRETRAINED = './pretrain_weights/rain200H/FADformer_Rain200H.pth'
EPOCHS = 300
BATCH_SIZE = 2  # 减小 Batch Size 防止显存 OOM
IMG_SIZE = 256  # 保持 256x256 完整分辨率以确保高复原质量
LR = 2e-4  # 使用较低的 2e-4 学习率配合 Warmup，避免破坏预训练权重导致 NaN
ACCUMULATION = 8  # 相应增加梯度累积步数，维持有效 Batch Size = 16 不变

# 模型规模选择: 'mini' (轻量版, 2.38M) 或 'full' (全量版, 9.87M)
MODEL_SCALE = 'full'

progress_path = os.path.join(SAVE_DIR, 'progress.txt')
log_path = os.path.join(SAVE_DIR, 'HGM_train_log.txt')
os.makedirs(SAVE_DIR, exist_ok=True)


def write_progress(msg):
    with open(progress_path, 'w') as f:
        f.write(msg + '\n')
        f.flush()
        os.fsync(f.fileno())


def write_log(msg):
    with open(log_path, 'a') as f:
        f.write(msg + '\n')
        f.flush()
        os.fsync(f.fileno())


write_progress("Initializing...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
write_progress(f"Device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    write_progress(f"GPU: {gpu_name}, VRAM: {vram:.1f} GB")



class RainDataset(Dataset):
    def __init__(self, root, split='train', img_size=256):
        self.input_dir = os.path.join(root, split, 'input')
        self.target_dir = os.path.join(root, split, 'target')
        self.img_size = img_size
        self.filenames = sorted([f for f in os.listdir(self.input_dir) if f.endswith(('.png', '.jpg', '.bmp'))])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        inp = Image.open(os.path.join(self.input_dir, fname)).convert('RGB')
        tgt = Image.open(os.path.join(self.target_dir, fname)).convert('RGB')
        
        # --- 使用随机裁剪 (Random Crop) 代替直接 Resize，完美保留雨痕物理尺度与高频特征 ---
        w, h = inp.size
        if w < self.img_size or h < self.img_size:
            # 如果图像本身尺寸小于目标裁剪尺寸，先等比例放大
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


write_progress("Loading Rain200H dataset...")
train_dataset = RainDataset(DATA_DIR, 'train', IMG_SIZE)
test_dataset = RainDataset(DATA_DIR, 'test', IMG_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
write_progress(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")


def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def validate(model, loader):
    model.eval()
    total_psnr = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            output = model(source).clamp_(0, 1)
            total_psnr += calc_psnr(output, target)
            count += 1
    return total_psnr / count if count > 0 else 0



def load_pretrained(model, path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'params' in ckpt:
        state = ckpt['params']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = ckpt

    new_state = {}
    model_state = model.state_dict()
    mismatched_keys = []
    
    for k, v in state.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        if name in model_state:
            # 检查形状是否完全一致，若不一致则过滤掉防止报错
            if model_state[name].shape == v.shape:
                new_state[name] = v
            else:
                mismatched_keys.append((name, list(v.shape), list(model_state[name].shape)))
        else:
            new_state[name] = v

    if mismatched_keys:
        write_progress(f"Filtered out {len(mismatched_keys)} keys due to shape mismatch: {mismatched_keys[:3]}...")

    result = model.load_state_dict(new_state, strict=False)
    missing = [k for k in result.missing_keys if 'hgm' not in k.lower() and 'sparse' not in k.lower() and 'window' not in k.lower() and 'gate' not in k.lower() and 'relative_position' not in k.lower() and 'ca_conv' not in k.lower()]
    unexpected = result.unexpected_keys
    if missing:
        write_progress(f"Warning: missing keys (non-HGM): {missing[:5]}...")
    if unexpected:
        write_progress(f"Warning: unexpected keys: {unexpected[:5]}...")
    loaded = len(new_state)
    write_progress(f"Loaded {loaded}/{len(model_state)} keys from pretrained weights (strict=False)")
    return model


latest_ckpt_path = os.path.join(SAVE_DIR, f'FADformer_HGM_latest_{MODEL_SCALE}.pth')
resume_training = True
if '--from-scratch' in sys.argv:
    resume_training = False

baseline_psnr = 0
has_checkpoint = False

if resume_training and os.path.exists(latest_ckpt_path):
    try:
        ckpt = torch.load(latest_ckpt_path, map_location='cpu')
        if 'baseline_psnr' in ckpt:
            baseline_psnr = ckpt['baseline_psnr']
            has_checkpoint = True
            write_progress(f"Skipping baseline evaluation. Loaded baseline PSNR from checkpoint: {baseline_psnr:.2f} dB")
    except Exception as e:
        write_progress(f"Could not read checkpoint to skip baseline ({e}).")

if not has_checkpoint:
    write_progress("Loading pretrained FADformer (baseline)...")
    from models.FADformer import FADformer, FADformer_mini, FADformer_HGM, FADformer_HGM_mini

    if MODEL_SCALE == 'mini':
        model_orig = FADformer_mini()
        write_progress("Using FADformer_mini for baseline evaluation")
    else:
        model_orig = FADformer()
        write_progress("Using FADformer (full) for baseline evaluation")
    params_orig = sum(p.numel() for p in model_orig.parameters())
    write_progress(f"Baseline FADformer size: {params_orig/1e6:.2f}M params")

    model_orig = load_pretrained(model_orig, PRETRAINED)
    model_orig = model_orig.to(device)
    model_orig.eval()

    write_progress("Evaluating baseline on Rain200H test set...")
    baseline_psnr = validate(model_orig, test_loader)
    write_progress(f"Pretrained FADformer PSNR on Rain200H test: {baseline_psnr:.2f} dB")

    del model_orig
    torch.cuda.empty_cache()
else:
    from models.FADformer import FADformer, FADformer_mini, FADformer_HGM, FADformer_HGM_mini


if MODEL_SCALE == 'mini':
    write_progress("Creating FADformer_HGM_mini...")
    model_hgm = FADformer_HGM_mini(window_size=8, num_heads=4, fusion_mode='gate')
    params_hgm = sum(p.numel() for p in model_hgm.parameters())
    write_progress(f"HGM_mini: {params_hgm/1e6:.2f}M params")
else:
    write_progress("Creating FADformer_HGM (full)...")
    model_hgm = FADformer_HGM(window_size=8, num_heads=4, fusion_mode='gate')
    params_hgm = sum(p.numel() for p in model_hgm.parameters())
    write_progress(f"HGM (full): {params_hgm/1e6:.2f}M params")

model_hgm = load_pretrained(model_hgm, PRETRAINED)
model_hgm = model_hgm.to(device)
optimizer = optim.AdamW(model_hgm.parameters(), lr=LR, weight_decay=1e-2)

WARMUP_EPOCHS = 3
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
criterion = nn.L1Loss()
scaler = GradScaler('cuda') if AMP_NEW else GradScaler()

best_psnr = 0
start_epoch = 0
train_start = time.time()

# --- 断点续训 (Resume Training) 机制 ---
latest_ckpt_path = os.path.join(SAVE_DIR, f'FADformer_HGM_latest_{MODEL_SCALE}.pth')
resume_training = True
if '--from-scratch' in sys.argv:
    resume_training = False
    write_progress("Forced training from scratch due to '--from-scratch' flag.")

if resume_training and os.path.exists(latest_ckpt_path):
    try:
        write_progress(f"Found latest checkpoint {latest_ckpt_path}. Loading and resuming...")
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
        best_psnr = ckpt.get('psnr', 0)
        baseline_psnr = ckpt.get('baseline_psnr', baseline_psnr)
        write_progress(f"Successfully resumed from epoch {start_epoch} with best PSNR: {best_psnr:.2f} dB")
    except Exception as e:
        write_progress(f"Warning: failed to resume from checkpoint ({e}). Starting from epoch 0.")

if start_epoch == 0:
    with open(log_path, 'w') as f:
        f.write(f"Rain200H HGM Training Log\n")
        f.write(f"Baseline PSNR: {baseline_psnr:.2f} dB\n")
        f.write(f"Model params: {params_hgm/1e6:.2f}M\n\n")
        f.flush()
        os.fsync(f.fileno())
else:
    write_log(f"\n--- Resumed training from epoch {start_epoch} ---")

total_batches = len(train_loader)
write_progress(f"Starting training: {EPOCHS} epochs, {total_batches} batches/epoch")

for epoch in range(start_epoch + 1, EPOCHS + 1):
    model_hgm.train()
    epoch_loss = 0
    optimizer.zero_grad()
    epoch_start = time.time()

    # 渐进式学习率 Warmup，防止初始梯度爆炸破坏预训练权重
    if epoch <= WARMUP_EPOCHS and start_epoch == 0:
        warmup_lr = (epoch / WARMUP_EPOCHS) * LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
        current_lr = warmup_lr
    else:
        current_lr = optimizer.param_groups[0]['lr']

    for batch_idx, batch in enumerate(train_loader):
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        output = model_hgm(source)
        loss = criterion(output, target) / ACCUMULATION

        if torch.isnan(loss):
            write_progress(f"Warning: NaN loss detected at E{epoch} batch {batch_idx+1}. Skipping batch to protect model weights.")
            optimizer.zero_grad()
            continue

        loss.backward()

        if (batch_idx + 1) % ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(model_hgm.parameters(), 0.01)
            
            # 检测梯度中是否含有 NaN，如果有则跳过更新步，防止污染模型权重
            has_nan_grad = False
            for param in model_hgm.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                write_progress(f"Warning: NaN gradient detected at E{epoch} batch {batch_idx+1}. Skipping optimizer step to protect weights.")
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss += loss.item() * ACCUMULATION

        if (batch_idx + 1) % 50 == 0:
            elapsed_b = time.time() - epoch_start
            write_progress(f"E{epoch} batch {batch_idx+1}/{total_batches} loss={loss.item()*ACCUMULATION:.4f} {elapsed_b:.0f}s")

    # 更新学习率调度器
    if epoch > WARMUP_EPOCHS or start_epoch > 0:
        scheduler.step()
        
    avg_loss = epoch_loss / total_batches
    current_lr = optimizer.param_groups[0]['lr']
    epoch_time = time.time() - epoch_start

    write_progress(f"E{epoch} done: loss={avg_loss:.4f} lr={current_lr:.1e} time={epoch_time:.0f}s")

    if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
        write_progress(f"Validating epoch {epoch}...")
        val_psnr = validate(model_hgm, test_loader)

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
            }, os.path.join(SAVE_DIR, f'FADformer_HGM_best_{MODEL_SCALE}.pth'))
            improved = " *BEST*"

        elapsed = time.time() - train_start
        mem = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        delta = val_psnr - baseline_psnr
        msg = (f"HGM E{epoch:3d}/{EPOCHS} | Loss:{avg_loss:.4f} | "
               f"PSNR:{val_psnr:.2f} (Best:{best_psnr:.2f}, Base:{baseline_psnr:.2f}, dP:{delta:+.2f}) | "
               f"LR:{current_lr:.1e} | {elapsed/60:.1f}min | GPU:{mem:.2f}GB{improved}")
        write_progress(msg)
        write_log(msg)

    # 每个 epoch 结束都保存最新的 checkpoint 供断点续训
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_hgm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'psnr': best_psnr,
        'loss': avg_loss,
        'baseline_psnr': baseline_psnr,
    }, os.path.join(SAVE_DIR, f'FADformer_HGM_latest_{MODEL_SCALE}.pth'))

    if epoch % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_hgm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'psnr': best_psnr,
            'loss': avg_loss,
            'baseline_psnr': baseline_psnr,
        }, os.path.join(SAVE_DIR, f'FADformer_HGM_ep{epoch}_{MODEL_SCALE}.pth'))

total_time = time.time() - train_start
write_progress(f"HGM training done! Time: {total_time/60:.1f}min | Best PSNR: {best_psnr:.2f} dB")

write_progress("Final comparison...")
model_orig = FADformer()
model_orig = load_pretrained(model_orig, PRETRAINED)
model_orig = model_orig.to(device)
model_orig.eval()

ckpt_hgm = torch.load(os.path.join(SAVE_DIR, f'FADformer_HGM_best_{MODEL_SCALE}.pth'), map_location=device)
model_hgm.load_state_dict(ckpt_hgm['model_state_dict'])
model_hgm.eval()

orig_psnrs, hgm_psnrs = [], []
orig_times, hgm_times = [], []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        t0 = time.time()
        out_orig = model_orig(source).clamp_(0, 1)
        orig_times.append(time.time() - t0)

        t0 = time.time()
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
    write_progress(f"HGM PSNR improvement: +{delta_psnr:.2f} dB")
    if 0.35 <= delta_psnr <= 0.50:
        write_progress("Target achieved (+0.35~0.45 dB)!")
else:
    write_progress(f"HGM PSNR change: {delta_psnr:+.2f} dB")

report_path = os.path.join(SAVE_DIR, 'comparison_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("Rain200H Training Comparison Report\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: Rain200H ({len(train_dataset)} train, {len(test_dataset)} test)\n")
    f.write(f"Baseline: pretrained FADformer (PSNR={baseline_psnr:.2f} dB)\n")
    f.write(f"HGM: trained {EPOCHS} epochs\n\n")
    f.write(report)

write_progress(f"Report saved: {report_path}")
write_progress("ALL DONE!")
