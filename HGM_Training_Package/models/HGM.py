import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图划分为不重叠的局部窗口
    
    Args:
        x: (B, H, W, C) 输入特征
        window_size: 窗口大小 (M)
    
    Returns:
        (num_windows*B, window_size, window_size, C) 窗口化特征
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    将窗口还原为特征图
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: 窗口大小 (M)
        H: 原始高度
        W: 原始宽度
    
    Returns:
        (B, H, W, C) 还原的特征图
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    窗口内的多头自注意力机制
    
    基于相对位置偏置（Relative Position Bias）
    
    Args:
        dim: 输入特征维度
        window_size: 窗口大小
        num_heads: 注意力头数
        qkv_bias: 是否使用偏置
        attn_drop: 注意力 dropout 比例
        proj_drop: 输出投影 dropout 比例
    """
    
    def __init__(self, dim: int, window_size: int, num_heads: int, 
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 计算相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # 相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (num_windows*B, N, C) 输入特征，N = window_size*window_size
            mask: (num_windows, N, N) 或 None 注意力掩码
        
        Returns:
            (num_windows*B, N, C) 输出特征
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SparseWindowAttention(nn.Module):
    """
    稀疏窗口自注意力（基于 Swin Transformer 的 Shifted Window Attention）
    
    特点：
    - 将特征图划分为不重叠的局部窗口（默认 8x8）
    - 在窗口内计算自注意力，复杂度 O(n)
    - 通过移位窗口机制实现跨窗口信息交互
    - 支持相对位置偏置
    
    Args:
        dim: 特征维度
        window_size: 窗口大小（推荐：4/8/16，默认 8）
        num_heads: 注意力头数
        shift_size: 移位大小（0 表示不移位）
        qkv_bias: QKV 是否使用偏置
        attn_drop: 注意力 dropout
        proj_drop: 投影 dropout
    """
    
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8,
                 shift_size: int = 0, qkv_bias: bool = True, 
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        
        assert 0 <= self.shift_size < self.window_size, \
            f"shift_size must be in [0, {self.window_size})"
        
        self.norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, C, H, W) 输入特征图
            attn_mask: 注意力掩码（可选）
        
        Returns:
            (B, C, H, W) 输出特征图
        """
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x
        
        # 划分窗口
        x_windows = window_partition(shifted_x.permute(0, 2, 3, 1), self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # 还原窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W).permute(0, 3, 1, 2)
        
        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x
        
        x = shortcut + x
        return x
    
    def get_attn_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        生成移位窗口的注意力掩码
        
        Args:
            H: 高度
            W: 宽度
            device: 设备
        
        Returns:
            (nW, M*M, M*M) 注意力掩码，其中 M = window_size
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask


class HybridGlobalMixer(nn.Module):
    """
    Hybrid Global Mixer (HGM): 稀疏窗口注意力 + FFCM 并联融合
    
    结构：
    ├── SparseWindowAttention (空间分支) - 捕捉局部空间关系
    │   └── Shifted Window Self-Attention (O(n) 复杂度)
    ├── FFCM (频域分支) - 捕捉全局频域模式
    │   └── Fused_Fourier_Conv_Mixer (O(n log n) 复杂度)
    └── Adaptive Gate (门控融合) - 自适应平衡两个分支
        ├── Concat + Conv1x1 + Sigmoid
        └── Weighted Sum: Y = g ⊙ X_attn + (1-g) ⊙ X_fft
    
    Args:
        dim: 特征维度
        window_size: 注意力窗口大小（默认 8）
        num_heads: 注意力头数（默认 8）
        shift_size: 移位大小（默认 window_size//2）
        fusion_mode: 融合模式 ('gate'/'sum'/'learnable')
        attn_ratio: 注意力分支的通道缩放比例（仅 learnable 模式使用）
        attn_drop: 注意力 dropout
        proj_drop: 投影 dropout
    """
    
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8,
                 shift_size: int = None, fusion_mode: str = 'gate',
                 attn_ratio: float = 1.0, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.fusion_mode = fusion_mode
        self.window_size = window_size
        
        if shift_size is None:
            shift_size = window_size // 2
        
        # 分支1: 稀疏窗口注意力（空间分支）
        self.sparse_attn = SparseWindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            shift_size=shift_size,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # 分支2: 频域融合卷积混合器（复用现有 FFCM）
        from models.FADformer import Fused_Fourier_Conv_Mixer
        self.ffcm = Fused_Fourier_Conv_Mixer(
            dim=dim,
            mixer_kernel_size=[1, 3, 5, 7],
            local_size=window_size
        )
        
        # 门控融合机制
        if fusion_mode == 'gate':
            # 可学习门控: g = σ(Conv1x1(Concat[X_attn, X_fft]))
            self.gate_conv = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1),
                nn.Sigmoid()
            )
        elif fusion_mode == 'learnable':
            # 可学习权重: α·X_attn + β·X_fft
            self.alpha = nn.Parameter(torch.ones(1) * 0.5)
            self.beta = nn.Parameter(torch.ones(1) * 0.5)
        elif fusion_mode != 'sum':
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}. Must be 'gate', 'sum', or 'learnable'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Algorithm:
            1. X_attn ← SparseWindowAttention(X)
            2. X_fft ← FFCM(X)
            3. g ← σ(Conv_{1×1}(Concat[X_attn, X_fft]))  [if gate mode]
            4. Y ← g ⊙ X_attn + (1-g) ⊙ X_fft           [adaptive fusion]
        
        Args:
            x: (B, C, H, W) 输入特征图
        
        Returns:
            (B, C, H, W) 融合后的输出特征图
        """
        # 分支1: 空间注意力
        x_attn = self.sparse_attn(x)
        
        # 分支2: 频域处理
        x_fft = self.ffcm(x)
        
        # 自适应融合
        if self.fusion_mode == 'gate':
            # 计算门控权重
            gate = self.gate_conv(torch.cat([x_attn, x_fft], dim=1))
            # 加权融合: Y = g ⊙ X_attn + (1-g) ⊙ X_fft
            out = gate * x_attn + (1 - gate) * x_fft
        elif self.fusion_mode == 'learnable':
            # 可学习权重融合
            alpha = torch.sigmoid(self.alpha)
            beta = torch.sigmoid(self.beta)
            out = alpha * x_attn + beta * x_fft
        else:  # sum mode
            # 简单求和
            out = x_attn + x_fft
        
        return out
    
    def get_fusion_weights(self) -> dict:
        """
        获取当前融合权重（用于可视化和分析）
        
        Returns:
            dict: 包含各分支的权重信息
        """
        if self.fusion_mode == 'gate':
            return {
                'mode': 'gate',
                'alpha_mean': None,  # 需要在前向传播后计算
                'beta_mean': None
            }
        elif self.fusion_mode == 'learnable':
            alpha = torch.sigmoid(self.alpha).item()
            beta = torch.sigmoid(self.beta).item()
            return {
                'mode': 'learnable',
                'attn_weight': alpha,
                'fft_weight': beta
            }
        else:
            return {'mode': 'sum'}
