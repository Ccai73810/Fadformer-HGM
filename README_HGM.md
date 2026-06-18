# FADformer-HGM: 混合全局混合器改进方案

> 基于稀疏窗口注意力与融合傅里叶卷积的并行融合架构，用于图像去雨任务

---

## 1. 项目背景

FADformer（Frequency-Aware Deraining Transformer）是一种基于频域感知的图像去雨网络。其核心思想是利用傅里叶变换在频域中分离雨痕与图像内容，再通过卷积混合器进行特征融合。

**原始方法的局限性**：FADformer 的 Token Mixer 仅依赖频域分支（FFCM）进行全局建模，缺少显式的空间注意力机制，导致对复杂空间变化的雨痕建模能力不足。

**改进思路**：引入稀疏窗口注意力作为空间分支，与频域分支并行融合，通过自适应门控机制动态平衡两个分支的贡献。

---

## 2. 原始方法：Fused Fourier Conv Mixer (FFCM)

### 2.1 架构概述

原始 FADformer 的 Token Mixer 由 `Fused_Fourier_Conv_Mixer` 实现，其处理流程如下：

```
输入 x (B, C, H, W)
    │
    ├── conv_init: Conv1x1(C → 2C) + GELU
    │
    ├── split ──┬── dw_conv_1: DWConv3x3 + GELU  (局部特征1)
    │           └── dw_conv_2: DWConv5x5 + GELU  (局部特征2)
    │
    ├── concat → Freq_Fusion (频域融合)
    │       │
    │       ├── conv_init_1 + conv_init_2 (双路投影)
    │       ├── FourierUnit (FFT → Conv1x1 → ReLU → iFFT)
    │       └── 残差连接: FFC(x) + x
    │
    ├── ca_conv: Conv1x1 + DWConv3x3 + GELU
    │
    └── Channel Attention (SE模块)
            │
            └── 输出 (B, C, H, W)
```

### 2.2 核心组件

#### FourierUnit

```python
class FourierUnit(nn.Module):
    def forward(self, x):
        ffted = torch.fft.rfft2(x, norm='ortho')     # 实数FFT
        ffted = cat(real(ffted), imag(ffted))          # 拼接实部虚部
        ffted = self.conv_layer(ffted)                  # 频域卷积
        ffted = self.relu(self.bn(ffted))
        output = torch.fft.irfft2(ffted, norm='ortho') # 逆FFT
        return output
```

- 对输入做 2D 实数 FFT，将空间信号转换到频域
- 在频域用 1×1 卷积处理频谱系数
- 再通过逆 FFT 还原到空间域
- 复杂度：O(n log n)

#### Freq_Fusion

```python
class Freq_Fusion(nn.Module):
    def forward(self, x):
        x_1, x_2 = split(x, dim=dim)     # 通道分割
        x_1 = conv_init_1(x_1)            # 投影1
        x_2 = conv_init_2(x_2)            # 投影2
        x0 = cat([x_1, x_2])              # 拼接
        x = FFC(x0) + x0                  # 频域处理 + 残差
        return x
```

#### Channel Attention (SE模块)

```python
# 通道注意力
x = AdaptiveAvgPool2d(1)(x)     # 全局平均池化
x = Conv1x1(C → C/4)(x)         # 降维
x = GELU(x)
x = Conv1x1(C/4 → C)(x)         # 升维
x = Sigmoid(x)                   # 生成通道权重
output = x * input               # 通道加权
```

### 2.3 FFCM 的优缺点

| 优点 | 缺点 |
|------|------|
| 频域全局感受野，一次 FFT 覆盖整张图 | 缺少显式空间关系建模 |
| O(n log n) 复杂度，效率较高 | 对局部空间结构关注不足 |
| 频域卷积天然适合周期性雨痕 | 无法捕捉非周期性的空间变化 |
| 深度卷积提取多尺度局部特征 | 局部与全局特征融合方式简单（仅拼接+频域处理） |

---

## 3. 改进方法：Hybrid Global Mixer (HGM)

### 3.1 设计动机

雨痕具有双重特性：
1. **频域特性**：雨痕在频域中表现为高频分量，FFCM 能有效处理
2. **空间特性**：雨痕的方向、密度、遮挡关系具有空间局部性，需要空间注意力建模

HGM 的核心思想：**空间注意力分支 + 频域分支并行融合**，让模型同时从两个互补的角度理解雨痕。

### 3.2 架构概述

```
输入 x (B, C, H, W)
    │
    ├── conv_init: Conv1x1(C → 2C) + GELU
    │
    ├── split ──┬── dw_conv_1: DWConv3x3 + GELU  (局部特征1)
    │           └── dw_conv_2: DWConv5x5 + GELU  (局部特征2)
    │
    ├── x_local = x_local_1 + x_local_2  (逐元素求和)
    │
    ├── HybridGlobalMixer (核心创新)
    │       │
    │       ├─── 分支1: SparseWindowAttention (空间分支)
    │       │       │
    │       │       ├── LayerNorm
    │       │       ├── Cyclic Shift (循环移位)
    │       │       ├── Window Partition (窗口划分)
    │       │       ├── WindowAttention (窗口内自注意力 + 相对位置偏置)
    │       │       ├── Window Reverse (窗口还原)
    │       │       ├── Cyclic Unshift (反向移位)
    │       │       └── 残差连接: shortcut + attn_output
    │       │
    │       ├─── 分支2: FFCM (频域分支，与原始相同)
    │       │       │
    │       │       └── Freq_Fusion → Channel Attention
    │       │
    │       └─── Adaptive Gate (自适应门控融合)
    │               │
    │               ├── Concat[x_attn, x_fft]
    │               ├── Conv1x1(2C → C) + Sigmoid → g
    │               └── Y = g ⊙ x_attn + (1-g) ⊙ x_fft
    │
    ├── ca_conv: Conv1x1 + DWConv3x3 + GELU
    │
    └── Channel Attention (SE模块)
            │
            └── 输出 (B, C, H, W)
```

### 3.3 核心组件详解

#### 3.3.1 SparseWindowAttention（稀疏窗口注意力）

基于 Swin Transformer 的 Shifted Window Self-Attention，将全局注意力分解为局部窗口注意力：

```python
class SparseWindowAttention(nn.Module):
    def forward(self, x):
        # x: (B, C, H, W)
        shortcut = x
        x = LayerNorm(x)

        # 循环移位（实现跨窗口信息交互）
        shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))

        # 窗口划分: (B, C, H, W) → (num_windows*B, M*M, C)
        x_windows = window_partition(shifted_x, window_size)

        # 窗口内自注意力（含相对位置偏置）
        attn_windows = WindowAttention(x_windows, mask=attn_mask)

        # 窗口还原 + 反向移位
        x = window_reverse(attn_windows)
        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))

        # 残差连接
        return shortcut + x
```

**关键设计**：
- **窗口划分**：将特征图切为 M×M 的不重叠窗口（默认 M=8）
- **移位窗口**：交替使用常规窗口和移位窗口，实现跨窗口信息流动
- **相对位置偏置**：每个注意力头学习独立的相对位置偏置表
- **复杂度**：O(n)，其中 n = H×W，远低于全局注意力的 O(n²)

#### 3.3.2 WindowAttention（窗口内注意力）

```python
class WindowAttention(nn.Module):
    def forward(self, x, mask=None):
        # x: (num_windows*B, M*M, C)
        qkv = Linear(x).reshape(B_, N, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.T) / sqrt(head_dim)
        attn = attn + relative_position_bias    # 相对位置偏置
        attn = softmax(attn)

        output = (attn @ v).reshape(B_, N, C)
        output = Linear(output)
        return output
```

#### 3.3.3 Adaptive Gate（自适应门控融合）

```python
# 门控机制
gate = Sigmoid(Conv1x1(Concat[x_attn, x_fft]))  # g ∈ [0, 1]
output = gate * x_attn + (1 - gate) * x_fft      # 自适应加权
```

**门控机制的三种模式**：

| 模式 | 公式 | 特点 |
|------|------|------|
| `gate`（推荐） | `Y = g ⊙ X_attn + (1-g) ⊙ X_fft` | 逐像素自适应，最灵活 |
| `learnable` | `Y = α·X_attn + β·X_fft` | 全局可学习权重，参数少 |
| `sum` | `Y = X_attn + X_fft` | 最简单，无额外参数 |

### 3.4 HGM 与 FFCM 的关键区别

```
原始 FFCM:
  局部特征 → 频域处理 → 通道注意力
  (串行处理，频域是唯一的全局建模手段)

改进 HGM:
  局部特征 ─┬→ 空间注意力 ─┐
             └→ 频域处理   ─┤→ 门控融合 → 通道注意力
                             (并行处理，空间+频域双重建模)
```

| 维度 | FFCM（原始） | HGM（改进） |
|------|-------------|-------------|
| 全局建模方式 | 仅频域（FFT） | 频域 + 空间注意力 |
| 空间关系建模 | 无显式建模 | 窗口自注意力 + 相对位置偏置 |
| 分支结构 | 单分支（串行） | 双分支（并行） |
| 融合方式 | 拼接 + 频域卷积 | 自适应门控 |
| 复杂度 | O(n log n) | O(n) + O(n log n) |
| 参数量 | 基准 | +30.8% |

---

## 4. 两种方法的详细对比

### 4.1 架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    原始 FADformer Token Mixer                    │
│                                                                  │
│   Input ──→ Conv1x1 ──→ Split ──→ DWConv3×3 ──┐                │
│                                  └→ DWConv5×5 ──┤                │
│                                                  ↓                │
│                                          Freq_Fusion              │
│                                           (FFT→Conv→iFFT)         │
│                                                  ↓                │
│                                          Channel Attn ──→ Output  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    改进 HGM Token Mixer                          │
│                                                                  │
│   Input ──→ Conv1x1 ──→ Split ──→ DWConv3×3 ──┐                │
│                                  └→ DWConv5×5 ──┤                │
│                                          ↓ (sum)                 │
│                                   x_local_fused                  │
│                                    ┌─────┴─────┐                 │
│                                    ↓             ↓                │
│                          SparseWinAttn      FFCM                 │
│                          (空间分支)        (频域分支)              │
│                                    ↓             ↓                │
│                              x_attn          x_fft               │
│                                    └─────┬─────┘                 │
│                                          ↓                        │
│                              Adaptive Gate Fusion                 │
│                           g⊙x_attn + (1-g)⊙x_fft                │
│                                          ↓                        │
│                                  Channel Attn ──→ Output         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 参数量对比

| 模型 | 参数量 | 相对增加 |
|------|--------|---------|
| FADformer_mini（原始） | 1.82M | 基准 |
| FADformer_HGM_mini（改进） | 2.38M | +30.8% |

增加的参数主要来自：
- WindowAttention 的 QKV 投影和输出投影
- 相对位置偏置表
- 门控融合的 Conv1x1

### 4.3 复杂度对比

| 操作 | FFCM | HGM |
|------|------|-----|
| 局部特征提取 | O(n) | O(n) |
| 全局建模 | O(n log n)（FFT） | O(n)（窗口注意力）+ O(n log n)（FFT） |
| 融合 | O(n)（拼接+卷积） | O(n)（门控） |
| **总计** | **O(n log n)** | **O(n log n)** |

HGM 的渐近复杂度与 FFCM 相同，因为窗口注意力是 O(n) 的。

### 4.4 感受野对比

| 特性 | FFCM | HGM |
|------|------|-----|
| 频域感受野 | 全局（一次 FFT） | 全局（一次 FFT） |
| 空间感受野 | 仅局部（DWConv） | 局部 + 窗口内全局 + 跨窗口（移位机制） |
| 位置感知 | 无 | 相对位置偏置 |
| 空间自适应 | 无 | 逐像素门控权重 |

### 4.5 理论优势分析

1. **互补性**：空间注意力捕捉局部结构关系（雨痕方向、遮挡），频域处理捕捉全局周期性模式（雨痕频率），两者互补
2. **自适应性**：门控机制让模型根据输入动态调整两个分支的权重，不同区域可以有不同的融合策略
3. **位置感知**：相对位置偏置让模型显式编码空间位置关系，有利于区分雨痕和图像纹理
4. **跨窗口交互**：移位窗口机制在不增加计算量的情况下实现跨窗口信息流动

---

## 5. 实验设置

### 5.1 训练配置

| 参数 | 值 |
|------|-----|
| 数据集 | Rain200L (1800 训练 + 200 测试) |
| 图像尺寸 | 256 × 256 |
| Batch Size | 4 |
| 优化器 | AdamW (weight_decay=0.01) |
| 学习率 | 1e-3 → 1e-6 (Cosine Annealing) |
| 损失函数 | L1 Loss |
| 训练轮数 | 300 epochs |
| 混合精度 | FP16 (GradScaler + autocast) |
| GPU | NVIDIA RTX 4060 Laptop (8GB) |

### 5.2 模型配置

| 参数 | 原始 FADformer | HGM FADformer |
|------|---------------|---------------|
| embed_dim | [24, 48, 96, 48, 24] | [24, 48, 96, 48, 24] |
| depth | [2, 3, 4, 3, 2] | [2, 3, 4, 3, 2] |
| window_size | - | 8 |
| num_heads | - | 4 |
| fusion_mode | - | gate |

---

## 6. 文件结构

```
FADformer/
├── models/
│   ├── FADformer.py          # 原始模型 + HGM集成
│   │   ├── Fused_Fourier_Conv_Mixer   # 原始Token Mixer
│   │   ├── HGM_TokenMixer             # HGM Token Mixer
│   │   ├── FADBlock                   # 基础Block (支持两种Mixer)
│   │   ├── FADBackbone                # 骨干网络
│   │   ├── FADformer_mini()           # 原始mini模型
│   │   └── FADformer_HGM_mini()       # HGM mini模型
│   │
│   ├── HGM.py                # HGM核心模块
│   │   ├── window_partition()         # 窗口划分
│   │   ├── window_reverse()           # 窗口还原
│   │   ├── WindowAttention            # 窗口内注意力
│   │   ├── SparseWindowAttention      # 稀疏窗口注意力
│   │   └── HybridGlobalMixer          # 混合全局混合器
│   │
│   └── FCR.py                # 原始频域组件
│
├── train_real_data.py        # 真实数据训练脚本 (Rain200L)
├── train_hgm_4060.py         # RTX 4060优化训练脚本
├── continue_training.py      # 断点续训脚本
├── compare_models.py         # 模型对比脚本
├── compare_real_data.py      # 真实数据对比脚本
├── diagnose_hgm.py           # HGM诊断工具
│
├── saved_models/
│   ├── rain200l_real/        # 真实数据训练结果
│   └── hgm_4060/             # RTX 4060训练结果
│
├── tests/
│   └── test_hgm.py           # HGM单元测试 (9/9通过)
│
└── configs/
    └── rain200/
        └── FADformer_HGM_ablation.json  # 消融实验配置
```

---

## 7. 使用方法

### 7.1 训练

```bash
# 真实数据训练 (Rain200L)
C:\ProgramData\anaconda3\python.exe train_real_data.py

# RTX 4060 优化训练
C:\ProgramData\anaconda3\python.exe train_hgm_4060.py

# 断点续训
C:\ProgramData\anaconda3\python.exe continue_training.py
```

### 7.2 对比测试

```bash
# 真实数据对比
C:\ProgramData\anaconda3\python.exe compare_real_data.py

# 合成数据对比
C:\ProgramData\anaconda3\python.exe compare_models.py
```

### 7.3 代码中创建模型

```python
from models.FADformer import FADformer_mini, FADformer_HGM_mini

# 原始模型
model_orig = FADformer_mini()

# HGM改进模型
model_hgm = FADformer_HGM_mini(
    window_size=8,       # 窗口大小
    num_heads=4,         # 注意力头数
    fusion_mode='gate'   # 融合模式: 'gate' / 'sum' / 'learnable'
)
```

---

## 8. 消融实验建议

| 实验 | 配置 | 目的 |
|------|------|------|
| Baseline | FADformer_mini | 基准性能 |
| +HGM (sum) | fusion_mode='sum' | 验证双分支并行是否有效 |
| +HGM (gate) | fusion_mode='gate' | 验证门控融合是否优于简单相加 |
| +HGM (w=4) | window_size=4 | 窗口大小影响 |
| +HGM (w=16) | window_size=16 | 窗口大小影响 |
| +HGM (h=2) | num_heads=2 | 注意力头数影响 |
| +HGM (h=8) | num_heads=8 | 注意力头数影响 |
| 仅注意力 | 去掉FFCM分支 | 注意力分支单独贡献 |
| 仅FFCM | 去掉注意力分支 | 频域分支单独贡献 |

---

## 9. 参考文献

1. **FADformer**: Frequency-Aware Deraining Transformer (原始基线模型)
2. **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021 (窗口注意力设计参考)
3. **FFC**: Chi et al., "Fast Fourier Convolution", NeurIPS 2020 (频域卷积设计参考)
4. **NeRD-Rain**: Chen et al., "Bidirectional Multi-Scale Implicit Neural Representations for Image Deraining", CVPR 2024 (项目基础框架)
