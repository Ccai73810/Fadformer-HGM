# FADformer 复现指南

## 项目简介
FADformer (Efficient Frequency-Domain Image Deraining with Contrastive Regularization) 是一个高效的频域图像去雨模型，发表在 ECCV 2024 上。

## 目录结构
```
FADformer/
├── configs/              # 配置文件
├── datasets/             # 数据加载器
├── demo/                 # 示例输入输出
├── models/               # 模型定义
├── utils/                # 工具函数
├── train_*.py            # 训练脚本
├── predict_*.py          # 预测脚本
└── simple_test.py        # 简化测试脚本
```

## 环境要求

### 基础依赖
- Python 3.7+
- PyTorch 1.7+
- CUDA (推荐，用于加速训练和推理)

### 依赖库
```
torch
torchvision
numpy
opencv-python
scikit-image
tqdm
tensorboardX (用于训练可视化)
pytorch_msssim (用于评估指标)
```

## 安装步骤

### 1. 克隆项目
当前项目已经位于：`d:\NeRD-Rain\FADformer`

### 2. 创建虚拟环境 (可选)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3. 安装依赖
```bash
pip install torch torchvision numpy opencv-python scikit-image tqdm
pip install tensorboardX pytorch_msssim
```

## 使用方法

### 1. 快速测试 (推荐)
使用我们创建的简化测试脚本：
```bash
cd d:\NeRD-Rain\FADformer
python simple_test.py
```

这个脚本会：
- 检查并创建 FADformer 模型
- 使用随机输入测试前向传播
- 处理 demo/input/ 目录下的示例图片
- 结果保存到 demo/output_test/ 目录

### 2. 使用预训练权重进行推理

#### 下载预训练权重
根据 README.md 中的说明，从百度网盘下载预训练权重：
- Rain200L: [下载链接](https://pan.baidu.com/s/14-xuieEB4gW6VO5KHCcNFQ?pwd=cozn)
- Rain200H: [下载链接](https://pan.baidu.com/s/1kTEeWv6FvicdAa-m49M33A?pwd=qw5d)
- 其他数据集请参考原 README

#### 组织权重文件
创建目录结构并放置权重文件：
```
FADformer/
└── pretrain_weights/
    ├── rain200H/
    │   └── FADformer_Rain200H.pth
    ├── rain200L/
    │   └── FADformer_Rain200L.pth
    └── ...
```

#### 运行预测
```bash
python predict_and_save.py
```

### 3. 训练模型

#### 准备数据集
根据需要下载相应的数据集（Rain200, DID, DDN, SPA等）并放置到 `./datasets/` 目录。

#### 开始训练
以 Rain200 为例：
```bash
python train_rain200.py
```

其他数据集的训练脚本：
- `train_DDN.py` - 训练 DDN 数据集
- `train_DID.py` - 训练 DID 数据集  
- `train_spa.py` - 训练 SPA 数据集

## 模型架构

### 核心组件
1. **Fused Fourier Convolution Mixer (FFCM)** - 结合空域和频域的特征提取
2. **Prior-Gated Feed-forward Network (PGFN)** - 引入先验门控机制
3. **Frequency Contrastive Regularization (FCR)** - 频域对比学习正则化

### 模型变体
- `FADformer()` - 标准版本
- `FADformer_mini()` - 轻量级版本

## 代码示例

### 基本使用
```python
import torch
from models.FADformer import FADformer

# 创建模型
model = FADformer()
model = model.cuda()  # 如果有GPU
model.eval()

# 准备输入
input_tensor = torch.randn(1, 3, 256, 256).cuda()

# 推理
with torch.no_grad():
    output = model(input_tensor)
```

### 加载预训练权重
```python
checkpoint = torch.load('pretrain_weights/rain200H/FADformer_Rain200H.pth')
model.load_state_dict(checkpoint['state_dict'])
```

## 性能指标

模型在标准去雨数据集上的表现（参考原论文）：
- Rain200L: PSNR ~39dB+, SSIM ~0.98+
- Rain200H: PSNR ~30dB+, SSIM ~0.90+
- 其他数据集请参考原论文

## 常见问题

### 1. GPU内存不足
- 减小 batch_size（在 config 文件中修改）
- 使用 FADformer_mini 轻量级版本
- 使用梯度累积

### 2. 找不到数据集
- 检查数据集路径是否正确
- 确保数据集目录结构符合要求

### 3. 导入错误
- 确保当前目录在 Python 路径中
- 检查所有依赖是否正确安装

## 引用

如果使用本项目，请引用原论文：
```
@inproceedings{gao2025efficient,
  title={Efficient Frequency-Domain Image Deraining with Contrastive Regularization},
  author={Gao, Ning and Jiang, Xingyu and Zhang, Xiuhui and Deng, Yue},
  booktitle={European Conference on Computer Vision},
  pages={240--257},
  year={2025},
  organization={Springer}
}
```

## 联系方式
如有问题，请联系原作者：gaoning_ai@buaa.edu.cn
