# FADformer-HGM 独立训练与部署包

本仓库提供了集成了 **混合全局混频器 (Hybrid Global Mixer, HGM)**（基于 Swin 窗口自注意力 + 快速傅里叶变换混频器 FFCM 配合自适应门控融合）的 **FADformer** 图像去雨模型核心代码及独立训练脚本。

---

## 📁 目录结构

* **`models/`**: 模型结构定义
  * `FADformer.py`: 骨干网络定义（包含 `FADformer_HGM`、`FADformer_HGM_mini` 等）
  * `HGM.py`: 混合全局混频器（HGM）模块实现
  * `FCR.py`: 频域对比正则化相关模块
* **`datasets/`**: 数据集加载
  * `Rain_Dataloader.py`: Rain200H/Rain200L 等数据集加载器
* **`utils/`**: 工具函数
  * 包含模型保存、日志记录、PSNR计算等辅助工具
* **`train_rain200h.py`**: 主训练微调脚本，支持实数尺度随机裁剪与余弦退火学习率
* **`diagnose_baseline.py`**: 基准测试异常排查诊断工具，检查数据集和权重对齐性
* **`requirements.txt`**: PyTorch 及其他依赖库列表

---

## 💻 本地部署与环境配置

本部分指引您在本地（Windows 或 Linux）配置运行环境。

### 1. 硬件建议
* **GPU**: 推荐 NVIDIA GPU，显存 >= 12GB。如果在 16GB (如 Tesla T4) 或 24GB (如 RTX 4090) 显存上运行，均可完美跑通。
* **CUDA**: 建议 CUDA 11.3 及以上版本。

### 2. 创建并激活虚拟环境 (可选)
建议使用虚拟环境隔离依赖：

* **Python venv**:
  ```bash
  python -m venv .venv
  # Windows 激活:
  .venv\\Scripts\\activate
  # Linux/macOS 激活:
  source .venv/bin/activate
  ```
* **Conda 虚拟环境**:
  ```bash
  conda create -n fadformer python=3.9 -y
  conda activate fadformer
  ```

### 3. 安装依赖项
首先安装 PyTorch 和 Torchvision。如果您使用 GPU 训练，请确保安装了支持 CUDA 的 PyTorch。

**安装 GPU 版 PyTorch（以 CUDA 11.8 为例）：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**安装其他依赖库：**
```bash
pip install -r requirements.txt
```

---

## 📥 数据集与预训练权重准备

### 1. 下载 Rain200H 数据集
* **下载链接**: [百度网盘](https://pan.baidu.com/s/1kTEeWv6FvicdAa-m49M33A) 提取码：`qw5d`
* 下载完成后，解压并将文件夹组织为以下结构：
  ```text
  HGM_Training_Package/
  └── Rain200H/
      ├── train/
      │   ├── input/     # 有雨图像 (例如 1.png, 2.png)
      │   └── target/    # 无雨真值图像 (对应 1.png, 2.png)
      └── test/
          ├── input/     # 测试有雨图像
          └── target/    # 测试无雨真值图像
  ```

### 2. 下载原始 FADformer 预训练权重 (作为基线)
* **下载链接**: [百度网盘](https://pan.baidu.com/s/1kTEeWv6FvicdAa-m49M33A) 提取码：`qw5d` （或在上面同一链接中下载对应权重的 `.pth` 文件）
* 在主目录下创建 `pretrain_weights/rain200H/` 目录，并将权重文件命名为 `FADformer_Rain200H.pth` 放入其中：
  ```text
  HGM_Training_Package/
  └── pretrain_weights/
      └── rain200H/
          └── FADformer_Rain200H.pth
  ```

---

## ⚙️ 训练配置与参数说明

打开 `train_rain200h.py`，可以在脚本头部找到以下核心配置参数，根据您的硬件配置进行修改：

```python
DATA_DIR = './Rain200H'          # 数据集根目录路径
SAVE_DIR = './saved_models/rain200h_real'  # 训练结果与模型保存路径
PRETRAINED = './pretrain_weights/rain200H/FADformer_Rain200H.pth'  # 预训练权重路径

EPOCHS = 300                     # 训练轮数
LR = 1e-3                        # 初始学习率 (推荐 1e-3 以保证稳定性)

# --- 显存与规模配置 ---
MODEL_SCALE = 'full'            # 模型规模: 'full'(全量版, 9.87M) 或 'mini'(轻量版, 2.38M)
BATCH_SIZE = 4                  # 单卡批大小 (24GB 显卡推荐设为 4, 16GB 显存推荐设为 2)
ACCUMULATION = 4                # 梯度累积步数。保持 BATCH_SIZE * ACCUMULATION = 16 左右以维持稳定收敛
IMG_SIZE = 256                  # 随机裁剪的目标图像大小（256x256，保持与原论文一致的感受野）
```

---

## 🚀 启动训练与验证

### 1. 运行训练脚本
```bash
python train_rain200h.py
```

### 2. 进度监控与日志
* 训练过程中，脚本会实时生成进度文件 `saved_models/rain200h_real/progress.txt`。
* 完整的训练日志将写入 `saved_models/rain200h_real/HGM_train_log.txt`。
* 最优模型将自动保存为 `saved_models/rain200h_real/FADformer_HGM_best_full.pth` (或 `_mini.pth`)。

### 3. 后台训练 (适用于 Linux 服务器)
如果是在远程服务器上训练，建议使用后台命令以防终端断开连接：
```bash
# 开启防碎片化显存优化，并后台运行
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python3 train_rain200h.py > train.out 2>&1 &
```
可随时查看进度：
```bash
tail -f saved_models/rain200h_real/progress.txt
```

### 4. 基准测试异常排查 (重要)
如果您在开始训练前，发现评估出来的 **Baseline PSNR 异常低（例如低于 20 dB，正常应接近 30 dB）**，请立即在当前包下运行排查脚本诊断数据集或权重问题：
```bash
python diagnose_baseline.py
```
该脚本会自动检测权重大小与参数项匹配度、测试集文件数目及文件名对齐性、以及计算不经过模型的直连 PSNR。

---

## 💡 关键特性与优化机制

### 1. 梯度检查点 (Gradient Checkpointing)
本包对 `FADformer_HGM` 全量模型原生支持梯度检查点优化（默认开启）。该技术在正向传播时不保存中间层激活值，而是在反向传播时实时重算。
* **效果**：将训练全量模型时的显存占用直接降低 **60% 以上**（从 23GB+ 降至 **5~6GB**），让您可以在 12G/16G/24G 的显卡上极速运行较大的 Batch Size。

### 2. 纯 FP32 单精度运行（杜绝 NaN 报错）
傅里叶变换（FFT/IFFT）操作对数值精度极其敏感。如果在训练或验证时使用自动混合精度（AMP/autocast），会导致 FFT 算子在 `ComplexHalf` 精度下产生数值溢出，使 Loss 变为 `nan`。
* **设计**：本包的训练和验证默认运行于纯 **FP32 精度** 下，保证数值绝对稳定，杜绝任何 `nan` 错误。

### 3. 鲁棒权重加载机制
我们在 `load_pretrained` 函数中增加了自动维度检查。当全量版和轻量版存在通道差异（或 HGM 的参数发生更改）时，加载器会**自动过滤并跳过维度不匹配的参数**，而将匹配的绝大部分原网络骨干层参数正常载入，实现平滑的微调（Fine-tuning）起步，不会因为参数维度冲突而报错崩溃。

### 4. 显存防碎片优化
如果您在启动训练时显存较为紧张，可以通过设置以下环境变量，利用 PyTorch 的虚拟内存映射机制彻底解决显存碎片堆积问题：
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
