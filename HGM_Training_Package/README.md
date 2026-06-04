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
* **`requirements.txt`**: PyTorch 及其他依赖库列表

---

## 💻 本地部署与环境配置

本部分指引您在本地（Windows 或 Linux）配置运行环境。

### 1. 硬件建议
* **GPU**: 推荐 NVIDIA GPU，显存 >= 12GB（如 RTX 3060/4060 及以上，若显存为 8GB-16GB，可通过调整 Batch Size 和梯度累积步数以防 OOM）。
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
LR = 2e-3                        # 初始学习率

# --- 显存与规模配置 ---
MODEL_SCALE = 'full'            # 模型规模: 'full'(全量版, 9.87M) 或 'mini'(轻量版, 2.38M)
BATCH_SIZE = 4                  # 单卡批大小 (显存不够时请调小, 如设为 2)
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
nohup python3 train_rain200h.py > train.out 2>&1 &
```
可随时查看进度：
```bash
tail -f saved_models/rain200h_real/progress.txt
```

---

## 💡 常见问题与优化策略 (Troubleshooting)

1. **显存溢出 (CUDA Out-of-Memory)**
   * **解决方法**: 在 `train_rain200h.py` 头部调小 `BATCH_SIZE` (例如从 4 改为 2)，并成比例调大 `ACCUMULATION` (例如从 4 改为 8)。这样可以保持相同的有效批大小 (Effective Batch Size = 16) 且显存占用减半。
2. **训练时报错缺少 FCR 模块 (`ModuleNotFoundError: No module named 'models.FCR'`)**
   * **原因**: `models/__init__.py` 中存在对 `FCR` 的全局导入，但训练中实际并没有使用它。
   * **解决方法**: 本包已附带 `FCR.py`。若在旧包中遇到，只需在 `models/__init__.py` 中将 `from models.FCR import FCR` 注释掉，或者直接补全 `models/FCR.py`。
3. **找不到数据路径 (`FileNotFoundError`)**
   * **解决方法**: 请核对 `Rain200H` 目录下的结构是否严格为 `train/input`、`train/target`、`test/input`、`test/target`。
