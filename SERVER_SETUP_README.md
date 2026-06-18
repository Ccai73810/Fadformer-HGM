# 华为云 FADformer HGM 训练服务器配置文档

## 服务器基本信息

| 项目 | 配置 |
|------|------|
| 云服务商 | 华为云 |
| 操作系统 | Huawei Cloud EulerOS 2.0 (x86_64) |
| IP地址 | 113.47.10.87 |
| 用户名 | root |
| GPU | Tesla T4 (16GB VRAM) |
| Python | 3.9.9 |
| PyTorch | 2.0.1+cu117 |

## 服务器文件结构

```
/root/FADformer/
├── train_rain200h.py          # 训练脚本（需要更新）
├── datasets/                  # 数据集加载代码
├── models/                    # 模型代码
│   ├── FADformer.py          # 原始FADformer模型
│   ├── HGM.py                # HGM混合全局混合器
│   └── __init__.py
├── utils/                     # 工具函数
├── pretrain_weights/          # 预训练权重
│   └── rain200H/
│       └── FADformer_Rain200H.pth
├── Rain200H/                  # Rain200H数据集
│   ├── train/
│   │   ├── input/            # 训练输入（含雨图像）
│   │   └── target/           # 训练目标（去雨真值）
│   └── test/
│       ├── input/            # 测试输入
│       └── target/           # 测试目标
├── saved_models/              # 保存训练结果
│   └── rain200h_real/
└── tests/                     # 测试代码
```

## 环境配置关键信息

### 1. PyTorch CUDA 库路径配置

服务器的 `LD_LIBRARY_PATH` 需要正确配置才能让 PyTorch 找到 NVIDIA CUDA 库：

```bash
# 临时配置（当前终端有效）
export LD_LIBRARY_PATH=$(find /usr/local/lib/python3.9/site-packages/nvidia -type d -name lib | tr '\n' ':')$LD_LIBRARY_PATH

# 永久配置（已写入 ~/.bashrc）
echo 'export LD_LIBRARY_PATH=$(find /usr/local/lib/python3.9/site-packages/nvidia -type d -name lib | tr "\n" ":")$LD_LIBRARY_PATH' >> ~/.bashrc

# 验证配置
python3 -c "import torch; print(f'PyTorch:{torch.__version__} CUDA:{torch.cuda.is_available()} GPU:{torch.cuda.get_device_name(0)}')"
```

预期输出：
```
PyTorch:2.0.1+cu117 CUDA:True GPU:Tesla T4
```

### 2. 已安装依赖

```bash
pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install pillow numpy scipy
```

### 3. 训练脚本问题修复

服务器上的 `train_rain200h.py` 需要修复以下问题：

**问题1：`from torch.amp import autocast, GradScaler` ImportError**
- 原因：PyTorch 2.0.1 的 GradScaler 还在 `torch.cuda.amp` 里
- 修复方案：使用兼容导入

**问题2：数据集路径**
- 已修复为 `DATA_DIR = './Rain200H'`
- 图像尺寸 `IMG_SIZE = 256`
- Batch size `BATCH_SIZE = 8`

## 如何连接服务器

```bash
# SSH连接
ssh root@113.47.10.87
# 密码：（联系管理员获取）
```

## 如何上传文件

```bash
# 从本地Windows上传文件到服务器（在本地cmd/PowerShell执行）
scp d:\NeRD-Rain\FADformer\文件.py root@113.47.10.87:/root/FADformer/

# 上传文件夹
scp -r d:\NeRD-Rain\FADformer\文件夹 root@113.47.10.87:/root/FADformer/
```

## 如何启动训练

```bash
# 1. 登录服务器
ssh root@113.47.10.87

# 2. 进入项目目录
cd /root/FADformer

# 3. 确保LD_LIBRARY_PATH已加载（如果SSH重连）
source ~/.bashrc

# 4. 启动训练（后台运行）
nohup python3 train_rain200h.py > train.out 2>&1 &

# 5. 查看训练进度
tail -f saved_models/rain200h_real/progress.txt

# 6. 查看完整日志
cat saved_models/rain200h_real/HGM_train_log.txt

# 7. 如果出错，查看错误日志
cat train.out
```

## 训练配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| DATA_DIR | ./Rain200H | 数据集目录 |
| IMG_SIZE | 256 | 输入图像尺寸（完整分辨率，公平对比） |
| BATCH_SIZE | 8 | Tesla T4 16GB显存可以支持 |
| LR | 2e-3 | 学习率 |
| EPOCHS | 300 | 训练轮数 |
| ACCUMULATION | 2 | 梯度累积步数 |

## 目标

在 Rain200H 数据集上训练 FADformer_HGM_mini，目标 PSNR 相比预训练 FADformer 基线提升 **+0.35~0.45 dB**。

## 快速诊断命令

```bash
# 检查GPU状态
nvidia-smi

# 检查PyTorch是否能用GPU
python3 -c "import torch; print(f'CUDA:{torch.cuda.is_available()} GPU:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 检查数据集文件数量
ls -1 /root/FADformer/Rain200H/train/input | wc -l
ls -1 /root/FADformer/Rain200H/test/input | wc -l

# 检查训练是否在运行
ps aux | grep python3 | grep -v grep

# 查看当前训练进度
tail -f /root/FADformer/saved_models/rain200h_real/progress.txt
```

## 常见问题

### Q: SSH连接后PyTorch找不到CUDA？
A: 需要重新加载 `~/.bashrc`：
```bash
source ~/.bashrc
```

### Q: 训练脚本报 `ImportError: cannot import name 'GradScaler' from 'torch.amp'`？
A: 需要更新训练脚本为兼容版本，参考本地 `d:\NeRD-Rain\FADformer\train_rain200h.py`

### Q: 如何停止训练？
A:
```bash
# 找到训练进程PID
ps aux | grep python3
# 停止进程
kill -9 <PID>
```
