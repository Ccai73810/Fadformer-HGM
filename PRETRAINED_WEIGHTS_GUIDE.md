# FADformer 预训练权重下载指南

## 下载链接

### 主要数据集权重
| 数据集 | 下载链接 | 提取码 |
|--------|----------|--------|
| Rain200L | https://pan.baidu.com/s/14-xuieEB4gW6VO5KHCcNFQ | cozn |
| Rain200H | https://pan.baidu.com/s/1kTEeWv6FvicdAa-m49M33A | qw5d |
| DID-Data | https://pan.baidu.com/s/12wkegevMjiQCh6yvG8dDXA | 0vcr |
| DDN-Data | https://pan.baidu.com/s/132Qz9TflresThDdjZAzvDA | c313 |
| SPA-Data | https://pan.baidu.com/s/1iHbPEjuUMVYt9do7odrtmg | 3s40 |

### 可视化结果（可选）
| 数据集 | 下载链接 | 提取码 |
|--------|----------|--------|
| Rain200L | https://pan.baidu.com/s/1rObEpOlg3Edikkc07-qRyg | ktqb |
| Rain200H | https://pan.baidu.com/s/12c3jj0a0S-6V9HsBBKtlFw | qty6 |
| DID-Data | https://pan.baidu.com/s/1waEU-SMkAfzW5QLeD9q1yA | u3ju |
| DDN-Data | https://pan.baidu.com/s/1HwsAlcMZRuzfSopGICyD5g | mqsr |
| SPA-Data | https://pan.baidu.com/s/1v26LfteVl852d1ESDJjPsw | q6hx |

## 目录结构

下载权重后，请按以下结构组织文件：

```
d:\NeRD-Rain\FADformer\
└── pretrain_weights/
    ├── rain200L/
    │   └── FADformer_Rain200L.pth
    ├── rain200H/
    │   └── FADformer_Rain200H.pth
    ├── did/
    │   └── FADformer_DID.pth
    ├── ddn/
    │   └── FADformer_DDN.pth
    └── spa/
        └── FADformer_SPA.pth
```

## 快速开始

### 1. 创建目录
在 FADformer 文件夹中创建 `pretrain_weights` 目录：

```bash
cd d:\NeRD-Rain\FADformer
mkdir pretrain_weights
```

### 2. 下载权重
1. 点击上面的链接
2. 输入提取码
3. 下载对应的 `.pth` 文件
4. 将文件放置到对应的子目录中

### 3. 验证安装
下载完成后，可以使用以下命令验证：

```python
import torch
from models.FADformer import FADformer

# 加载模型
model = FADformer()
checkpoint = torch.load('pretrain_weights/rain200H/FADformer_Rain200H.pth')
model.load_state_dict(checkpoint['state_dict'])
print("✓ 权重加载成功！")
```

## 使用预训练权重推理

使用 `predict_and_save.py` 脚本进行推理：

```bash
# 激活虚拟环境
& "d:\NeRD-Rain\NeRD-Rain\.venv\Scripts\activate.ps1"

# 运行预测
cd d:\NeRD-Rain\FADformer
python predict_and_save.py
```

## 注意事项

- 百度网盘下载可能需要百度账号
- 下载速度可能因网络情况而异
- 确保下载的文件名与代码中使用的名称一致
- 如果遇到问题，可以查看原项目的 GitHub issues

## 原项目引用

如果使用这些权重，请引用原论文：

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
