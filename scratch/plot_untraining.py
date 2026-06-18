import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for academic/professional look
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

epochs = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
loss = np.array([0.1125, 0.1186, 0.1473, 0.1546, 0.1521, 0.1594, 0.1037, 0.0995, 0.0700, 0.0821, 0.0843, 0.0830, 0.0770, 0.0696])
psnr = np.array([14.89, 16.12, 13.47, 13.45, 14.10, 13.07, 17.38, 17.49, 19.69, 16.96, 18.04, 13.74, 17.81, 16.74])

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot training loss on left axis
color = '#d62728'
ax1.set_xlabel('训练轮数 (Epoch)', fontsize=12, fontweight='bold')
ax1.set_ylabel('训练损失 (L1/SSIM/FFT Loss)', color=color, fontsize=12, fontweight='bold')
line1 = ax1.plot(epochs, loss, color=color, marker='o', linestyle='-', linewidth=2.5, label='训练损失 (Loss)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.05, 0.18)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  
color = '#1f77b4'
ax2.set_ylabel('验证集 PSNR (dB)', color=color, fontsize=12, fontweight='bold')
line2 = ax2.plot(epochs, psnr, color=color, marker='s', linestyle='--', linewidth=2.0, label='验证集 PSNR (dB)')
# Draw baseline line
line3 = ax2.axhline(y=18.52, color='#2ca02c', linestyle=':', linewidth=2.0, label='基线模型 PSNR (18.52 dB)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(10, 22)

# Add annotations to explain the un-training effect
ax1.annotate('加载预训练权重\n但新模块随机初始化', xy=(1, 0.1125), xytext=(15, 0.10),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6))

ax1.annotate('梯度冲刷造成\n表征崩溃 (Loss 陡增)', xy=(30, 0.1546), xytext=(35, 0.165),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6))

ax1.annotate('缓慢重新收敛\n但 130 轮仍未达基线', xy=(130, 0.0696), xytext=(75, 0.06),
             arrowprops=dict(facecolor='orange', shrink=0.08, width=1.5, headwidth=6))

# Combine legends
lines = line1 + line2 + [line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

plt.title('HGM-Full 模型“解训练效应 (Un-training Effect)”收敛曲线图', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

# Save the plot
os.makedirs('figs', exist_ok=True)
save_path = 'figs/untraining_loss.png'
plt.savefig(save_path, dpi=200)
plt.close()
print(f"Plot successfully saved to: {os.path.abspath(save_path)}")
