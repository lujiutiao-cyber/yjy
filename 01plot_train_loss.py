"""
plot_train_loss.py — 从 checkpoint 文件名解析各 epoch 的 test loss，画曲线并导出论文用图。

输入
  - checkpoint_dir（默认 ./checkpoints）内 gpu0_<epoch>_<loss>_net_detector.pth

输出
  - test_loss_curve.pdf、test_loss_curve.png（当前目录，300 DPI）
  - 可选弹出 matplotlib 窗口（plt.show）

运行：python plot_train_loss.py；图题/图例为 Test Loss，与文件名中的验证指标一致。
"""
import os
import re
import matplotlib.pyplot as plt
# ===================== 论文绘图配置（非常重要）=====================
plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Sans']  # 自动 fallback
#plt.rcParams['font.family'] = 'Times New Roman'  # 论文标准字体
plt.rcParams['figure.dpi'] = 300                  # 高清 300DPI
plt.rcParams['axes.linewidth'] = 1.2              # 坐标轴线粗细
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# 模型路径
checkpoint_dir = "./checkpoints"

# 提取 epoch 和 loss
epochs = []
losses = []

for filename in os.listdir(checkpoint_dir):
    if filename.endswith(".pth"):
        match = re.search(r"gpu0_(\d+)_(-?\d+\.\d+)_net_detector.pth", filename)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs.append(epoch)
            losses.append(loss)

# 按 epoch 排序
sorted_data = sorted(zip(epochs, losses))
epochs, losses = zip(*sorted_data)

# ===================== 绘图（论文版，无网格）=====================
plt.figure(figsize=(8, 5))

# 专业论文线条：蓝色实线，粗细适中
plt.plot(epochs, losses, color='#1f77b4', linewidth=1.8, label='Test Loss')

# 标签（论文必须规范）
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('USIP Detector Test Loss Curve', fontsize=16, pad=12)

# 图例
plt.legend(loc='best', fontsize=12, frameon=True)

# ===================== 关键：去掉网格 =====================
# 这里默认就是不显示网格，所以不用额外操作

# 布局自适应
plt.tight_layout()

# 保存高清图片（可直接插入论文）
plt.savefig('test_loss_curve.pdf', dpi=300, bbox_inches='tight')
plt.savefig('test_loss_curve.png', dpi=300, bbox_inches='tight')

plt.show()
