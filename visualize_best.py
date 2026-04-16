"""
visualize_best.py — 加载已训练 detector，对一批 .npy 点云做推理，3D 可视化点云与预测关键点。

输入
  - DESK_FOLDER：含 .npy 点云的目录（数组形状一般为 N×6：xyz + 法向；取前若干文件）
  - best_pth：detector 权重路径
  - opt / dataroot / device 等与 ModelDetector 一致（脚本内硬编码，运行前需改成你的路径）

输出
  - SAVE_FOLDER 下 desk_01.png … 等 3D 散点图（蓝点=点云，红点=关键点）

运行：在 USIP-master 根目录设置好路径后 python visualize_best.py；需 GPU 与项目依赖。
"""
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models.keypoint_detector import ModelDetector
from modelnet.options_detector import Options

# ===================== 固定路径 =====================
#/home/pc/yjy_data/modelnet40-normal_numpy/chair
DESK_FOLDER = "/home/pc/yjy_data/npy_files"
SAVE_FOLDER = "/home/pc/Desktop/my_keypoints"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ===================== 配置 =====================
opt = Options().parse()
opt.gpu_ids = [0]
opt.batch_size = 1
opt.is_train = False
opt.input_pc_num = 5000
opt.node_num = 512
opt.dataroot = "/home/pc/yjy_data/modelnet40-normal_numpy"
opt.device = torch.device('cuda:0')

# ===================== 加载模型 =====================
model = ModelDetector(opt)
best_pth = "/home/pc/yjy/USIP-master/checkpoints/gpu0_490_-7.035990_net_detector.pth"
model.detector.load_state_dict(torch.load(best_pth))
model.detector.eval()

# ===================== 取前10个desk文件 =====================
all_files = sorted([f for f in os.listdir(DESK_FOLDER) if f.endswith(".npy")])
top10_files = all_files[:10]

# ===================== 推理 + 保存图片到桌面 =====================
with torch.no_grad():
    for idx, fname in enumerate(top10_files):
        print(f"正在生成第 {idx + 1}/10 张图...")

        # 读取点云
        path = os.path.join(DESK_FOLDER, fname)
        pc = np.load(path)
        xyz = pc[:, :3].T[None, ...]
        sn = pc[:, 3:6].T[None, ...]

        # 采样节点
        np.random.seed(0)
        node_idx = np.random.choice(xyz.shape[-1], opt.node_num, replace=False)
        node = xyz[:, :, node_idx]

        # 【关键修复】全部送到 GPU
        xyz = torch.from_numpy(xyz).float().to(opt.device)
        sn = torch.from_numpy(sn).float().to(opt.device)
        node = torch.from_numpy(node).float().to(opt.device)

        # 推理关键点
        keypoints, _ = model.run_model(xyz, sn, node)

        # 转回CPU画图
        pts = xyz[0].cpu().numpy().T
        kps = keypoints[0].cpu().numpy().T

        # 3D 绘图
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, projection='3d')

        # ===================== 颜色修改在这里 =====================
        # 原始点云 → 纯蓝色 #0000FF
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='#0000FF', s=1, alpha=0.6)
        # 关键点 → 红色
        ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2], c='red', s=20, label='Keypoints')

        plt.legend()
        plt.title(f"Desk {idx + 1}")

        # 保存到桌面
        save_path = os.path.join(SAVE_FOLDER, f"desk_{idx + 1:02d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

print("\n🎉🎉🎉 全部完成！图片在桌面：desk_keypoints")
