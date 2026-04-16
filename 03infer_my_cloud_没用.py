import os
import torch
import numpy as np
import open3d as o3d
#将最优模型用在自己的点云上
# ------------------- 你的配置 -------------------
YOUR_POINT_CLOUD_PATH = "/home/pc/yjy/USIP-master/mytestdatas/5_5_0_result.pcd"
MODEL_PATH = "/home/pc/yjy/USIP-master/modelnet/checkpoints/5000-64-k1k9-3d/best.pth"
SAVE_KEYPOINTS_PATH = "/home/pc/yjy/USIP-master/mytestdatas/keypoints_result.pcd"
# -------------------------------------------------

# 配置
from modelnet.options_detector import Options
opt = Options().parse()
opt.gpu_ids = [0]
opt.input_pc_num = 5000
opt.node_num = 64
opt.node_knn_k_1 = 32
opt.k = 9
opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型
from models.keypoint_detector import ModelDetector
model = ModelDetector(opt)
model.detector.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.freeze_model()

# ===================== 读取 完整原始点云 =====================
pcd_raw = o3d.io.read_point_cloud(YOUR_POINT_CLOUD_PATH)
points_full = np.asarray(pcd_raw.points).astype(np.float32)  # 完整点云（后面可视化用）

# ===================== 模型输入：采样到5000 =====================
points = points_full.copy()
N = 5000
if len(points) > N:
    points = points[np.random.choice(len(points), N, replace=False)]
else:
    points = np.pad(points, [(0, N-len(points)), (0,0)], mode="wrap")

# 构造输入
pc = torch.from_numpy(points).unsqueeze(0).to(opt.device)
sn = torch.zeros_like(pc).to(opt.device)

# 生成 node
pc_in = pc.permute(0, 2, 1)
sn_in = sn.permute(0, 2, 1)
node = pc_in[:, :, np.random.choice(N, opt.node_num, replace=False)]

# 推理
with torch.no_grad():
    keypoints, sigmas = model.run_model(pc_in, sn_in, node)

# 保存关键点
kp = keypoints[0].permute(1, 0).cpu().numpy()
out_pcd = o3d.geometry.PointCloud()
out_pcd.points = o3d.utility.Vector3dVector(kp)
o3d.io.write_point_cloud(SAVE_KEYPOINTS_PATH, out_pcd)

print("✅ 关键点保存成功：", SAVE_KEYPOINTS_PATH)

# ==============================================
# ============ 🔥 可视化：完整原始点云 ==============
# ==============================================
print("正在打开 3D 可视化窗口...")

# 1. 完整原始点云 → 蓝色（不采样！）
pcd_original = o3d.geometry.PointCloud()
pcd_original.points = o3d.utility.Vector3dVector(points_full)
pcd_original.paint_uniform_color([0, 0, 1])

# 2. 关键点 → 红色
pcd_keypoint = o3d.geometry.PointCloud()
pcd_keypoint.points = o3d.utility.Vector3dVector(kp)
pcd_keypoint.paint_uniform_color([1, 0, 0])

# 3. 显示
o3d.visualization.draw_geometries(
    [pcd_original, pcd_keypoint],
    window_name="USIP 关键点检测结果（完整原始点云）",
    width=1200,
    height=900
)
