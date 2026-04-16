"""
批量：遍历 npy_files 下各子文件夹中所有 .npy（USIP 格式 N×6），用指定 checkpoint 推理关键点。

输出（在 my_keypoints 下保持相同子目录结构）
  - <stem>_viz.png          3D 散点：蓝=点云，红=关键点（有 meta 时用反归一化坐标作图）
  - <stem>_keypoints_denorm.npy  关键点反归一化 M×3（无同目录 .json meta 时跳过，仅保留归一化下的 npy）
  - <stem>_keypoints_denorm.pcd  同上，Open3D 可写时生成

路径默认对应 AutoDL 截图；根目录均为 /root 时：
  USIP_DATA_ROOT=/root/autodl-tmp/usip

运行（在 USIP-master 目录）:
  python batch_infer_npy_keypoints.py
  USIP_DATA_ROOT=/root/autodl-tmp/usip python batch_infer_npy_keypoints.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 保证以 USIP-master 为工作目录与 import 根
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

from models.keypoint_detector import ModelDetector
from modelnet.options_detector import Options

try:
    import open3d as o3d

    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False

# ===================== 路径（与 /root/autodl-tmp/usip 布局一致）=====================
USIP_DATA_ROOT = os.environ.get("USIP_DATA_ROOT", "/root/autodl-tmp/usip")
NPY_ROOT = os.path.join(USIP_DATA_ROOT, "npy_files")
OUT_ROOT = os.path.join(USIP_DATA_ROOT, "usip_keypoints")
CKPT_NAME = "gpu0_285_-7.801909_net_detector.pth"
CHECKPOINT_PATH = os.path.join(USIP_DATA_ROOT, "USIP-master", "checkpoints", CKPT_NAME)


def load_meta_json(npy_path: str):
    """同目录、同主名的 .json（pcd_to_npy 导出的 meta）。"""
    base = os.path.splitext(npy_path)[0] + ".json"
    if not os.path.isfile(base):
        return None
    with open(base, "r", encoding="utf-8") as f:
        return json.load(f)


def denorm_xyz(xyz_norm: np.ndarray, center, radius: float) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    r = float(radius)
    return (np.asarray(xyz_norm, dtype=np.float64) * r + c).astype(np.float32)


def load_state_dict_flexible(module, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(str(k).startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    module.load_state_dict(state, strict=True)


def save_keypoints_pcd(path: str, xyz_m3: np.ndarray) -> None:
    if not _HAS_O3D:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_m3.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)


def iter_npy_files(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in sorted(filenames):
            if not fname.lower().endswith(".npy"):
                continue
            if "_keypoints" in fname.lower():
                continue
            yield os.path.join(dirpath, fname)


def main() -> None:
    if not os.path.isdir(NPY_ROOT):
        print(f"错误：输入目录不存在: {NPY_ROOT}")
        sys.exit(1)
    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"错误：权重不存在: {CHECKPOINT_PATH}")
        sys.exit(1)

    os.makedirs(OUT_ROOT, exist_ok=True)

    sys.argv = [sys.argv[0]]
    opt = Options().parse()
    opt.batch_size = 1
    opt.is_train = False
    opt.input_pc_num = 5000
    opt.node_num = 512
    opt.dataroot = os.path.join(USIP_DATA_ROOT, "modelnet40-normal_numpy")
    if not os.path.isdir(opt.dataroot):
        opt.dataroot = os.path.join(USIP_DATA_ROOT, "USIP-master", "modelnet40-normal_numpy")

    model = ModelDetector(opt)
    load_state_dict_flexible(model.detector, CHECKPOINT_PATH, opt.device)
    model.detector.eval()

    npy_list = list(iter_npy_files(NPY_ROOT))
    if not npy_list:
        print(f"未在 {NPY_ROOT} 下找到 .npy 文件。")
        sys.exit(0)

    print(f"共 {len(npy_list)} 个 npy，输出根目录: {OUT_ROOT}")

    with torch.no_grad():
        for i, npy_path in enumerate(npy_list, 1):
            rel = os.path.relpath(npy_path, NPY_ROOT)
            sub_rel = os.path.dirname(rel)
            stem = os.path.splitext(os.path.basename(npy_path))[0]
            out_dir = os.path.join(OUT_ROOT, sub_rel)
            os.makedirs(out_dir, exist_ok=True)

            pc = np.load(npy_path)
            if pc.ndim != 2 or pc.shape[1] < 3:
                print(f"[跳过] 形状异常: {npy_path} -> {pc.shape}")
                continue
            xyz = pc[:, :3].T[None, ...]
            sn = pc[:, 3:6].T[None, ...] if pc.shape[1] >= 6 else np.zeros_like(xyz)

            np.random.seed(0)
            n_pts = xyz.shape[-1]
            if n_pts < opt.node_num:
                print(f"[跳过] 点数不足 {opt.node_num}: {npy_path} (N={n_pts})")
                continue
            node_idx = np.random.choice(n_pts, opt.node_num, replace=False)
            node = xyz[:, :, node_idx]

            xyz_t = torch.from_numpy(xyz).float().to(opt.device)
            sn_t = torch.from_numpy(sn).float().to(opt.device)
            node_t = torch.from_numpy(node).float().to(opt.device)

            keypoints, _ = model.run_model(xyz_t, sn_t, node_t)

            pts_norm = xyz_t[0].cpu().numpy().T
            kps_norm = keypoints[0].cpu().numpy().T

            meta = load_meta_json(npy_path)
            if meta and meta.get("normalized") and "center" in meta and "radius" in meta:
                pts_vis = denorm_xyz(pts_norm, meta["center"], meta["radius"])
                kps_denorm = denorm_xyz(kps_norm, meta["center"], meta["radius"])
                np.save(os.path.join(out_dir, f"{stem}_keypoints_denorm.npy"), kps_denorm)
                save_keypoints_pcd(os.path.join(out_dir, f"{stem}_keypoints_denorm.pcd"), kps_denorm)
            else:
                pts_vis = pts_norm
                kps_denorm = kps_norm
                np.save(os.path.join(out_dir, f"{stem}_keypoints_norm.npy"), kps_norm)
                print(f"[提示] 无 meta json，仅保存归一化关键点: {npy_path}")
                save_keypoints_pcd(os.path.join(out_dir, f"{stem}_keypoints_norm.pcd"), kps_norm)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pts_vis[:, 0], pts_vis[:, 1], pts_vis[:, 2], c="#0000FF", s=1, alpha=0.6)
            ax.scatter(kps_denorm[:, 0], kps_denorm[:, 1], kps_denorm[:, 2], c="red", s=20, label="Keypoints")
            ax.legend()
            ax.set_title(f"{rel}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{stem}_viz.png"), dpi=150)
            plt.close(fig)

            if i % 20 == 0 or i == len(npy_list):
                print(f"  进度 {i}/{len(npy_list)}")

    print("完成。")


if __name__ == "__main__":
    main()
