# -*- coding: utf-8 -*-
"""
在 denorm 下两幅关键点 PCD 上做粗配准（互最近邻 + Open3D RANSAC），
输出旋转矩阵 R、平移向量 t（与 Open3D 4x4 齐次阵一致：列向量 p' = R @ p + t），
再将变换作用到 pcafilter 下整片 PCD，单图叠加：蓝=配准前源、红=目标、橙=配准后源；Open3D 单窗口同三色。

默认路径（可改参数）:
  关键点: E:\\pointData\\denorm\\4\\5_5_0_result_keypoints.pcd 与 15_15_45_result_keypoints.pcd
  整云:   E:\\pointData\\pcafilter\\4\\5_5_0_result.pcd 与 15_15_45_result.pcd

  python coarse_register_keypoints_vis_full.py

依赖: open3d, numpy, matplotlib
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("需要 open3d: pip install open3d")
    sys.exit(1)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    print("需要 matplotlib")
    sys.exit(1)


def _knn_indices(query_xyz: np.ndarray, pcd_tree_cloud: o3d.geometry.PointCloud) -> np.ndarray:
    tree = o3d.geometry.KDTreeFlann(pcd_tree_cloud)
    idx = np.empty(query_xyz.shape[0], dtype=np.int64)
    for i in range(query_xyz.shape[0]):
        _, ids, _ = tree.search_knn_vector_3d(query_xyz[i], 1)
        idx[i] = ids[0]
    return idx


def mutual_nearest_neighbor_correspondences(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    s_pcd = o3d.geometry.PointCloud()
    s_pcd.points = o3d.utility.Vector3dVector(src)
    d_pcd = o3d.geometry.PointCloud()
    d_pcd.points = o3d.utility.Vector3dVector(dst)
    idx_s = _knn_indices(src, d_pcd)
    idx_t = _knn_indices(dst, s_pcd)
    pairs = []
    for i in range(src.shape[0]):
        j = int(idx_s[i])
        if int(idx_t[j]) == i:
            pairs.append((i, j))
    if len(pairs) < 3:
        pairs = [(i, int(idx_s[i])) for i in range(src.shape[0])]
    return np.asarray(pairs, dtype=np.int32)


def register_ransac_3d3d(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    distance_factor: float = 2.5,
    ransac_n: int = 3,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Open3D: transform 作用在 source，使对齐到 target。"""
    src_xyz = np.asarray(src_xyz, dtype=np.float64)
    dst_xyz = np.asarray(dst_xyz, dtype=np.float64)
    pairs = mutual_nearest_neighbor_correspondences(src_xyz, dst_xyz)
    if pairs.shape[0] < ransac_n:
        return None, {"error": "对应点不足"}

    dists = np.linalg.norm(
        src_xyz[pairs[:, 0]] - dst_xyz[pairs[:, 1]], axis=1
    )
    thresh = max(1e-6, float(np.median(dists)) * distance_factor)

    s_pcd = o3d.geometry.PointCloud()
    s_pcd.points = o3d.utility.Vector3dVector(src_xyz)
    t_pcd = o3d.geometry.PointCloud()
    t_pcd.points = o3d.utility.Vector3dVector(dst_xyz)
    corres = o3d.utility.Vector2iVector(pairs)

    if hasattr(o3d, "pipelines"):
        reg = o3d.pipelines.registration
        est = reg.TransformationEstimationPointToPoint(False)
        criteria = reg.RANSACConvergenceCriteria(
            max_iteration=100000, confidence=0.999
        )
        # ransac_n 之后是 checkers（列表），再是 criteria；勿把 criteria 当第 7 个位置参数
        result = reg.registration_ransac_based_on_correspondence(
            s_pcd,
            t_pcd,
            corres,
            thresh,
            est,
            ransac_n,
            checkers=[],
            criteria=criteria,
        )
    else:
        reg = o3d.registration
        est = reg.TransformationEstimationPointToPoint(False)
        criteria = reg.RANSACConvergenceCriteria(100000, 100000, 0.999)
        result = reg.registration_ransac_based_on_correspondence(
            s_pcd,
            t_pcd,
            corres,
            max_correspondence_distance=thresh,
            estimation_method=est,
            ransac_n=ransac_n,
            criteria=criteria,
        )

    T = np.asarray(result.transformation, dtype=np.float64)
    meta = {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "correspondence_count": int(pairs.shape[0]),
        "distance_threshold": float(thresh),
    }
    return T, meta


def Rt_from_T_open3d(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Open3D 齐次矩阵：对列向量 p_hom，p'_hom = T @ p_hom。
    R = T[0:3,0:3], t = T[0:3,3] 满足 p' = R @ p + t（p 为列向量 3x1）。
    """
    R = np.asarray(T[0:3, 0:3], dtype=np.float64)
    t = np.asarray(T[0:3, 3], dtype=np.float64).reshape(3)
    return R, t


def apply_T_row_points(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """行向量点 Nx3：与 Open3D transform 一致，p_new = (p_hom @ T.T)[:,:3]"""
    N = xyz.shape[0]
    h = np.hstack([xyz.astype(np.float64), np.ones((N, 1))])
    return (h @ T.T)[:, :3]


def _subsample_pair_same_src_idx(
    xyz_src: np.ndarray,
    xyz_src_transformed: np.ndarray,
    xyz_tgt: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """source 变换前后用同一组索引下采样，便于对比同一批点在变换前后的位置。"""
    rs = np.random.RandomState(seed)
    s = np.asarray(xyz_src, dtype=np.float64)
    st = np.asarray(xyz_src_transformed, dtype=np.float64)
    t = np.asarray(xyz_tgt, dtype=np.float64)
    assert s.shape == st.shape
    if s.shape[0] > max_points:
        idx = rs.choice(s.shape[0], max_points, replace=False)
        s_sub, st_sub = s[idx], st[idx]
    else:
        s_sub, st_sub = s, st
    if t.shape[0] > max_points:
        idx_t = rs.choice(t.shape[0], max_points, replace=False)
        t_sub = t[idx_t]
    else:
        t_sub = t
    return s_sub, st_sub, t_sub


def _set_3d_equal_box(ax, *pts_blocks: np.ndarray, pad: float = 1.05):
    comb = np.vstack([np.asarray(p, dtype=np.float64) for p in pts_blocks])
    lo, hi = comb.min(axis=0), comb.max(axis=0)
    c = 0.5 * (lo + hi)
    span = max(float((hi - lo).max()) * 0.5 * pad, 1e-6)
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_zlim(c[2] - span, c[2] + span)


def save_vis_triple_overlay(
    xyz_src_before: np.ndarray,
    xyz_tgt: np.ndarray,
    xyz_src_after: np.ndarray,
    out_png: str,
    max_points: int = 12000,
    seed: int = 0,
):
    """单图：蓝=配准前源，红=目标，橙=配准后源（同一坐标系叠加显示）。"""
    s_b, s_a, t_v = _subsample_pair_same_src_idx(
        xyz_src_before, xyz_src_after, xyz_tgt, max_points, seed
    )
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        s_b[:, 0],
        s_b[:, 1],
        s_b[:, 2],
        c="tab:blue",
        s=1,
        alpha=0.22,
        label="源点云（配准前）",
    )
    ax.scatter(
        t_v[:, 0],
        t_v[:, 1],
        t_v[:, 2],
        c="tab:red",
        s=1,
        alpha=0.22,
        label="目标点云",
    )
    ax.scatter(
        s_a[:, 0],
        s_a[:, 1],
        s_a[:, 2],
        c="tab:orange",
        s=1,
        alpha=0.22,
        label="源点云（配准后）",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "整云叠加：蓝=配准前源 | 红=目标 | 橙=配准后源（橙与红应接近）",
        fontsize=11,
    )
    _set_3d_equal_box(ax, s_b, t_v, s_a)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kp-src",
        default=r"E:\pointData\denorm\4\5_5_0_result_keypoints.pcd",
    )
    ap.add_argument(
        "--kp-tgt",
        default=r"E:\pointData\denorm\4\15_15_45_result_keypoints.pcd",
    )
    ap.add_argument(
        "--full-src",
        default=r"E:\pointData\pcafilter\4\5_5_0_result.pcd",
    )
    ap.add_argument(
        "--full-tgt",
        default=r"E:\pointData\pcafilter\4\15_15_45_result.pcd",
    )
    ap.add_argument(
        "--out-dir",
        default=r"E:\pointData\coarse_reg_vis\4",
    )
    args = ap.parse_args()

    for p in (args.kp_src, args.kp_tgt, args.full_src, args.full_tgt):
        if not os.path.isfile(p):
            print("文件不存在: %s" % p)
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    kp_s = o3d.io.read_point_cloud(args.kp_src)
    kp_t = o3d.io.read_point_cloud(args.kp_tgt)
    xyz_s = np.asarray(kp_s.points, dtype=np.float64)
    xyz_t = np.asarray(kp_t.points, dtype=np.float64)

    T, meta = register_ransac_3d3d(xyz_s, xyz_t)
    if T is None:
        print("配准失败: %s" % meta)
        sys.exit(1)

    R, t = Rt_from_T_open3d(T)

    np.save(os.path.join(args.out_dir, "R_3x3.npy"), R)
    np.save(os.path.join(args.out_dir, "t_3.npy"), t)
    np.savetxt(os.path.join(args.out_dir, "R_3x3.txt"), R, fmt="%.18e")
    np.savetxt(os.path.join(args.out_dir, "t_3x1.txt"), t.reshape(3, 1), fmt="%.18e")

    payload = {
        "source_keypoints_pcd": args.kp_src.replace("\\", "/"),
        "target_keypoints_pcd": args.kp_tgt.replace("\\", "/"),
        "convention": "列向量 p_target = R @ p_source + t；齐次阵 T 与 Open3D 一致 (4x4 row-major 存 JSON)",
        "R_row_major": R.tolist(),
        "t_xyz": t.tolist(),
        "T_4x4_row_major": T.tolist(),
        **meta,
    }
    with open(os.path.join(args.out_dir, "coarse_registration.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("R (3x3) 与 t (3,) 已写入: %s" % args.out_dir)
    print("t = [%s, %s, %s]" % (t[0], t[1], t[2]))

    full_s = o3d.io.read_point_cloud(args.full_src)
    full_t = o3d.io.read_point_cloud(args.full_tgt)
    xyz_fs = np.asarray(full_s.points, dtype=np.float64)
    xyz_ft = np.asarray(full_t.points, dtype=np.float64)

    xyz_fs_al = apply_T_row_points(xyz_fs, T)

    vis_path = os.path.join(
        args.out_dir, "vis_full_triple_blue_src_red_tgt_orange_aligned.png"
    )
    save_vis_triple_overlay(xyz_fs, xyz_ft, xyz_fs_al, vis_path)
    print("可视化已保存（蓝/红/橙 同图）: %s" % vis_path)

    try:
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(xyz_ft)
        pcd_tgt.paint_uniform_color([0.9, 0.15, 0.12])
        pcd_src_before = o3d.geometry.PointCloud()
        pcd_src_before.points = o3d.utility.Vector3dVector(xyz_fs)
        pcd_src_before.paint_uniform_color([0.15, 0.35, 0.95])
        pcd_src_after = o3d.geometry.PointCloud()
        pcd_src_after.points = o3d.utility.Vector3dVector(xyz_fs_al)
        pcd_src_after.paint_uniform_color([0.98, 0.55, 0.08])
        o3d.visualization.draw_geometries(
            [pcd_src_before, pcd_tgt, pcd_src_after],
            window_name="蓝=配准前源 | 红=目标 | 橙=配准后源",
            width=1280,
            height=800,
        )
    except Exception as ex:
        print("Open3D 窗口未打开: %s" % ex)


if __name__ == "__main__":
    main()
