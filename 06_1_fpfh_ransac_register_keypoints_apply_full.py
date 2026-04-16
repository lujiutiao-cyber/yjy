# -*- coding: utf-8 -*-
"""
钢包 / 一般刚体：用 FPFH + RANSAC 粗配准，再 ICP 细化；变换作用到整片点云并导出 PCD + PNG。

重要（为何只在关键点上会失败）：
  - FPFH 需要「每个点周围一片邻域」的几何统计；USIP 关键点只有稀疏几十～几百点，
    法向/邻域不可靠，RANSAC 极易落到错误解 —— 视觉上两团点云各转各的。
  - 文件名 5_5_0 与 15_15_45 通常表示 **平面位置 (x,y) 也不同**，不只是绕轴转 45°，
    真实变换是「旋转 + 平移」，必须用完整点云或好的初值。

默认在 **整片点云**（pcafilter）上下采样后做 FPFH+RANSAC；可用 --register-on keypoints 恢复旧行为（不推荐）。

  python fpfh_ransac_register_keypoints_apply_full.py
  python fpfh_ransac_register_keypoints_apply_full.py --yaw-deg 45 --axis z
  python fpfh_ransac_register_keypoints_apply_full.py --voxel-mm 0.02

依赖: open3d, numpy, matplotlib
"""
import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("需要安装 open3d: pip install open3d")
    sys.exit(1)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _bbox_diagonal(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    return float(np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound()))


def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel_size: float):
    if len(pcd.points) == 0:
        raise ValueError("空点云")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) < 50:
        pcd_down = pcd
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5.0
    if hasattr(o3d.pipelines.registration, "compute_fpfh_feature"):
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
    else:
        fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
    return pcd_down, fpfh


def register_fpfh_ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    est = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        if hasattr(o3d, "pipelines")
        else o3d.registration.TransformationEstimationPointToPoint(False)
    )
    criteria = (
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=100000, confidence=0.999
        )
        if hasattr(o3d, "pipelines")
        else o3d.registration.RANSACConvergenceCriteria(100000, 100000, 0.999)
    )
    checkers = []
    if hasattr(o3d.pipelines.registration, "CorrespondenceCheckerBasedOnEdgeLength"):
        checkers = [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ]
    if hasattr(o3d.pipelines.registration, "registration_ransac_based_on_feature_matching"):
        try:
            return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                True,
                distance_threshold,
                est,
                3,
                checkers,
                criteria,
            )
        except TypeError:
            pass
    if hasattr(o3d, "registration"):
        return o3d.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            distance_threshold,
            est,
            3,
            checkers,
            criteria,
        )
    raise RuntimeError("当前 Open3D 不支持 RANSAC 特征配准")


def T_centroid_yaw_row(
    xyz_src: np.ndarray, xyz_tgt: np.ndarray, yaw_deg: float, axis: str
) -> np.ndarray:
    """行向量点云：p' = p @ R^T + t，使源质心经绕 axis 旋转后与目标质心贴合（粗初值）。"""
    cs = xyz_src.mean(axis=0)
    ct = xyz_tgt.mean(axis=0)
    th = np.radians(yaw_deg)
    if axis == "z":
        R = np.array(
            [
                [np.cos(th), -np.sin(th), 0.0],
                [np.sin(th), np.cos(th), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif axis == "y":
        R = np.array(
            [
                [np.cos(th), 0.0, np.sin(th)],
                [0.0, 1.0, 0.0],
                [-np.sin(th), 0.0, np.cos(th)],
            ],
            dtype=np.float64,
        )
    else:
        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(th), -np.sin(th)],
                [0.0, np.sin(th), np.cos(th)],
            ],
            dtype=np.float64,
        )
    t = ct - cs @ R.T
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def icp_refine(
    source_full: o3d.geometry.PointCloud,
    target_full: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, object]:
    """点对面 ICP 细化（在 downsample 上算初值后，可对全点或再下采样）。"""
    dist = voxel_size * 2.5
    src = source_full.voxel_down_sample(voxel_size)
    tgt = target_full.voxel_down_sample(voxel_size)
    src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    if hasattr(o3d.pipelines.registration, "TransformationEstimationPointToPlane"):
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, dist, T_init, est, crit
        )
    else:
        est = o3d.registration.TransformationEstimationPointToPlane()
        crit = o3d.registration.ICPConvergenceCriteria(max_iteration=80)
        reg = o3d.registration.registration_icp(src, tgt, dist, T_init, est, crit)
    return np.asarray(reg.transformation, dtype=np.float64), reg


def apply_transform_numpy_pcd(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    N = xyz.shape[0]
    h = np.hstack([xyz.astype(np.float64), np.ones((N, 1))])
    return (h @ T.T)[:, :3]


def save_matplotlib_vis(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    out_png: str,
    max_points: int = 12000,
    seed: int = 0,
):
    if not _HAS_MPL:
        return
    rs = np.random.RandomState(seed)

    def sub(x):
        x = np.asarray(x)
        if x.shape[0] > max_points:
            i = rs.choice(x.shape[0], max_points, replace=False)
            return x[i]
        return x

    xyz_a = sub(xyz_a)
    xyz_b = sub(xyz_b)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz_a[:, 0],
        xyz_a[:, 1],
        xyz_a[:, 2],
        c="tab:blue",
        s=1,
        alpha=0.25,
        label="source (orig)",
    )
    ax.scatter(
        xyz_b[:, 0],
        xyz_b[:, 1],
        xyz_b[:, 2],
        c="tab:orange",
        s=1,
        alpha=0.25,
        label="target (T^-1)",
    )
    ax.legend(loc="upper right")
    ax.set_title("Same frame: source + target @ inv(T_final)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def save_matplotlib_vis_target_frame(
    xyz_s_aligned: np.ndarray,
    xyz_t: np.ndarray,
    out_png: str,
    max_points: int = 12000,
    seed: int = 1,
):
    if not _HAS_MPL:
        return
    rs = np.random.RandomState(seed)

    def sub(x):
        x = np.asarray(x)
        if x.shape[0] > max_points:
            i = rs.choice(x.shape[0], max_points, replace=False)
            return x[i]
        return x

    xyz_s_aligned = sub(xyz_s_aligned)
    xyz_t = sub(xyz_t)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz_s_aligned[:, 0],
        xyz_s_aligned[:, 1],
        xyz_s_aligned[:, 2],
        c="tab:blue",
        s=1,
        alpha=0.25,
        label="source (T)",
    )
    ax.scatter(
        xyz_t[:, 0], xyz_t[:, 1], xyz_t[:, 2], c="tab:orange", s=1, alpha=0.25, label="target"
    )
    ax.legend(loc="upper right")
    ax.set_title("Target frame: source aligned + target")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def write_pcd_xyz_normals(path: str, xyz: np.ndarray, normals: Optional[np.ndarray]):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if normals is not None and normals.shape[0] == xyz.shape[0]:
        pc.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    o3d.io.write_point_cloud(path, pc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--register-on",
        choices=("full", "keypoints"),
        default="full",
        help="在整片点云或关键点上算 FPFH（默认 full，关键点极易失败）",
    )
    ap.add_argument(
        "--kp-src",
        default=r"E:\pointData\denorm\4\5_5_0_result_keypoints.pcd",
        help="关键点源（仅 register-on=keypoints 时必需）",
    )
    ap.add_argument(
        "--kp-tgt",
        default=r"E:\pointData\denorm\4\15_15_45_result_keypoints.pcd",
    )
    ap.add_argument("--full-src", default=r"E:\pointData\pcafilter\4\5_5_0_result.pcd")
    ap.add_argument("--full-tgt", default=r"E:\pointData\pcafilter\4\15_15_45_result.pcd")
    ap.add_argument("--out-dir", default=r"E:\pointData\fpfh_registration\4")
    ap.add_argument(
        "--voxel-scale",
        type=float,
        default=0.04,
        help="体素 = 对角线 * 系数（full 模式）；略大更稳、略小更精细",
    )
    ap.add_argument(
        "--voxel-mm",
        type=float,
        default=None,
        help="若设置则强制体素边长为该值（米或与你点云单位一致）",
    )
    ap.add_argument(
        "--yaw-deg",
        type=float,
        default=None,
        help="已知绕轴转角（度）时提供粗初值，再与 FPFH 结果择优并 ICP",
    )
    ap.add_argument(
        "--axis",
        choices=("z", "y", "x"),
        default="z",
        help="yaw-deg 所绕轴（扫描转台多为 z）",
    )
    ap.add_argument("--skip-fpfh", action="store_true", help="仅用 yaw 初值 + ICP")
    args = ap.parse_args()

    need_kp = args.register_on == "keypoints"
    for p in (
        [args.kp_src, args.kp_tgt] if need_kp else []
    ) + [args.full_src, args.full_tgt]:
        if p and not os.path.isfile(p):
            print("文件不存在: %s" % p)
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    full_s = o3d.io.read_point_cloud(args.full_src)
    full_t = o3d.io.read_point_cloud(args.full_tgt)
    xyz_s_full = np.asarray(full_s.points)
    xyz_t_full = np.asarray(full_t.points)

    if args.register_on == "full":
        reg_s, reg_t = full_s, full_t
        diag = max(_bbox_diagonal(reg_s), _bbox_diagonal(reg_t), 1e-6)
        voxel = (
            float(args.voxel_mm)
            if args.voxel_mm is not None
            else max(diag * args.voxel_scale, diag * 0.015)
        )
    else:
        reg_s = o3d.io.read_point_cloud(args.kp_src)
        reg_t = o3d.io.read_point_cloud(args.kp_tgt)
        diag = max(_bbox_diagonal(reg_s), _bbox_diagonal(reg_t), 1e-6)
        voxel = max(diag * args.voxel_scale, diag * 0.02)
        if len(reg_s.points) < 200 or len(reg_t.points) < 200:
            voxel = max(diag * 0.01, 1e-4)
        print(
            "[警告] 在稀疏关键点上做 FPFH 极易错误；建议改用 --register-on full"
        )

    s_down, s_fpfh = preprocess_pcd(reg_s, voxel)
    t_down, t_fpfh = preprocess_pcd(reg_t, voxel)

    candidates = []

    if not args.skip_fpfh:
        result = register_fpfh_ransac(s_down, t_down, s_fpfh, t_fpfh, voxel)
        T_r = np.asarray(result.transformation, dtype=np.float64)
        candidates.append(
            ("fpfh_ransac", T_r, float(result.fitness), float(result.inlier_rmse))
        )
        print(
            "FPFH+RANSAC  fitness=%.6f  inlier_rmse=%.6f"
            % (result.fitness, result.inlier_rmse)
        )

    if args.yaw_deg is not None:
        xyz_s = np.asarray(s_down.points)
        xyz_t = np.asarray(t_down.points)
        T_y = T_centroid_yaw_row(xyz_s, xyz_t, args.yaw_deg, args.axis)
        candidates.append(("yaw_prior_centroid", T_y, -1.0, -1.0))
        print(
            "已加入转角先验: %.2f deg 绕 %s（质心对齐）" % (args.yaw_deg, args.axis)
        )

    if not candidates:
        print("请去掉 --skip-fpfh 或提供 --yaw-deg")
        sys.exit(1)

    best_T = None
    best_name = None
    best_score = -1.0
    for name, T0, fit, _ in candidates:
        T_icp, icp = icp_refine(full_s, full_t, T0, voxel)
        score = float(icp.fitness)
        rmse = getattr(icp, "inlier_rmse", float("nan"))
        print("ICP refine from %-22s  fitness=%.6f  rmse=%s" % (name, score, rmse))
        if score > best_score:
            best_score = score
            best_T = T_icp
            best_name = name + "+icp"

    T = best_T
    T_inv = np.linalg.inv(T)

    np.save(os.path.join(args.out_dir, "final_T_src_to_tgt.npy"), T)
    np.save(os.path.join(args.out_dir, "final_T_inv_tgt_to_src.npy"), T_inv)
    meta_path = os.path.join(args.out_dir, "registration_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("chosen_init: %s\n" % best_name)
        f.write("icp_fitness_after_refine: %.8f\n" % best_score)
        f.write("register_on: %s\n" % args.register_on)
        f.write("voxel_size_used: %.8f\n" % voxel)

    print("采用: %s  （最终 ICP fitness=%.6f）" % (best_name, best_score))

    n_s = np.asarray(full_s.normals) if full_s.has_normals() else None
    n_t = np.asarray(full_t.normals) if full_t.has_normals() else None

    xyz_s_in_tgt = apply_transform_numpy_pcd(xyz_s_full, T)
    R = T[:3, :3]
    n_s_in_tgt = (n_s @ R.T) if n_s is not None else None

    xyz_t_in_src = apply_transform_numpy_pcd(xyz_t_full, T_inv)
    R_inv = T_inv[:3, :3]
    n_t_in_src = (n_t @ R_inv.T) if n_t is not None else None

    out_src_orig = os.path.join(args.out_dir, "5_5_0_source_original.pcd")
    out_tgt_xform = os.path.join(
        args.out_dir, "15_15_45_target_transformed_to_source_frame.pcd"
    )
    out_src_al = os.path.join(args.out_dir, "5_5_0_source_aligned_to_target_frame.pcd")
    out_tgt_ref = os.path.join(args.out_dir, "15_15_45_target_reference.pcd")

    write_pcd_xyz_normals(out_src_orig, xyz_s_full, n_s)
    write_pcd_xyz_normals(out_tgt_xform, xyz_t_in_src, n_t_in_src)
    write_pcd_xyz_normals(out_src_al, xyz_s_in_tgt, n_s_in_tgt)
    write_pcd_xyz_normals(out_tgt_ref, xyz_t_full, n_t)

    png1 = os.path.join(args.out_dir, "vis_source_frame_blue_src_orange_tgt_invT.png")
    save_matplotlib_vis(xyz_s_full, xyz_t_in_src, png1)
    png2 = os.path.join(args.out_dir, "vis_target_frame_blue_src_T_orange_tgt.png")
    save_matplotlib_vis_target_frame(xyz_s_in_tgt, xyz_t_full, png2)

    print("已写出 PCD 与 PNG 到: %s" % args.out_dir)

    try:
        a = o3d.geometry.PointCloud()
        a.points = o3d.utility.Vector3dVector(xyz_s_full)
        a.paint_uniform_color([0.1, 0.35, 0.95])
        b = o3d.geometry.PointCloud()
        b.points = o3d.utility.Vector3dVector(xyz_t_in_src)
        b.paint_uniform_color([0.95, 0.45, 0.1])
        o3d.visualization.draw_geometries(
            [a, b],
            window_name="Refined: blue=source, orange=target@inv(T)",
            width=1280,
            height=720,
        )
    except Exception as ex:
        print("Open3D 交互窗口: %s" % ex)


if __name__ == "__main__":
    main()
