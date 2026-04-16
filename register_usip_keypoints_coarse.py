# -*- coding: utf-8 -*-
"""
USIP 稀疏关键点粗配准（不依赖 FPFH 邻域）：

1) 推荐（若你有每点描述子 MxD，例如自研/其它分支导出）:
   --desc-src / --desc-tgt 与关键点行一一对应，L2 归一化后做互最近邻（特征空间），
   再 Open3D RANSAC 估计刚性 T。这比「纯 3D 距离」在视角变化时稳得多。

2) 默认（仅坐标）:
   3D 欧氏互最近邻 + RANSAC（与 keypoints_denorm 里思路一致）。点少、对称物体时仍可能错。

3) 工程上常更稳: 整云下采样 + Super4PCS / FPFH+RANSAC + ICP；转台场景加转角初值。

说明: 当前仓库里 ModelDetector 的 detector 前向在多数配置下 descriptors=None，
      批量脚本若未另存描述子，只能用 (2)。可把 sigmas 当作弱权重（本脚本暂未加权）。

用法示例:
  python register_usip_keypoints_coarse.py \\
    --src-pcd E:/pointData/denorm/4/5_5_0_result_keypoints.pcd \\
    --tgt-pcd E:/pointData/denorm/4/15_15_45_result_keypoints.pcd \\
    --out-json E:/pointData/reg_T.json

  # 若有描述子:
  python register_usip_keypoints_coarse.py --src-pcd ... --tgt-pcd ... \\
    --desc-src E:/path/5_5_0_desc.npy --desc-tgt E:/path/15_15_45_desc.npy
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("需要 open3d")
    sys.exit(1)


def mutual_nn_from_descriptors(
    desc_src: np.ndarray, desc_tgt: np.ndarray
) -> np.ndarray:
    """返回 (K,2) 整数对应 src_idx, tgt_idx（互最近邻，余弦距离）。"""
    ds = desc_src.astype(np.float64)
    dt = desc_tgt.astype(np.float64)
    ds = ds / (np.linalg.norm(ds, axis=1, keepdims=True) + 1e-9)
    dt = dt / (np.linalg.norm(dt, axis=1, keepdims=True) + 1e-9)
    sim = ds @ dt.T
    j_for_i = sim.argmax(axis=1)
    i_for_j = sim.argmax(axis=0)
    pairs = []
    for i in range(ds.shape[0]):
        j = int(j_for_i[i])
        if int(i_for_j[j]) == i:
            pairs.append((i, j))
    if len(pairs) < 4:
        for i in range(ds.shape[0]):
            pairs.append((i, int(j_for_i[i])))
    return np.asarray(pairs, dtype=np.int32)


def mutual_nn_from_xyz(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """3D 欧氏互最近邻（Open3D KD）。"""
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    s_pcd = o3d.geometry.PointCloud()
    s_pcd.points = o3d.utility.Vector3dVector(src)
    t_pcd = o3d.geometry.PointCloud()
    t_pcd.points = o3d.utility.Vector3dVector(tgt)
    ts = o3d.geometry.KDTreeFlann(t_pcd)
    ss = o3d.geometry.KDTreeFlann(s_pcd)
    idx_s = np.empty(src.shape[0], dtype=np.int64)
    for i in range(src.shape[0]):
        _, ids, _ = ts.search_knn_vector_3d(src[i], 1)
        idx_s[i] = ids[0]
    idx_t = np.empty(tgt.shape[0], dtype=np.int64)
    for j in range(tgt.shape[0]):
        _, ids, _ = ss.search_knn_vector_3d(tgt[j], 1)
        idx_t[j] = ids[0]
    pairs = []
    for i in range(src.shape[0]):
        j = int(idx_s[i])
        if int(idx_t[j]) == i:
            pairs.append((i, j))
    if len(pairs) < 4:
        for i in range(src.shape[0]):
            pairs.append((i, int(idx_s[i])))
    return np.asarray(pairs, dtype=np.int32)


def ransac_from_pairs(
    src_xyz: np.ndarray,
    tgt_xyz: np.ndarray,
    pairs: np.ndarray,
    distance_factor: float = 2.5,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if pairs.shape[0] < 3:
        return None, {"error": "对应点不足"}
    dists = np.linalg.norm(
        src_xyz[pairs[:, 0]] - tgt_xyz[pairs[:, 1]], axis=1
    )
    thresh = max(1e-6, float(np.median(dists)) * distance_factor)
    s_pcd = o3d.geometry.PointCloud()
    s_pcd.points = o3d.utility.Vector3dVector(src_xyz.astype(np.float64))
    t_pcd = o3d.geometry.PointCloud()
    t_pcd.points = o3d.utility.Vector3dVector(tgt_xyz.astype(np.float64))
    corres = o3d.utility.Vector2iVector(pairs)
    if hasattr(o3d, "pipelines"):
        reg = o3d.pipelines.registration
        est = reg.TransformationEstimationPointToPoint(False)
        crit = reg.RANSACConvergenceCriteria(
            max_iteration=100000, confidence=0.999
        )
        res = reg.registration_ransac_based_on_correspondence(
            s_pcd, t_pcd, corres, thresh, est, 3, crit
        )
    else:
        reg = o3d.registration
        est = reg.TransformationEstimationPointToPoint(False)
        crit = reg.RANSACConvergenceCriteria(100000, 100000, 0.999)
        res = reg.registration_ransac_based_on_correspondence(
            s_pcd,
            t_pcd,
            corres,
            max_correspondence_distance=thresh,
            estimation_method=est,
            ransac_n=3,
            criteria=crit,
        )
    T = np.asarray(res.transformation, dtype=np.float64)
    meta = {
        "fitness": float(res.fitness),
        "inlier_rmse": float(res.inlier_rmse),
        "correspondence_count": int(pairs.shape[0]),
        "distance_threshold": thresh,
    }
    return T, meta


def load_xyz(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        a = np.load(path)
        if a.ndim != 2 or a.shape[1] < 3:
            raise ValueError("npy 需至少 Nx3")
        return a[:, :3].astype(np.float64)
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description="USIP 关键点粗配准（描述子优先 / 否则 3D 互近邻）")
    ap.add_argument("--src-pcd", required=True)
    ap.add_argument("--tgt-pcd", required=True)
    ap.add_argument("--desc-src", default=None, help="源描述子 NxD（与关键点同序）")
    ap.add_argument("--desc-tgt", default=None, help="目标描述子 MxD")
    ap.add_argument("--out-json", default="usip_keypoints_coarse_T.json")
    args = ap.parse_args()

    xyz_s = load_xyz(args.src_pcd)
    xyz_t = load_xyz(args.tgt_pcd)

    mode = "xyz_mutual_nn"
    if args.desc_src and args.desc_tgt:
        if not os.path.isfile(args.desc_src) or not os.path.isfile(args.desc_tgt):
            print("描述子文件不存在")
            sys.exit(1)
        ds = np.load(args.desc_src)
        dt = np.load(args.desc_tgt)
        if ds.shape[0] != xyz_s.shape[0] or dt.shape[0] != xyz_t.shape[0]:
            print(
                "描述子行数须与关键点一致: src %d vs %d, tgt %d vs %d"
                % (ds.shape[0], xyz_s.shape[0], dt.shape[0], xyz_t.shape[0])
            )
            sys.exit(1)
        pairs = mutual_nn_from_descriptors(ds, dt)
        mode = "descriptor_mutual_nn"
    else:
        pairs = mutual_nn_from_xyz(xyz_s, xyz_t)

    T, meta = ransac_from_pairs(xyz_s, xyz_t, pairs)
    out = {
        "mode": mode,
        "source": args.src_pcd,
        "target": args.tgt_pcd,
        "transformation_row_major": T.tolist() if T is not None else None,
        **meta,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("已写入 %s  mode=%s" % (args.out_json, mode))
    if T is not None:
        print("fitness=%s inlier_rmse=%s" % (meta.get("fitness"), meta.get("inlier_rmse")))
    else:
        print("失败: %s" % meta)


if __name__ == "__main__":
    main()
