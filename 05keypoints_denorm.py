# -*- coding: utf-8 -*-
"""
批量推理后处理（仅反归一化 + 汇总到 denorm/）：

1) 补全 denorm：若 my_keypoints 下仅有 *_keypoints_norm.npy，从 npy_files 镜像路径读同名 .json
   （center, radius, normalized），写出 my_keypoints 下的 *_keypoints_denorm.npy / .pcd。

2) 汇总：将已有 *_keypoints_denorm.npy 复制为 USIP_DATA_ROOT/denorm/<子目录>/<stem>_keypoints.npy
   与同名 .pcd（仅含真正反归一化后的关键点）。

环境变量：USIP_DATA_ROOT，例如 E:\\pointData

运行:
  python keypoints_denorm_and_pairwise_register.py
  python keypoints_denorm_and_pairwise_register.py --skip-denorm
  python keypoints_denorm_and_pairwise_register.py --skip-export-denorm
  python keypoints_denorm_and_pairwise_register.py --subdirs 3 4
"""
import argparse
import json
import os
import sys
from typing import List

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

try:
    import open3d as o3d

    _HAS_O3D = True
except ImportError:
    o3d = None  # type: ignore
    _HAS_O3D = False


def denorm_xyz(xyz_norm: np.ndarray, center, radius: float) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    r = float(radius)
    return (np.asarray(xyz_norm, dtype=np.float64) * r + c).astype(np.float32)


def load_meta_json_for_stem(npy_files_root: str, sub_rel: str, stem: str):
    base = os.path.join(npy_files_root, sub_rel, stem + ".json")
    if not os.path.isfile(base):
        return None
    with open(base, "r", encoding="utf-8") as f:
        return json.load(f)


def save_keypoints_pcd(path: str, xyz_m3: np.ndarray) -> None:
    if not _HAS_O3D:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_m3.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)


def export_denorm_keypoints_folder(
    my_keypoints_sub: str, denorm_root: str, sub_rel: str
) -> int:
    """仅导出存在 *_keypoints_denorm.npy 的 stem 到 denorm_root/sub_rel/（*.npy + *.pcd）。"""
    dest_dir = os.path.join(denorm_root, sub_rel)
    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    for fname in os.listdir(my_keypoints_sub):
        if not fname.endswith("_keypoints_denorm.npy"):
            continue
        stem = fname[: -len("_keypoints_denorm.npy")]
        k = np.load(os.path.join(my_keypoints_sub, fname))
        np.save(os.path.join(dest_dir, "%s_keypoints.npy" % stem), k)
        save_keypoints_pcd(os.path.join(dest_dir, "%s_keypoints.pcd" % stem), k)
        n += 1
    return n


def process_denorm_for_subdir(npy_root: str, out_root: str, sub_rel: str) -> int:
    out_dir = os.path.join(out_root, sub_rel)
    if not os.path.isdir(out_dir):
        return 0
    n_written = 0
    for fname in os.listdir(out_dir):
        if not fname.endswith("_keypoints_norm.npy"):
            continue
        stem = fname[: -len("_keypoints_norm.npy")]
        den_path = os.path.join(out_dir, "%s_keypoints_denorm.npy" % stem)
        if os.path.isfile(den_path):
            continue
        meta = load_meta_json_for_stem(npy_root, sub_rel, stem)
        if not meta or not meta.get("normalized") or "center" not in meta or "radius" not in meta:
            print(
                "[反归一化跳过] 无有效 meta: %s"
                % os.path.join(sub_rel, stem).replace("\\", "/")
            )
            continue
        k_norm = np.load(os.path.join(out_dir, fname))
        k_den = denorm_xyz(k_norm, meta["center"], meta["radius"])
        np.save(den_path, k_den)
        save_keypoints_pcd(
            os.path.join(out_dir, "%s_keypoints_denorm.pcd" % stem), k_den
        )
        n_written += 1
    return n_written


def list_subdirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )


def main():
    ap = argparse.ArgumentParser(
        description="关键点反归一化补全 + 汇总到 denorm/"
    )
    ap.add_argument(
        "--usip-data-root",
        default=os.environ.get("USIP_DATA_ROOT", "/root/autodl-tmp/usip"),
        help="数据根目录（其下含 npy_files 与 my_keypoints）",
    )
    ap.add_argument(
        "--skip-denorm",
        action="store_true",
        help="不补全 my_keypoints 下的 denorm，仅导出到 denorm/",
    )
    ap.add_argument(
        "--skip-export-denorm",
        action="store_true",
        help="不写入 denorm/，仅补全 my_keypoints",
    )
    ap.add_argument(
        "--subdirs",
        nargs="*",
        default=None,
        help="只处理这些子文件夹名；默认处理 my_keypoints 下全部一级子目录",
    )
    args = ap.parse_args()

    npy_root = os.path.join(args.usip_data_root, "npy_files")
    out_root = os.path.join(args.usip_data_root, "my_keypoints")
    denorm_root = os.path.join(args.usip_data_root, "denorm")

    if not os.path.isdir(out_root):
        print("错误：输出目录不存在: %s" % out_root)
        sys.exit(1)

    os.makedirs(denorm_root, exist_ok=True)

    if args.subdirs:
        subdirs = args.subdirs
    else:
        subdirs = list_subdirs(out_root)

    total_den = 0
    total_export = 0

    for sub in subdirs:
        sub_path = os.path.join(out_root, sub)
        if not os.path.isdir(sub_path):
            print("[跳过] 非目录: %s" % sub_path)
            continue
        sub_rel = sub
        if not args.skip_denorm:
            if not os.path.isdir(npy_root):
                print("警告：npy_files 不存在，无法补反归一化: %s" % npy_root)
            else:
                total_den += process_denorm_for_subdir(npy_root, out_root, sub_rel)

        if not args.skip_export_denorm:
            total_export += export_denorm_keypoints_folder(
                sub_path, denorm_root, sub_rel
            )

    if not args.skip_denorm:
        print("反归一化新写入 my_keypoints: %d 个 *_keypoints_denorm.npy" % total_den)
    if not args.skip_export_denorm:
        print(
            "denorm 汇总关键点: %d 个 stem（根目录: %s）" % (total_export, denorm_root)
        )


if __name__ == "__main__":
    main()
