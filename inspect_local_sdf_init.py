#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import pypose as pp
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent
GS_SEARCH_LIB = REPO_ROOT / "modules" / "gaussian_search" / "build" / "libgs_search.so"
SPARSE_HASH_LIB = REPO_ROOT / "modules" / "sparse_hash" / "build" / "libsvh.so"


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


_require_file(GS_SEARCH_LIB, "Gaussian search extension")
_require_file(SPARSE_HASH_LIB, "Sparse hash extension")

# local_sdf.py loads the Gaussian search extension through a relative path.
os.chdir(REPO_ROOT)

import dataset  # noqa: E402
import main_util  # noqa: E402
from local_sdf import LocalSDF  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build only the SLNR hash grid and local SDF anchors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--conf",
        type=str,
        default="configs/example.yaml",
        help="Path to the SLNR yaml config.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output directory. Defaults to Dataset.out_dir from the config.",
    )
    parser.add_argument(
        "--skip-load",
        type=int,
        default=None,
        help="Override HashTable.skip_load for quicker testing.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap the number of frames used to build the local SDFs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for frame and point subsampling.",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=256,
        help="Number of initialized anchors to use in the query smoke test.",
    )
    parser.add_argument(
        "--no-query-smoke",
        action="store_true",
        help="Skip the post-build LocalSDF.query_feature smoke test.",
    )
    return parser.parse_args()


def resolve_config_path(config_path: str) -> Path:
    # Accept either an absolute config path or one relative to the repo root.
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_output_dir(dataset_cfg: dict, output_override: Optional[str]) -> Path:
    if output_override is None:
        out_dir = Path(dataset_cfg["out_dir"])
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    else:
        out_dir = Path(output_override)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir

    return out_dir.resolve()


def resolve_file_data_paths(cfg: dict) -> Tuple[Path, Path, Path, Optional[Path]]:
    # Mirror the path construction used by run.py so this script exercises the
    # same dataset and output layout as the normal SLNR pipeline.
    dataset_cfg = cfg["Dataset"]

    workspace = Path(dataset_cfg["workspace"])
    data_dir = Path(dataset_cfg["data_dir"])
    root_dir = (REPO_ROOT / workspace / data_dir).resolve()

    pc_dir = (root_dir / dataset_cfg["pc_dir"]).resolve()
    pose_file = (root_dir / dataset_cfg["pose_file"]).resolve()
    calib_file = None
    if "calib_file" in dataset_cfg:
        calib_file = (root_dir / dataset_cfg["calib_file"]).resolve()

    return root_dir, pc_dir, pose_file, calib_file


def build_scene_dataset(
    cfg: dict,
    output_override: Optional[str],
    max_frames_override: Optional[int],
):
    dataset_cfg = cfg["Dataset"]
    input_mode = dataset_cfg.get("input_mode", "file").strip().lower()
    out_dir = resolve_output_dir(dataset_cfg, output_override)

    data_loader_cfg = cfg["DataLoader"]
    source_info: Dict[str, object] = {
        "input_mode": input_mode,
        "dataset_root": None,
        "pointcloud_source": None,
        "pose_source": None,
        "calib_file": None,
        "details": {},
    }

    if input_mode == "file":
        root_dir, pc_dir, pose_file, calib_file = resolve_file_data_paths(cfg)

        _require_file(pc_dir, "Point cloud directory")
        _require_file(pose_file, "Pose file")
        if calib_file is not None:
            _require_file(calib_file, "Calibration file")

        scene_dataset = dataset.LiDARDataset(
            str(pc_dir) + os.sep,
            None if calib_file is None else str(calib_file),
            str(pose_file),
            data_loader_cfg["min_range"],
            data_loader_cfg["max_range"],
            data_loader_cfg["sor_nn"],
            data_loader_cfg["sor_std"],
            data_loader_cfg["use_filter"],
        )

        source_info.update(
            {
                "dataset_root": str(root_dir),
                "pointcloud_source": str(pc_dir),
                "pose_source": str(pose_file),
                "calib_file": None if calib_file is None else str(calib_file),
            }
        )
        return scene_dataset, out_dir, source_info

    if input_mode == "ros2":
        from ros2_capture_dataset import capture_ros2_dataset

        ros2_cfg = cfg.get("ROS2", {})
        calib_file = None
        if "calib_file" in dataset_cfg:
            root_dir, _, _, calib_path = resolve_file_data_paths(cfg)
            calib_file = calib_path
            _require_file(calib_file, "Calibration file")
            source_info["dataset_root"] = str(root_dir)

        requested_frame_count = max_frames_override
        if requested_frame_count is None:
            requested_frame_count = int(ros2_cfg.get("max_frames", 200))
        requested_frame_count = max(1, int(requested_frame_count))

        scene_dataset, capture_info = capture_ros2_dataset(
            pointcloud_topic=str(ros2_cfg.get("pointcloud_topic", "/velodyne_points")),
            odom_topic=str(ros2_cfg.get("odom_topic", "/integrated_to_init")),
            max_frames=requested_frame_count,
            capture_timeout_sec=float(ros2_cfg.get("capture_timeout_sec", 30.0)),
            max_odom_age_sec=float(ros2_cfg.get("max_odom_age_sec", 0.05)),
            odom_pose_frame=str(ros2_cfg.get("odom_pose_frame", "lidar")).strip().lower(),
            pointcloud_qos=str(ros2_cfg.get("pointcloud_qos", "sensor_data")).strip().lower(),
            odom_qos=str(ros2_cfg.get("odom_qos", "default")).strip().lower(),
            use_tf=bool(ros2_cfg.get("use_tf", True)),
            tf_lookup_timeout_sec=float(ros2_cfg.get("tf_lookup_timeout_sec", 0.2)),
            min_range=float(data_loader_cfg["min_range"]),
            max_range=float(data_loader_cfg["max_range"]),
            use_filter=bool(data_loader_cfg["use_filter"]),
            sor_nn=int(data_loader_cfg["sor_nn"]),
            sor_std=float(data_loader_cfg["sor_std"]),
            calib_file=None if calib_file is None else str(calib_file),
        )

        source_info.update(
            {
                "pointcloud_source": capture_info["pointcloud_topic"],
                "pose_source": capture_info["odom_topic"],
                "calib_file": capture_info["calib_file"],
                "details": capture_info,
            }
        )
        return scene_dataset, out_dir, source_info

    raise ValueError(f"Unsupported Dataset.input_mode: {input_mode}")


def build_frame_indices(dataset_size: int, skip_load: int, max_frames: Optional[int]) -> np.ndarray:
    # Reproduce SLNR's frame subsampling policy: keep one frame every
    # `skip_load` frames, then optionally truncate for faster inspection.
    skip_load = max(1, int(skip_load))
    n_load_total = max(1, dataset_size // skip_load)
    indices = np.linspace(0, dataset_size, n_load_total, dtype=int, endpoint=False)
    if max_frames is not None:
        indices = indices[: max(1, max_frames)]
    return indices


def save_anchor_point_cloud(ply_path: Path, positions: np.ndarray, normals: np.ndarray) -> None:
    # Save anchor centers as a colored point cloud so they can be inspected in
    # Open3D or other point-cloud tools without loading the full SLNR model.
    colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(ply_path), point_cloud)


def main() -> None:
    args = parse_args()

    # LocalSDF is hard-wired to allocate tensors on CUDA in this repo, so this
    # inspection helper has the same GPU requirement as the original code path.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This harness requires CUDA because LocalSDF in slnr/local_sdf.py is hardcoded to use cuda."
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = resolve_config_path(args.conf)
    _require_file(config_path, "Config file")

    with open(config_path, encoding="utf-8") as config_file:
        cfg = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    # Build either the file-backed dataset or the ROS2-captured dataset while
    # keeping the downstream initialization path unchanged.
    scene_dataset, out_dir, source_info = build_scene_dataset(cfg, args.out_dir, args.max_frames)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.classes.load_library(str(SPARSE_HASH_LIB))

    hash_cfg = cfg["HashTable"]
    frame_indices = build_frame_indices(
        len(scene_dataset),
        hash_cfg["skip_load"] if args.skip_load is None else args.skip_load,
        args.max_frames,
    )

    print(f"Config        : {config_path}")
    print(f"Input mode    : {source_info['input_mode']}")
    print(f"Dataset root  : {source_info['dataset_root']}")
    print(f"Point clouds  : {source_info['pointcloud_source']}")
    print(f"Poses         : {source_info['pose_source']}")
    print(f"Calibration   : {source_info['calib_file']}")
    print(f"Output dir    : {out_dir}")
    print(f"Frames used   : {len(frame_indices)} / {len(scene_dataset)}")
    if source_info["input_mode"] == "ros2":
        print(f"ROS2 capture  : {json.dumps(source_info['details'], indent=2)}")

    # Construct the same sparse hash table and LocalSDF container used by the
    # main training pipeline, but stop after the initialization stage.
    svh = torch.classes.svh.HashTable(hash_cfg["voxel_size"], hash_cfg["ht_size"])
    local_sdfs = LocalSDF(
        resolution=cfg["LocalSDF"]["resolution"],
        query_nn_k=cfg["LocalSDF"]["query_nn_k"],
        gss_vox_size=cfg["LocalSDF"]["gss_vox_size"],
    )

    # This is the core stage-1 initialization step:
    # 1. accumulate/subsample input point clouds
    # 2. estimate or reuse normals
    # 3. insert occupied voxels into the sparse hash grid
    # 4. initialize local SDF anchor positions / rotations / scales
    main_util.allocate_localsdfs_in_svh(
        svh,
        scene_dataset,
        frame_indices,
        local_sdfs,
        res_scale=hash_cfg["res_scale"],
        down_voxel_size=hash_cfg["down_vox_size"],
    )

    ht_info, lcp_index, lcp_array = svh.get_ht_info()
    inval_val = hash_cfg["inval_val"]
    valid_voxel_mask = ht_info[:, 0] != inval_val

    # After `prepare_for_optimization`, the initialized anchor parameters live
    # in `local_sdfs` as torch Parameters. We detach them here for export only.
    positions = local_sdfs.positions.detach()
    rotations = local_sdfs.rotations.detach()
    scalings = local_sdfs.scalings.detach()

    # In SLNR, each anchor rotation aligns the local +Z axis with the estimated
    # surface normal. Recover that world-space normal for inspection/plotting.
    normals = pp.so3(rotations).Exp().matrix()[..., 2]

    positions_np = positions.cpu().numpy()
    rotations_np = rotations.cpu().numpy()
    scalings_np = scalings.cpu().numpy()
    normals_np = normals.cpu().numpy()
    linear_scales_np = np.exp(scalings_np)

    npz_path = out_dir / "local_sdf_init.npz"
    ply_path = out_dir / "local_sdf_init_points.ply"
    summary_path = out_dir / "local_sdf_init_summary.json"

    # Save the raw initialized anchor state in a compact form that is easy to
    # reload from analysis scripts or visualization tools.
    np.savez(
        npz_path,
        positions=positions_np,
        rotations=rotations_np,
        scalings=scalings_np,
        normals=normals_np,
        frame_indices=frame_indices,
    )
    save_anchor_point_cloud(ply_path, positions_np, normals_np)

    smoke_summary = None
    if not args.no_query_smoke and positions.shape[0] > 0:
        # Minimal sanity check: query some anchor centers back through
        # LocalSDF.query_feature and confirm neighbor lookup behaves sensibly.
        query_count = min(int(args.query_count), int(positions.shape[0]))
        sample_idx = torch.randperm(positions.shape[0], device=positions.device)[:query_count]
        query_points = positions[sample_idx]
        _, _, nn_counts = local_sdfs.query_feature(query_points)
        smoke_summary = {
            "query_count": int(query_count),
            "nn_count_min": int(nn_counts.min().item()),
            "nn_count_mean": float(nn_counts.float().mean().item()),
            "nn_count_max": int(nn_counts.max().item()),
        }

    # Summarize the exported initialization so you can inspect frame coverage,
    # anchor counts, scale statistics, and output file locations at a glance.
    summary = {
        "config_path": str(config_path),
        "input_mode": source_info["input_mode"],
        "dataset_root": source_info["dataset_root"],
        "pc_dir": source_info["pointcloud_source"],
        "pose_file": source_info["pose_source"],
        "calib_file": source_info["calib_file"],
        "source_details": source_info["details"],
        "output_dir": str(out_dir),
        "dataset_frame_count": int(len(scene_dataset)),
        "used_frame_count": int(len(frame_indices)),
        "hash_voxel_size": float(hash_cfg["voxel_size"]),
        "valid_hash_voxel_count": int(valid_voxel_mask.sum().item()),
        "lcp_index_shape": list(lcp_index.shape),
        "lcp_array_shape": list(lcp_array.shape),
        "local_sdf_count": int(positions.shape[0]),
        "position_min": positions_np.min(axis=0).tolist() if positions_np.size else [],
        "position_max": positions_np.max(axis=0).tolist() if positions_np.size else [],
        "scale_min": linear_scales_np.min(axis=0).tolist() if linear_scales_np.size else [],
        "scale_mean": linear_scales_np.mean(axis=0).tolist() if linear_scales_np.size else [],
        "scale_max": linear_scales_np.max(axis=0).tolist() if linear_scales_np.size else [],
        "artifacts": {
            "npz": str(npz_path),
            "ply": str(ply_path),
        },
        "query_smoke": smoke_summary,
    }

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary : {summary_path}")


if __name__ == "__main__":
    main()
