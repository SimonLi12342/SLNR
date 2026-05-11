#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import yaml


REPO_ROOT = Path(__file__).resolve().parent


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classic Open3D visualization for SLNR local-SDF initialization artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--conf",
        type=str,
        default="configs/example.yaml",
        help="Config path used only to infer the default artifact directory.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Directory containing local_sdf_init.npz and optional summary/json outputs.",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Explicit path to local_sdf_init.npz. Overrides --artifact-dir.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Explicit path to local_sdf_init_summary.json.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Maximum number of anchor points to render.",
    )
    parser.add_argument(
        "--max-normals",
        type=int,
        default=1500,
        help="Maximum number of normal vectors to render as lines.",
    )
    parser.add_argument(
        "--normal-scale",
        type=float,
        default=0.3,
        help="Length of rendered normal vectors in meters.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.5,
        help="Requested point size for the Open3D viewer.",
    )
    parser.add_argument(
        "--hide-normals",
        action="store_true",
        help="Render anchor points only.",
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
        help="Do not render the coordinate frame.",
    )
    return parser.parse_args()


def resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_default_artifact_dir(config_path: Path) -> Path:
    _require_file(config_path, "Config file")
    with open(config_path, encoding="utf-8") as config_file:
        cfg = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    out_dir = Path(cfg["Dataset"]["out_dir"])
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    return out_dir.resolve()


def resolve_artifact_paths(
    config_path: Path,
    artifact_dir_arg: Optional[str],
    npz_arg: Optional[str],
    summary_arg: Optional[str],
) -> Tuple[Path, Optional[Path], Path]:
    if npz_arg is not None:
        npz_path = Path(npz_arg)
        if not npz_path.is_absolute():
            npz_path = REPO_ROOT / npz_path
        npz_path = npz_path.resolve()
        artifact_dir = npz_path.parent
    else:
        if artifact_dir_arg is None:
            artifact_dir = resolve_default_artifact_dir(config_path)
        else:
            artifact_dir = Path(artifact_dir_arg)
            if not artifact_dir.is_absolute():
                artifact_dir = REPO_ROOT / artifact_dir
            artifact_dir = artifact_dir.resolve()
        npz_path = artifact_dir / "local_sdf_init.npz"

    if summary_arg is not None:
        summary_path = Path(summary_arg)
        if not summary_path.is_absolute():
            summary_path = REPO_ROOT / summary_path
        summary_path = summary_path.resolve()
    else:
        candidate = artifact_dir / "local_sdf_init_summary.json"
        summary_path = candidate if candidate.exists() else None

    return npz_path, summary_path, artifact_dir


def sample_indices(count: int, max_count: int) -> np.ndarray:
    if max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_count, dtype=np.int64)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


def compute_extent(positions: np.ndarray) -> np.ndarray:
    if positions.size == 0:
        return np.ones(3, dtype=np.float64)
    return positions.max(axis=0) - positions.min(axis=0)


def build_anchor_point_cloud(positions: np.ndarray, normals: np.ndarray) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return point_cloud


def build_normal_lines(
    positions: np.ndarray,
    normals: np.ndarray,
    normal_scale: float,
) -> o3d.geometry.LineSet:
    end_points = positions + normals * normal_scale
    line_points = np.concatenate((positions, end_points), axis=0)
    point_count = positions.shape[0]
    lines = np.stack(
        (
            np.arange(point_count, dtype=np.int32),
            np.arange(point_count, dtype=np.int32) + point_count,
        ),
        axis=1,
    )
    colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return line_set


def main() -> None:
    args = parse_args()

    config_path = resolve_config_path(args.conf)
    npz_path, summary_path, artifact_dir = resolve_artifact_paths(
        config_path,
        args.artifact_dir,
        args.npz,
        args.summary,
    )

    _require_file(npz_path, "Local SDF init npz")

    data = np.load(npz_path)
    positions = np.asarray(data["positions"], dtype=np.float32)
    normals = normalize_vectors(np.asarray(data["normals"], dtype=np.float32))
    scalings = np.asarray(data["scalings"], dtype=np.float32) if "scalings" in data else None

    point_indices = sample_indices(positions.shape[0], args.max_points)
    positions_view = positions[point_indices]
    normals_view = normals[point_indices]

    normal_indices = sample_indices(positions_view.shape[0], args.max_normals)
    positions_normals = positions_view[normal_indices]
    normals_normals = normals_view[normal_indices]

    print(f"Artifact dir : {artifact_dir}")
    print(f"NPZ path     : {npz_path}")
    print(f"Anchor count : {positions.shape[0]}")
    print(f"Rendered pts : {positions_view.shape[0]}")
    print(f"Rendered nrm : {0 if args.hide_normals else positions_normals.shape[0]}")

    if scalings is not None and scalings.size > 0:
        linear_scales = np.exp(scalings)
        print(f"Scale mean   : {linear_scales.mean(axis=0).tolist()}")
        print(f"Scale min    : {linear_scales.min(axis=0).tolist()}")
        print(f"Scale max    : {linear_scales.max(axis=0).tolist()}")

    if summary_path is not None:
        with open(summary_path, encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        print("Summary json : {}".format(summary_path))
        print(json.dumps(summary, indent=2))

    geometries = [build_anchor_point_cloud(positions_view, normals_view)]
    if not args.hide_normals and positions_normals.shape[0] > 0:
        geometries.append(build_normal_lines(positions_normals, normals_normals, args.normal_scale))

    if not args.no_frame:
        extent = compute_extent(positions)
        frame_size = max(float(np.max(extent)) * 0.1, args.normal_scale * 2.0, 0.2)
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SLNR Local SDF Init (Classic)", width=1600, height=900)
    render_option = vis.get_render_option()
    render_option.point_size = float(args.point_size)
    render_option.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    if hasattr(render_option, "line_width"):
        render_option.line_width = 1.5

    for geometry in geometries:
        vis.add_geometry(geometry)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
