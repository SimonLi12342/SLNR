#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import pypose as pp
import torch
import torch.optim as optim
import yaml

import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener


REPO_ROOT = Path(__file__).resolve().parent
SPARSE_HASH_LIB = REPO_ROOT / "modules" / "sparse_hash" / "build" / "libsvh.so"
GS_SEARCH_LIB = REPO_ROOT / "modules" / "gaussian_search" / "build" / "libgs_search.so"


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


_require_file(SPARSE_HASH_LIB, "Sparse hash extension")
_require_file(GS_SEARCH_LIB, "Gaussian search extension")

# Match the original repo behavior: relative shared-library loads assume the repo
# root is the current working directory.
os.chdir(REPO_ROOT)

torch.classes.load_library(str(SPARSE_HASH_LIB))

import main_util  # noqa: E402
import network  # noqa: E402
from local_sdf import LocalSDF  # noqa: E402
from ros2_capture_dataset import (  # noqa: E402
    _extract_points_and_normals,
    _filter_points,
    _load_calibration,
    _normalize_frame_id,
    _pose_from_position_quaternion,
    _pose_from_transform,
    _resolve_pose_frame,
    _stamp_to_sec,
)


@dataclass
class PendingFrame:
    points_local: np.ndarray
    normals_local: Optional[np.ndarray]
    pose_wc: np.ndarray
    cloud_stamp_sec: Optional[float]
    odom_stamp_sec: Optional[float]


@dataclass
class ProcessedFrame:
    pose_wc: np.ndarray
    train_points_world: np.ndarray
    insert_points_world: np.ndarray
    insert_normals_world: np.ndarray
    cloud_stamp_sec: Optional[float]
    odom_stamp_sec: Optional[float]


@dataclass
class TrainingFrame:
    points_world: np.ndarray
    pose_wc: np.ndarray
    cloud_stamp_sec: Optional[float]


class InMemoryFrameDataset:
    def __init__(self, frames: Sequence[PendingFrame]):
        self.frames = list(frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        frame = self.frames[idx]
        normals = None if frame.normals_local is None else frame.normals_local.copy()
        return frame.points_local.copy(), normals, frame.pose_wc.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online ROS2 runner for SLNR using externally provided odometry.",
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
        "--seed",
        type=int,
        default=0,
        help="Random seed for frame/point sampling.",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable the Open3D live viewer even if enabled in the config.",
    )
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=None,
        help="Optional wall-clock limit for the online loop.",
    )
    return parser.parse_args()


def resolve_config_path(config_path: str) -> Path:
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


def resolve_file_data_paths(cfg: dict) -> Tuple[Path, Optional[Path]]:
    dataset_cfg = cfg["Dataset"]
    workspace = Path(dataset_cfg["workspace"])
    data_dir = Path(dataset_cfg["data_dir"])
    root_dir = (REPO_ROOT / workspace / data_dir).resolve()

    calib_file = None
    if "calib_file" in dataset_cfg:
        calib_file = (root_dir / dataset_cfg["calib_file"]).resolve()
    return root_dir, calib_file


def transform_points(points_local: np.ndarray, pose_wc: np.ndarray) -> np.ndarray:
    rotation = pose_wc[:3, :3]
    translation = pose_wc[:3, 3]
    return (points_local @ rotation.T + translation[None, :]).astype(np.float32)


def transform_normals(normals_local: np.ndarray, pose_wc: np.ndarray) -> np.ndarray:
    rotation = pose_wc[:3, :3]
    normals_world = normals_local @ rotation.T
    norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
    return (normals_world / (norms + 1e-5)).astype(np.float32)


def estimate_world_normals(points_world: np.ndarray, pose_wc: np.ndarray) -> np.ndarray:
    if points_world.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
    point_cloud.estimate_normals()
    normals_world = np.asarray(point_cloud.normals, dtype=np.float32)

    rays_o = pose_wc[:3, 3]
    rays_d = rays_o[None, :] - points_world
    rays_d[:, 2] += 3.0
    ranges = np.linalg.norm(rays_d, axis=1, keepdims=True)
    rays_d = rays_d / (ranges + 1e-5)
    dd = (normals_world * rays_d).sum(axis=-1)
    normals_world[dd < 0.0] *= -1.0

    norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
    normals_world = normals_world / (norms + 1e-5)
    return normals_world.astype(np.float32)


def voxel_select_first(
    points: np.ndarray,
    normals: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or points.shape[0] <= 1:
        return points.astype(np.float32), normals.astype(np.float32)

    grid = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(grid, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points[unique_indices].astype(np.float32), normals[unique_indices].astype(np.float32)


def sample_fixed_points(points: np.ndarray, target_count: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    if target_count <= 0:
        return points.astype(np.float32)
    if points.shape[0] >= target_count:
        indices = rng.choice(points.shape[0], size=target_count, replace=False)
    else:
        indices = rng.choice(points.shape[0], size=target_count, replace=True)
    return points[indices].astype(np.float32)


def normals_to_rotvec(normals: torch.Tensor) -> torch.Tensor:
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-5)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype).view(1, 3)
    z_axis = z_axis.expand_as(normals)

    dots = torch.clamp((z_axis * normals).sum(dim=-1), -1.0, 1.0)
    angles = torch.arccos(dots)
    axis = torch.cross(z_axis, normals, dim=-1)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)

    rotvec = torch.zeros_like(normals)
    regular_mask = axis_norm.squeeze(-1) > 1e-6
    if regular_mask.any():
        axis_regular = axis[regular_mask] / axis_norm[regular_mask]
        rotvec[regular_mask] = axis_regular * angles[regular_mask].unsqueeze(-1)

    opposite_mask = (~regular_mask) & (dots < 0.0)
    if opposite_mask.any():
        rotvec[opposite_mask, 0] = math.pi

    return rotvec


def compute_bounds_from_voxels(
    vox_coords_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if vox_coords_world.shape[0] == 0:
        return np.ones(3, dtype=np.float32), np.eye(4, dtype=np.float32)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vox_coords_world.astype(np.float64))

    if vox_coords_world.shape[0] >= 4:
        obb = point_cloud.get_oriented_bounding_box()
        bounds_extents = np.asarray(obb.extent, dtype=np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = np.asarray(obb.R, dtype=np.float32)
        transform[:3, 3] = np.asarray(obb.center, dtype=np.float32)
        inv_transform = np.linalg.inv(transform).astype(np.float32)
        return bounds_extents, inv_transform

    coords_min = vox_coords_world.min(axis=0)
    coords_max = vox_coords_world.max(axis=0)
    bounds_extents = np.maximum(coords_max - coords_min, 1e-3).astype(np.float32)
    center = ((coords_min + coords_max) * 0.5).astype(np.float32)
    inv_transform = np.eye(4, dtype=np.float32)
    inv_transform[:3, 3] = -center
    return bounds_extents, inv_transform


def sample_items(items: Sequence[TrainingFrame], count: int, rng: np.random.Generator) -> List[TrainingFrame]:
    if count <= 0 or len(items) == 0:
        return []
    if len(items) >= count:
        indices = rng.choice(len(items), size=count, replace=False)
    else:
        indices = rng.choice(len(items), size=count, replace=True)
    return [items[int(idx)] for idx in indices]


def save_anchor_point_cloud(ply_path: Path, positions: np.ndarray, normals: np.ndarray) -> None:
    colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(ply_path), point_cloud)


class OnlineSLNRNode(Node):
    def __init__(self, cfg: dict, out_dir: Path, seed: int, enable_visualization: bool) -> None:
        super().__init__("slnr_online_ros2")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "run_online_ros2.py requires CUDA because LocalSDF in this repo is hardcoded to use cuda."
            )

        self.cfg = cfg
        self.dataset_cfg = cfg["Dataset"]
        self.ros2_cfg = cfg.get("ROS2", {})
        self.online_cfg = cfg.get("Online", {})
        self.hash_cfg = cfg["HashTable"]
        self.train_cfg = cfg["Train"]
        self.local_sdf_cfg = cfg["LocalSDF"]
        self.ray_cfg = cfg["RaySample"]
        self.loss_cfg = cfg["Loss"]
        self.save_cfg = cfg["Save"]
        self.data_loader_cfg = cfg["DataLoader"]

        self.device = "cuda"
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.out_dir / "online_status.json"

        _, calib_file = resolve_file_data_paths(cfg)
        if calib_file is not None:
            _require_file(calib_file, "Calibration file")
        self.calib_file = None if calib_file is None else str(calib_file)
        self.calib_tr = _load_calibration(self.calib_file)

        self.pointcloud_topic = str(self.ros2_cfg.get("pointcloud_topic", "/velodyne_points"))
        self.odom_topic = str(self.ros2_cfg.get("odom_topic", "/integrated_to_init"))
        self.pointcloud_qos = str(self.ros2_cfg.get("pointcloud_qos", "sensor_data")).strip().lower()
        self.odom_qos = str(self.ros2_cfg.get("odom_qos", "default")).strip().lower()
        self.max_odom_age_sec = float(self.ros2_cfg.get("max_odom_age_sec", 0.05))
        self.odom_pose_frame = str(self.ros2_cfg.get("odom_pose_frame", "lidar")).strip().lower()
        self.use_tf = bool(self.ros2_cfg.get("use_tf", True))
        self.tf_lookup_timeout_sec = float(self.ros2_cfg.get("tf_lookup_timeout_sec", 0.2))

        self.warmup_frames = max(1, int(self.online_cfg.get("warmup_frames", 20)))
        self.max_pending_frames = max(1, int(self.online_cfg.get("max_pending_frames", 20)))
        self.max_processed_frames_per_spin = max(1, int(self.online_cfg.get("max_processed_frames_per_spin", 1)))
        self.recent_buffer_size = max(1, int(self.online_cfg.get("recent_buffer_size", 30)))
        self.replay_buffer_size = max(1, int(self.online_cfg.get("replay_buffer_size", 120)))
        self.replay_insert_interval = max(1, int(self.online_cfg.get("replay_insert_interval", 5)))
        self.train_recent_ratio = float(self.online_cfg.get("train_recent_ratio", 0.7))
        self.train_frames_per_step = max(
            1,
            int(self.online_cfg.get("train_frames_per_step", self.ray_cfg.get("n_views_select", 1))),
        )
        self.train_steps_per_spin = max(1, int(self.online_cfg.get("train_steps_per_spin", 1)))
        self.train_points_per_frame = max(1, int(self.online_cfg.get("train_points_per_frame", 4096)))
        self.min_points_per_frame = max(1, int(self.online_cfg.get("min_points_per_frame", 256)))
        self.hash_insert_down_voxel_size = float(
            self.online_cfg.get("hash_insert_down_voxel_size", self.hash_cfg["down_vox_size"])
        )
        self.max_new_anchors_per_frame = max(0, int(self.online_cfg.get("max_new_anchors_per_frame", 2000)))
        self.refresh_map_every_n_frames = max(1, int(self.online_cfg.get("refresh_map_every_n_frames", 1)))
        self.log_every_n_iters = max(1, int(self.online_cfg.get("log_every_n_iters", 20)))
        self.vis_every_n_iters = max(1, int(self.online_cfg.get("vis_every_n_iters", 20)))
        self.save_every_n_iters = max(0, int(self.online_cfg.get("save_every_n_iters", 1000)))
        self.mesh_every_n_iters = max(0, int(self.online_cfg.get("mesh_every_n_iters", 0)))
        self.show_basis_sdf = bool(self.online_cfg.get("show_basis_sdf", False))
        self.enable_visualization = enable_visualization and bool(
            self.online_cfg.get("enable_visualization", True)
        )
        self.lr_decay_iters = max(
            1,
            int(
                self.online_cfg.get(
                    "lr_decay_iters",
                    int(self.train_cfg["n_epoch"]) * int(self.train_cfg["n_step"]),
                )
            ),
        )

        self.pending_frames: Deque[PendingFrame] = deque()
        self.warmup_raw_frames: List[PendingFrame] = []
        self.warmup_processed_frames: List[ProcessedFrame] = []
        self.recent_frames: Deque[TrainingFrame] = deque(maxlen=self.recent_buffer_size)
        self.replay_frames: Deque[TrainingFrame] = deque(maxlen=self.replay_buffer_size)

        self.received_odom_msgs = 0
        self.received_cloud_msgs = 0
        self.accepted_frames = 0
        self.accepted_direct_pose = 0
        self.accepted_tf_pose = 0
        self.accepted_legacy_pose = 0
        self.skipped_no_odom = 0
        self.skipped_stale_odom = 0
        self.skipped_empty_cloud = 0
        self.skipped_filtered_empty = 0
        self.skipped_tf_unavailable = 0
        self.dropped_pending_frames = 0
        self.last_tf_error: Optional[str] = None
        self.reported_pose_mappings = set()
        self.observed_cloud_frames = set()
        self.observed_odom_frames = set()
        self.observed_odom_child_frames = set()

        self.latest_odom_pose: Optional[np.ndarray] = None
        self.latest_odom_stamp_sec: Optional[float] = None
        self.latest_odom_frame_id = ""
        self.latest_odom_child_frame_id = ""

        self.initialized = False
        self.integrated_frame_count = 0
        self.processed_frame_count = 0
        self.train_iter = 0
        self.last_save_iter = -1

        self.svh = None
        self.local_sdfs: Optional[LocalSDF] = None
        self.neural_map = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ht_info_device = None
        self.ht_info_cpu = None
        self.bounds_extents = None
        self.inv_bounds_transform = None
        self.anchor_voxel_keys = set()

        self.visualizer = None
        self.vis_neural_map = None
        self.vis_geometry_added = False
        if self.enable_visualization:
            self.visualizer = o3d.visualization.Visualizer()
            self.visualizer.create_window(window_name="SLNR Online ROS2", width=1600, height=900)
            render_option = self.visualizer.get_render_option()
            render_option.point_size = 2.0
            render_option.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            self.vis_neural_map = o3d.geometry.PointCloud()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.create_subscription(PointCloud2, self.pointcloud_topic, self.on_point_cloud, self.make_qos(self.pointcloud_qos))
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, self.make_qos(self.odom_qos))

        self.get_logger().info(
            "SLNR online ROS2 runner started "
            f"(cloud={self.pointcloud_topic}, odom={self.odom_topic}, warmup_frames={self.warmup_frames})"
        )

    def make_qos(self, name: str):
        if name == "sensor_data":
            return qos_profile_sensor_data
        if name == "default":
            return QoSProfile(depth=10)
        if name == "best_effort":
            return QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
        if name == "reliable":
            return QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
            )
        raise ValueError(f"Unsupported ROS2 QoS profile: {name}")

    def on_odom(self, msg: Odometry) -> None:
        self.received_odom_msgs += 1
        self.latest_odom_pose = _pose_from_position_quaternion(msg.pose.pose.position, msg.pose.pose.orientation).astype(
            np.float32
        )
        self.latest_odom_stamp_sec = _stamp_to_sec(msg.header.stamp)
        self.latest_odom_frame_id = _normalize_frame_id(msg.header.frame_id)
        self.latest_odom_child_frame_id = _normalize_frame_id(msg.child_frame_id)
        if self.latest_odom_frame_id:
            self.observed_odom_frames.add(self.latest_odom_frame_id)
        if self.latest_odom_child_frame_id:
            self.observed_odom_child_frames.add(self.latest_odom_child_frame_id)

    def resolve_pose_for_cloud(self, cloud_frame_id: str) -> Tuple[Optional[np.ndarray], str]:
        if self.latest_odom_pose is None:
            return None, "missing_odom"

        cloud_frame = _normalize_frame_id(cloud_frame_id)
        odom_child_frame = self.latest_odom_child_frame_id
        if cloud_frame:
            self.observed_cloud_frames.add(cloud_frame)

        if odom_child_frame and cloud_frame:
            if odom_child_frame == cloud_frame:
                return self.latest_odom_pose.copy(), "direct"

            if self.use_tf:
                try:
                    child_from_cloud = self.tf_buffer.lookup_transform(
                        odom_child_frame,
                        cloud_frame,
                        Time(),
                        timeout=Duration(seconds=max(0.0, float(self.tf_lookup_timeout_sec))),
                    )
                except Exception as exc:
                    self.last_tf_error = str(exc)
                    return None, "tf_unavailable"

                pose_wc = self.latest_odom_pose.astype(np.float64) @ _pose_from_transform(child_from_cloud.transform)
                return pose_wc.astype(np.float32), "tf"

        pose_wc = _resolve_pose_frame(
            self.latest_odom_pose.astype(np.float64),
            self.odom_pose_frame,
            self.calib_tr,
        )
        return pose_wc.astype(np.float32), "legacy"

    def report_pose_mapping(self, cloud_frame_id: str, pose_mode: str) -> None:
        mapping = (
            self.latest_odom_frame_id,
            self.latest_odom_child_frame_id,
            _normalize_frame_id(cloud_frame_id),
            pose_mode,
        )
        if mapping in self.reported_pose_mappings:
            return
        self.reported_pose_mappings.add(mapping)
        self.get_logger().info(
            "Pose mapping: "
            f"world='{mapping[0] or '<unknown>'}', "
            f"odom_child='{mapping[1] or '<unknown>'}', "
            f"cloud='{mapping[2] or '<unknown>'}', "
            f"mode='{mapping[3]}'"
        )

    def on_point_cloud(self, msg: PointCloud2) -> None:
        self.received_cloud_msgs += 1

        if self.latest_odom_pose is None:
            self.skipped_no_odom += 1
            return

        cloud_stamp_sec = _stamp_to_sec(msg.header.stamp)
        if (
            cloud_stamp_sec is not None
            and self.latest_odom_stamp_sec is not None
            and abs(cloud_stamp_sec - self.latest_odom_stamp_sec) > self.max_odom_age_sec
        ):
            self.skipped_stale_odom += 1
            return

        pose_wc, pose_mode = self.resolve_pose_for_cloud(msg.header.frame_id)
        if pose_wc is None:
            if pose_mode == "tf_unavailable":
                self.skipped_tf_unavailable += 1
            else:
                self.skipped_no_odom += 1
            return

        points_local, normals_local = _extract_points_and_normals(point_cloud2, msg)
        if points_local.shape[0] == 0:
            self.skipped_empty_cloud += 1
            return

        points_local, normals_local = _filter_points(
            points_local,
            normals_local,
            min_range=float(self.data_loader_cfg["min_range"]),
            max_range=float(self.data_loader_cfg["max_range"]),
            use_filter=bool(self.data_loader_cfg["use_filter"]),
            sor_nn=int(self.data_loader_cfg["sor_nn"]),
            sor_std=float(self.data_loader_cfg["sor_std"]),
        )
        if points_local.shape[0] == 0:
            self.skipped_filtered_empty += 1
            return

        if len(self.pending_frames) >= self.max_pending_frames:
            self.pending_frames.popleft()
            self.dropped_pending_frames += 1

        self.report_pose_mapping(msg.header.frame_id, pose_mode)
        if pose_mode == "direct":
            self.accepted_direct_pose += 1
        elif pose_mode == "tf":
            self.accepted_tf_pose += 1
        else:
            self.accepted_legacy_pose += 1

        self.pending_frames.append(
            PendingFrame(
                points_local=points_local,
                normals_local=normals_local,
                pose_wc=pose_wc,
                cloud_stamp_sec=cloud_stamp_sec,
                odom_stamp_sec=self.latest_odom_stamp_sec,
            )
        )
        self.accepted_frames += 1

    def process_pending_frames(self) -> None:
        processed_this_spin = 0
        while self.pending_frames and processed_this_spin < self.max_processed_frames_per_spin:
            pending = self.pending_frames.popleft()
            processed = self.prepare_processed_frame(pending)
            if processed is None:
                processed_this_spin += 1
                continue

            if not self.initialized:
                self.warmup_raw_frames.append(pending)
                self.warmup_processed_frames.append(processed)
                if len(self.warmup_raw_frames) >= self.warmup_frames:
                    self.initialize_from_warmup()
            else:
                self.integrate_processed_frame(processed)

            processed_this_spin += 1

    def prepare_processed_frame(self, frame: PendingFrame) -> Optional[ProcessedFrame]:
        if frame.points_local.shape[0] < self.min_points_per_frame:
            return None

        points_world = transform_points(frame.points_local, frame.pose_wc)
        if frame.normals_local is not None:
            normals_world = transform_normals(frame.normals_local, frame.pose_wc)
        else:
            normals_world = estimate_world_normals(points_world, frame.pose_wc)

        insert_points_world, insert_normals_world = voxel_select_first(
            points_world,
            normals_world,
            self.hash_insert_down_voxel_size,
        )
        train_points_world = sample_fixed_points(points_world, self.train_points_per_frame, self.rng)

        if train_points_world.shape[0] < self.min_points_per_frame:
            return None

        return ProcessedFrame(
            pose_wc=frame.pose_wc.astype(np.float32),
            train_points_world=train_points_world,
            insert_points_world=insert_points_world,
            insert_normals_world=insert_normals_world,
            cloud_stamp_sec=frame.cloud_stamp_sec,
            odom_stamp_sec=frame.odom_stamp_sec,
        )

    def initialize_from_warmup(self) -> None:
        self.get_logger().info(f"Initializing SLNR from {len(self.warmup_raw_frames)} live frames")

        frame_dataset = InMemoryFrameDataset(self.warmup_raw_frames)
        frame_indices = np.arange(len(frame_dataset), dtype=np.int64)

        self.svh = torch.classes.svh.HashTable(self.hash_cfg["voxel_size"], self.hash_cfg["ht_size"])
        self.local_sdfs = LocalSDF(
            resolution=self.local_sdf_cfg["resolution"],
            query_nn_k=self.local_sdf_cfg["query_nn_k"],
            gss_vox_size=self.local_sdf_cfg["gss_vox_size"],
        )

        main_util.allocate_localsdfs_in_svh(
            self.svh,
            frame_dataset,
            frame_indices,
            self.local_sdfs,
            res_scale=self.hash_cfg["res_scale"],
            down_voxel_size=self.hash_cfg["down_vox_size"],
        )
        self.refresh_hash_state()

        self.neural_map = network.NeuralMap(
            self.local_sdfs,
            num_layers=self.cfg["Network"]["num_layers"],
            hidden_dim=self.cfg["Network"]["hidden_dim"],
        ).to(self.device)

        params = [
            {"params": self.neural_map.sdf_net.parameters(), "name": "sdf_net"},
            {"params": [self.local_sdfs.positions], "name": "positions"},
            {"params": [self.local_sdfs.rotations], "name": "rotations"},
            {"params": [self.local_sdfs.scalings], "name": "scalings"},
        ]
        self.optimizer = optim.Adam(params, lr=5e-3)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda iteration: self.train_cfg["lr_ratio"] ** min(iteration / self.lr_decay_iters, 1.0),
        )

        self.rebuild_anchor_voxel_keys()
        for processed in self.warmup_processed_frames:
            self.processed_frame_count += 1
            self.add_training_frame(processed)

        self.initialized = True
        self.write_status_json()
        self.get_logger().info(
            f"Warmup complete: anchors={int(self.local_sdfs.positions.shape[0])}, "
            f"recent_buffer={len(self.recent_frames)}, replay_buffer={len(self.replay_frames)}"
        )

    def integrate_processed_frame(self, processed: ProcessedFrame) -> None:
        self.processed_frame_count += 1
        self.integrated_frame_count += 1
        self.add_training_frame(processed)

        if processed.insert_points_world.shape[0] == 0:
            return

        insert_tensor = torch.from_numpy(processed.insert_points_world).float()
        self.svh.insert(insert_tensor)

        new_anchor_count = self.append_new_anchors(
            processed.insert_points_world,
            processed.insert_normals_world,
        )
        if (
            self.integrated_frame_count % self.refresh_map_every_n_frames == 0
            or new_anchor_count > 0
        ):
            self.refresh_hash_state()

    def add_training_frame(self, processed: ProcessedFrame) -> None:
        training_frame = TrainingFrame(
            points_world=processed.train_points_world.astype(np.float32),
            pose_wc=processed.pose_wc.astype(np.float32),
            cloud_stamp_sec=processed.cloud_stamp_sec,
        )
        self.recent_frames.append(training_frame)
        if self.processed_frame_count % self.replay_insert_interval == 0 or not self.replay_frames:
            self.replay_frames.append(training_frame)

    def append_new_anchors(self, points_world: np.ndarray, normals_world: np.ndarray) -> int:
        if self.local_sdfs is None or self.optimizer is None:
            return 0
        if points_world.shape[0] == 0:
            return 0

        candidate_points, candidate_normals = voxel_select_first(
            points_world,
            normals_world,
            float(self.local_sdfs.resolution),
        )
        if candidate_points.shape[0] == 0:
            return 0

        anchor_grid = np.floor(candidate_points / float(self.local_sdfs.resolution)).astype(np.int64)
        keep_indices = []
        keep_keys = []
        for idx, key in enumerate(anchor_grid):
            key_tuple = (int(key[0]), int(key[1]), int(key[2]))
            if key_tuple in self.anchor_voxel_keys:
                continue
            keep_indices.append(idx)
            keep_keys.append(key_tuple)

        if not keep_indices:
            return 0

        if 0 < self.max_new_anchors_per_frame < len(keep_indices):
            selected = self.rng.choice(len(keep_indices), size=self.max_new_anchors_per_frame, replace=False)
            keep_indices = [keep_indices[int(i)] for i in selected]
            keep_keys = [keep_keys[int(i)] for i in selected]

        new_points = torch.from_numpy(candidate_points[keep_indices]).to(self.device).float()
        new_normals = torch.from_numpy(candidate_normals[keep_indices]).to(self.device).float()
        new_rotations = normals_to_rotvec(new_normals)
        new_scalings = torch.ones((new_points.shape[0], 3), device=self.device) * math.log(self.local_sdfs.resolution)

        self.local_sdfs.densification_postfix(new_points, new_rotations, new_scalings, self.optimizer)
        self.anchor_voxel_keys.update(keep_keys)
        return int(new_points.shape[0])

    def rebuild_anchor_voxel_keys(self) -> None:
        if self.local_sdfs is None or self.local_sdfs.positions.shape[0] == 0:
            self.anchor_voxel_keys = set()
            return
        positions = self.local_sdfs.positions.detach().cpu().numpy()
        anchor_grid = np.floor(positions / float(self.local_sdfs.resolution)).astype(np.int64)
        self.anchor_voxel_keys = {
            (int(key[0]), int(key[1]), int(key[2]))
            for key in anchor_grid
        }

    def refresh_hash_state(self) -> None:
        ht_info, _, _ = self.svh.get_ht_info()
        self.ht_info_cpu = ht_info
        self.ht_info_device = ht_info.to(self.device)

        inval_val = self.hash_cfg["inval_val"]
        vox_coords = ht_info[:, :3]
        valid_mask = vox_coords[:, 0] != inval_val
        valid_voxels = vox_coords[valid_mask]
        vox_world = ((valid_voxels + 0.5) * self.hash_cfg["voxel_size"]).cpu().numpy().astype(np.float32)

        bounds_extents_np, inv_bounds_np = compute_bounds_from_voxels(vox_world)
        bounds_extents_np = bounds_extents_np * 1.1
        self.bounds_extents = torch.from_numpy(bounds_extents_np).float().to(self.device)
        self.inv_bounds_transform = torch.from_numpy(inv_bounds_np).float().to(self.device)

    def sample_training_frames(self) -> List[TrainingFrame]:
        if len(self.recent_frames) == 0 and len(self.replay_frames) == 0:
            return []

        n_total = self.train_frames_per_step
        n_recent = min(len(self.recent_frames), max(1, int(round(n_total * self.train_recent_ratio))))
        n_replay = max(0, n_total - n_recent)

        frames = sample_items(list(self.recent_frames), n_recent, self.rng)
        frames.extend(sample_items(list(self.replay_frames), n_replay, self.rng))

        if len(frames) < n_total:
            frames.extend(sample_items(list(self.recent_frames), n_total - len(frames), self.rng))
        return frames[:n_total]

    def optimize_step(self) -> None:
        if not self.initialized or self.neural_map is None or self.optimizer is None:
            return
        if self.ht_info_device is None or self.bounds_extents is None or self.inv_bounds_transform is None:
            return

        frames = self.sample_training_frames()
        if len(frames) == 0:
            return

        pc_batch = torch.from_numpy(np.stack([frame.points_world for frame in frames], axis=0)).float().to(self.device)
        T_batch = torch.from_numpy(np.stack([frame.pose_wc for frame in frames], axis=0)).float().to(self.device)

        sample_pts = main_util.sample_points_svh(
            self.svh,
            self.ht_info_device,
            pc_batch,
            T_batch,
            self.bounds_extents,
            self.inv_bounds_transform,
            n_rays=self.ray_cfg["n_rays"],
            n_max=self.ray_cfg["n_max_interset"],
            sur_behind_dis=self.ray_cfg["sur_behind_dis"],
            n_surf_samples=self.ray_cfg["n_surf_samples"],
            s_dev=self.ray_cfg["s_dev"],
            step_size_sdf=self.ray_cfg["step_size_sdf"],
            device=self.device,
        )
        if sample_pts["pc_sdf"].numel() == 0 or sample_pts["sample_mask_sdf"].sum().item() == 0:
            return

        self.optimizer.zero_grad(set_to_none=True)
        loss = main_util.compute_loss(
            self.neural_map,
            sample_pts,
            iter=self.train_iter,
            trunc_distance=self.loss_cfg["trunc_distance"],
            trunc_weight=self.loss_cfg["trunc_weight"],
            add_scale_loss=self.local_sdf_cfg["add_scale_loss"],
        )
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            return

        loss.backward()

        if self.train_iter > self.train_cfg["freeze_after_iters"]:
            main_util.freeze_model(self.neural_map.sdf_net)

        if self.show_basis_sdf:
            self.neural_map.vis_basis_sdf(resolution=256, vis="xz")

        densified = False
        if self.local_sdf_cfg["exe_densitify"]:
            self.local_sdfs.add_densification_stats()
            if (self.train_iter + 1) >= 1000 and (self.train_iter + 1) % self.local_sdf_cfg["n_iters_densitify"] == 0:
                self.local_sdfs.densify_and_prune(
                    self.cfg,
                    iter=self.train_iter,
                    optimizer=self.optimizer,
                    neural_map=self.neural_map,
                )
                densified = True

        self.optimizer.step()
        self.lr_scheduler.step()
        self.train_iter += 1

        if densified:
            self.rebuild_anchor_voxel_keys()

        if self.train_iter % self.log_every_n_iters == 0:
            self.get_logger().info(
                f"iter={self.train_iter} "
                f"loss={float(loss.item()):.6f} "
                f"anchors={int(self.local_sdfs.positions.shape[0])} "
                f"pending={len(self.pending_frames)} "
                f"recent={len(self.recent_frames)} "
                f"replay={len(self.replay_frames)} "
                f"integrated_frames={self.integrated_frame_count}"
            )

    def maybe_run_optimization(self) -> None:
        if not self.initialized:
            return
        for _ in range(self.train_steps_per_spin):
            self.optimize_step()
        self.maybe_update_visualization()
        self.maybe_save_artifacts()
        self.write_status_json()

    def maybe_update_visualization(self) -> None:
        if not self.enable_visualization or self.visualizer is None or self.vis_neural_map is None:
            return
        if self.local_sdfs is None or self.local_sdfs.positions.shape[0] == 0:
            self.visualizer.poll_events()
            self.visualizer.update_renderer()
            return

        if (self.train_iter % self.vis_every_n_iters) == 0 or not self.vis_geometry_added:
            positions_np = self.local_sdfs.positions.detach().cpu().numpy()
            normals_np = (
                pp.so3(self.local_sdfs.rotations.detach())
                .Exp()
                .matrix()[..., 2]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            colors = (normals_np + 1.0) * 0.5
            self.vis_neural_map.points = o3d.utility.Vector3dVector(positions_np)
            self.vis_neural_map.colors = o3d.utility.Vector3dVector(colors)
            if not self.vis_geometry_added:
                self.visualizer.add_geometry(self.vis_neural_map)
                self.vis_geometry_added = True
            else:
                self.visualizer.update_geometry(self.vis_neural_map)

        self.visualizer.poll_events()
        self.visualizer.update_renderer()

    def maybe_save_artifacts(self) -> None:
        if not self.initialized or self.neural_map is None:
            return
        if self.save_every_n_iters <= 0:
            return
        if self.train_iter <= 0 or self.train_iter == self.last_save_iter:
            return
        if self.train_iter % self.save_every_n_iters != 0:
            return

        model_path = self.out_dir / f"online_model_{self.train_iter}.pth"
        state = {
            "net_params": self.neural_map.state_dict(),
            "n_iter": self.train_iter,
            "integrated_frames": self.integrated_frame_count,
        }
        torch.save(state, model_path)
        self.last_save_iter = self.train_iter
        self.get_logger().info(f"Saved checkpoint: {model_path}")

        if self.mesh_every_n_iters > 0 and self.train_iter % self.mesh_every_n_iters == 0:
            mesh_path = self.out_dir / f"online_mesh_{self.train_iter}.ply"
            main_util.create_mesh_svh(
                self.ht_info_device,
                self.hash_cfg["voxel_size"],
                self.neural_map,
                grid_res=self.save_cfg["grid_res"],
                chunk_size=256,
                mesh_min_nn=self.save_cfg["mesh_min_nn"],
                save_path=str(mesh_path),
                device=self.device,
            )
            self.get_logger().info(f"Saved mesh: {mesh_path}")

    def export_anchor_artifacts(self) -> None:
        if self.local_sdfs is None or self.local_sdfs.positions.shape[0] == 0:
            return

        positions = self.local_sdfs.positions.detach()
        rotations = self.local_sdfs.rotations.detach()
        scalings = self.local_sdfs.scalings.detach()
        normals = pp.so3(rotations).Exp().matrix()[..., 2]

        positions_np = positions.cpu().numpy()
        rotations_np = rotations.cpu().numpy()
        scalings_np = scalings.cpu().numpy()
        normals_np = normals.cpu().numpy()
        linear_scales_np = np.exp(scalings_np)

        npz_path = self.out_dir / "local_sdf_init.npz"
        ply_path = self.out_dir / "local_sdf_init_points.ply"
        summary_path = self.out_dir / "local_sdf_init_summary.json"

        np.savez(
            npz_path,
            positions=positions_np,
            rotations=rotations_np,
            scalings=scalings_np,
            normals=normals_np,
            frame_indices=np.arange(self.processed_frame_count, dtype=np.int64),
        )
        save_anchor_point_cloud(ply_path, positions_np, normals_np)

        summary = {
            "config_path": None,
            "input_mode": "ros2_online",
            "dataset_root": None,
            "pc_dir": self.pointcloud_topic,
            "pose_file": self.odom_topic,
            "calib_file": self.calib_file,
            "source_details": {
                "pointcloud_topic": self.pointcloud_topic,
                "odom_topic": self.odom_topic,
                "pointcloud_qos": self.pointcloud_qos,
                "odom_qos": self.odom_qos,
                "odom_pose_frame": self.odom_pose_frame,
                "use_tf": self.use_tf,
                "tf_lookup_timeout_sec": self.tf_lookup_timeout_sec,
            },
            "output_dir": str(self.out_dir),
            "dataset_frame_count": int(self.accepted_frames),
            "used_frame_count": int(self.processed_frame_count),
            "hash_voxel_size": float(self.hash_cfg["voxel_size"]),
            "valid_hash_voxel_count": 0 if self.ht_info_cpu is None else int((self.ht_info_cpu[:, 0] != self.hash_cfg["inval_val"]).sum().item()),
            "lcp_index_shape": [],
            "lcp_array_shape": [],
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
            "query_smoke": None,
            "online_status": {
                "initialized": self.initialized,
                "integrated_frame_count": int(self.integrated_frame_count),
                "train_iter": int(self.train_iter),
                "received_cloud_msgs": int(self.received_cloud_msgs),
                "received_odom_msgs": int(self.received_odom_msgs),
            },
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.get_logger().info(f"Exported anchor artifacts: {npz_path}")

    def write_status_json(self) -> None:
        summary = {
            "pointcloud_topic": self.pointcloud_topic,
            "odom_topic": self.odom_topic,
            "pointcloud_qos": self.pointcloud_qos,
            "odom_qos": self.odom_qos,
            "odom_pose_frame": self.odom_pose_frame,
            "use_tf": self.use_tf,
            "tf_lookup_timeout_sec": self.tf_lookup_timeout_sec,
            "received_cloud_msgs": self.received_cloud_msgs,
            "received_odom_msgs": self.received_odom_msgs,
            "accepted_frames": self.accepted_frames,
            "accepted_direct_pose": self.accepted_direct_pose,
            "accepted_tf_pose": self.accepted_tf_pose,
            "accepted_legacy_pose": self.accepted_legacy_pose,
            "skipped_no_odom": self.skipped_no_odom,
            "skipped_stale_odom": self.skipped_stale_odom,
            "skipped_empty_cloud": self.skipped_empty_cloud,
            "skipped_filtered_empty": self.skipped_filtered_empty,
            "skipped_tf_unavailable": self.skipped_tf_unavailable,
            "dropped_pending_frames": self.dropped_pending_frames,
            "pending_frames": len(self.pending_frames),
            "warmup_frames_collected": len(self.warmup_raw_frames),
            "initialized": self.initialized,
            "integrated_frame_count": self.integrated_frame_count,
            "train_iter": self.train_iter,
            "recent_buffer_size": len(self.recent_frames),
            "replay_buffer_size": len(self.replay_frames),
            "observed_cloud_frames": sorted(self.observed_cloud_frames),
            "observed_odom_frames": sorted(self.observed_odom_frames),
            "observed_odom_child_frames": sorted(self.observed_odom_child_frames),
            "last_tf_error": self.last_tf_error,
            "anchor_count": 0 if self.local_sdfs is None else int(self.local_sdfs.positions.shape[0]),
        }
        with open(self.status_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def close(self) -> None:
        self.export_anchor_artifacts()
        self.write_status_json()
        if self.visualizer is not None:
            self.visualizer.destroy_window()
            self.visualizer = None
        self.destroy_node()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.conf)
    _require_file(config_path, "Config file")

    with open(config_path, encoding="utf-8") as config_file:
        cfg = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    out_dir = resolve_output_dir(cfg["Dataset"], args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = OnlineSLNRNode(
        cfg=cfg,
        out_dir=out_dir,
        seed=args.seed,
        enable_visualization=not args.no_vis,
    )

    deadline = None
    if args.max_runtime_sec is not None:
        deadline = time.monotonic() + max(0.1, float(args.max_runtime_sec))

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            node.process_pending_frames()
            node.maybe_run_optimization()
            if deadline is not None and time.monotonic() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
