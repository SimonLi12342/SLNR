#!/usr/bin/env python3
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class CapturedFrame:
    points: np.ndarray
    normals: Optional[np.ndarray]
    pose: np.ndarray
    cloud_stamp_sec: Optional[float]
    odom_stamp_sec: Optional[float]


class ROS2CapturedDataset:
    def __init__(self, frames: List[CapturedFrame]):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        frame = self.frames[idx]
        normals = None if frame.normals is None else frame.normals.copy()
        return frame.points.copy(), normals, frame.pose.copy()


def _load_calibration(calib_file: Optional[str]) -> np.ndarray:
    if calib_file is None:
        return np.eye(4, dtype=np.float64)

    calib = {}
    with open(calib_file, encoding="utf-8") as calib_handle:
        for line in calib_handle:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            pose = np.zeros((4, 4), dtype=np.float64)
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            calib[key] = pose

    return calib.get("Tr", np.eye(4, dtype=np.float64))


def _quat_xyzw_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _pose_from_position_quaternion(position, orientation) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _quat_xyzw_to_rotmat(orientation.x, orientation.y, orientation.z, orientation.w)
    pose[:3, 3] = np.array([position.x, position.y, position.z], dtype=np.float64)
    return pose


def _pose_from_transform(transform) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _quat_xyzw_to_rotmat(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    )
    pose[:3, 3] = np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        dtype=np.float64,
    )
    return pose


def _stamp_to_sec(stamp) -> Optional[float]:
    if stamp is None:
        return None
    if hasattr(stamp, "nanosec") and hasattr(stamp, "sec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return None


def _resolve_pose_frame(
    odom_pose: np.ndarray,
    odom_pose_frame: str,
    calib_tr: np.ndarray,
) -> np.ndarray:
    if odom_pose_frame == "lidar":
        return odom_pose
    if odom_pose_frame == "base":
        calib_inv = np.linalg.inv(calib_tr)
        return calib_inv @ odom_pose @ calib_tr
    raise ValueError(f"Unsupported ROS2 odom_pose_frame: {odom_pose_frame}")


def _normalize_frame_id(frame_id: Optional[str]) -> str:
    if frame_id is None:
        return ""
    return str(frame_id).strip().lstrip("/")


def _extract_points_and_normals(point_cloud2_module, msg) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    def to_float_matrix(rows, field_names) -> np.ndarray:
        data = np.asarray(rows)
        if data.size == 0:
            return np.empty((0, len(field_names)), dtype=np.float32)
        if data.dtype.names is not None:
            columns = [np.asarray(data[name], dtype=np.float32) for name in field_names]
            return np.stack(columns, axis=1)
        return np.asarray(rows, dtype=np.float32)

    field_names = {field.name for field in msg.fields}
    has_normals = {"normal_x", "normal_y", "normal_z"}.issubset(field_names)

    if has_normals:
        request_fields = ("x", "y", "z", "normal_x", "normal_y", "normal_z")
        rows = list(
            point_cloud2_module.read_points(
                msg,
                field_names=request_fields,
                skip_nans=True,
            )
        )
        if not rows:
            return np.empty((0, 3), dtype=np.float32), None
        data = to_float_matrix(rows, request_fields)
        return data[:, :3], data[:, 3:]

    request_fields = ("x", "y", "z")
    rows = list(point_cloud2_module.read_points(msg, field_names=request_fields, skip_nans=True))
    if not rows:
        return np.empty((0, 3), dtype=np.float32), None
    return to_float_matrix(rows, request_fields), None


def _filter_points(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    min_range: float,
    max_range: float,
    use_filter: bool,
    sor_nn: int,
    sor_std: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points.size == 0:
        return points.astype(np.float32), normals

    if use_filter:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        point_cloud, keep_indices = point_cloud.remove_statistical_outlier(
            sor_nn,
            sor_std,
            print_progress=False,
        )
        keep_indices = np.asarray(keep_indices, dtype=np.int64)
        points = np.asarray(point_cloud.points, dtype=np.float32)
        if normals is not None:
            normals = normals[keep_indices]

    ray_length = np.linalg.norm(points, axis=1)
    mask = (ray_length > min_range) & (ray_length <= max_range)
    points = points[mask].astype(np.float32)
    if normals is not None:
        normals = normals[mask].astype(np.float32)

    return points, normals


def capture_ros2_dataset(
    *,
    pointcloud_topic: str,
    odom_topic: str,
    max_frames: int,
    capture_timeout_sec: float,
    max_odom_age_sec: float,
    odom_pose_frame: str,
    pointcloud_qos: str,
    odom_qos: str,
    use_tf: bool,
    tf_lookup_timeout_sec: float,
    min_range: float,
    max_range: float,
    use_filter: bool,
    sor_nn: int,
    sor_std: float,
    calib_file: Optional[str],
) -> Tuple[ROS2CapturedDataset, Dict[str, object]]:
    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs_py import point_cloud2
        from tf2_ros import Buffer, TransformListener
    except ImportError as exc:
        raise ImportError(
            "ROS2 mode requires rclpy, sensor_msgs, nav_msgs, sensor_msgs_py, and tf2_ros in the active Python environment."
        ) from exc

    def make_qos(name: str):
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

    class CaptureNode(Node):
        def __init__(self) -> None:
            super().__init__("slnr_local_sdf_init_capture")
            self.frames: List[CapturedFrame] = []
            self.latest_odom_pose: Optional[np.ndarray] = None
            self.latest_odom_stamp_sec: Optional[float] = None
            self.latest_odom_frame_id = ""
            self.latest_odom_child_frame_id = ""
            self.received_odom_msgs = 0
            self.received_cloud_msgs = 0
            self.skipped_no_odom = 0
            self.skipped_stale_odom = 0
            self.skipped_empty_cloud = 0
            self.skipped_filtered_empty = 0
            self.skipped_tf_unavailable = 0
            self.accepted_direct_pose = 0
            self.accepted_tf_pose = 0
            self.accepted_legacy_pose = 0
            self.observed_cloud_frames = set()
            self.observed_odom_frames = set()
            self.observed_odom_child_frames = set()
            self.last_tf_error: Optional[str] = None
            self.reported_pose_mappings = set()
            self.calib_tr = _load_calibration(calib_file)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

            self.create_subscription(PointCloud2, pointcloud_topic, self.on_point_cloud, make_qos(pointcloud_qos))
            self.create_subscription(Odometry, odom_topic, self.on_odom, make_qos(odom_qos))

        def on_odom(self, msg: Odometry) -> None:
            self.received_odom_msgs += 1
            odom_pose = _pose_from_position_quaternion(msg.pose.pose.position, msg.pose.pose.orientation)
            self.latest_odom_pose = odom_pose.astype(np.float32)
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

                if use_tf:
                    try:
                        child_from_cloud = self.tf_buffer.lookup_transform(
                            odom_child_frame,
                            cloud_frame,
                            Time(),
                            timeout=Duration(seconds=max(0.0, float(tf_lookup_timeout_sec))),
                        )
                    except Exception as exc:
                        self.last_tf_error = str(exc)
                        return None, "tf_unavailable"

                    pose = self.latest_odom_pose.astype(np.float64) @ _pose_from_transform(
                        child_from_cloud.transform
                    )
                    return pose.astype(np.float32), "tf"

            pose = _resolve_pose_frame(
                self.latest_odom_pose.astype(np.float64),
                odom_pose_frame,
                self.calib_tr,
            )
            return pose.astype(np.float32), "legacy"

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
            print(
                "ROS2 capture pose mapping: "
                f"world='{mapping[0] or '<unknown>'}', "
                f"odom_child='{mapping[1] or '<unknown>'}', "
                f"cloud='{mapping[2] or '<unknown>'}', "
                f"mode='{mapping[3]}'"
            )

        def on_point_cloud(self, msg: PointCloud2) -> None:
            self.received_cloud_msgs += 1
            if len(self.frames) >= max_frames:
                return

            if self.latest_odom_pose is None:
                self.skipped_no_odom += 1
                return

            cloud_stamp_sec = _stamp_to_sec(msg.header.stamp)
            if (
                cloud_stamp_sec is not None
                and self.latest_odom_stamp_sec is not None
                and abs(cloud_stamp_sec - self.latest_odom_stamp_sec) > max_odom_age_sec
            ):
                self.skipped_stale_odom += 1
                return

            pose, pose_mode = self.resolve_pose_for_cloud(msg.header.frame_id)
            if pose is None:
                if pose_mode == "tf_unavailable":
                    self.skipped_tf_unavailable += 1
                else:
                    self.skipped_no_odom += 1
                return

            points, normals = _extract_points_and_normals(point_cloud2, msg)
            if points.shape[0] == 0:
                self.skipped_empty_cloud += 1
                return

            points, normals = _filter_points(
                points,
                normals,
                min_range=min_range,
                max_range=max_range,
                use_filter=use_filter,
                sor_nn=sor_nn,
                sor_std=sor_std,
            )
            if points.shape[0] == 0:
                self.skipped_filtered_empty += 1
                return

            self.report_pose_mapping(msg.header.frame_id, pose_mode)
            if pose_mode == "direct":
                self.accepted_direct_pose += 1
            elif pose_mode == "tf":
                self.accepted_tf_pose += 1
            else:
                self.accepted_legacy_pose += 1

            self.frames.append(
                CapturedFrame(
                    points=points,
                    normals=normals,
                    pose=pose.copy(),
                    cloud_stamp_sec=cloud_stamp_sec,
                    odom_stamp_sec=self.latest_odom_stamp_sec,
                )
            )

    initialized_here = False
    if not rclpy.ok():
        rclpy.init()
        initialized_here = True

    node = CaptureNode()
    deadline = time.monotonic() + max(0.1, float(capture_timeout_sec))
    next_status_time = time.monotonic() + 2.0
    print(
        "ROS2 capture waiting for frames "
        f"(cloud={pointcloud_topic}, odom={odom_topic}, timeout={capture_timeout_sec}s, "
        f"max_frames={max_frames}, use_tf={use_tf})"
    )
    try:
        while len(node.frames) < max_frames and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if now >= next_status_time:
                print(
                    "ROS2 capture status: "
                    f"cloud_msgs={node.received_cloud_msgs}, "
                    f"odom_msgs={node.received_odom_msgs}, "
                    f"accepted_frames={len(node.frames)}, "
                    f"accepted_direct_pose={node.accepted_direct_pose}, "
                    f"accepted_tf_pose={node.accepted_tf_pose}, "
                    f"accepted_legacy_pose={node.accepted_legacy_pose}, "
                    f"skipped_no_odom={node.skipped_no_odom}, "
                    f"skipped_stale_odom={node.skipped_stale_odom}, "
                    f"skipped_empty_cloud={node.skipped_empty_cloud}, "
                    f"skipped_filtered_empty={node.skipped_filtered_empty}, "
                    f"skipped_tf_unavailable={node.skipped_tf_unavailable}"
                )
                next_status_time = now + 2.0
    finally:
        metadata = {
            "pointcloud_topic": pointcloud_topic,
            "odom_topic": odom_topic,
            "pointcloud_qos": pointcloud_qos,
            "odom_qos": odom_qos,
            "odom_pose_frame": odom_pose_frame,
            "use_tf": bool(use_tf),
            "tf_lookup_timeout_sec": float(tf_lookup_timeout_sec),
            "requested_frame_count": int(max_frames),
            "captured_frame_count": int(len(node.frames)),
            "received_cloud_msgs": int(node.received_cloud_msgs),
            "received_odom_msgs": int(node.received_odom_msgs),
            "accepted_direct_pose": int(node.accepted_direct_pose),
            "accepted_tf_pose": int(node.accepted_tf_pose),
            "accepted_legacy_pose": int(node.accepted_legacy_pose),
            "skipped_no_odom": int(node.skipped_no_odom),
            "skipped_stale_odom": int(node.skipped_stale_odom),
            "skipped_empty_cloud": int(node.skipped_empty_cloud),
            "skipped_filtered_empty": int(node.skipped_filtered_empty),
            "skipped_tf_unavailable": int(node.skipped_tf_unavailable),
            "capture_timeout_sec": float(capture_timeout_sec),
            "max_odom_age_sec": float(max_odom_age_sec),
            "observed_cloud_frames": sorted(node.observed_cloud_frames),
            "observed_odom_frames": sorted(node.observed_odom_frames),
            "observed_odom_child_frames": sorted(node.observed_odom_child_frames),
            "last_tf_error": node.last_tf_error,
            "calib_file": calib_file,
        }
        node.destroy_node()
        if initialized_here:
            rclpy.shutdown()

    if not node.frames:
        raise RuntimeError(
            "ROS2 capture did not collect any usable frames.\n"
            "Capture diagnostics:\n"
            f"{json.dumps(metadata, indent=2)}\n"
            "Check topic QoS, message timing, TF availability, ROS_DOMAIN_ID, and filtering thresholds."
        )

    return ROS2CapturedDataset(node.frames), metadata
