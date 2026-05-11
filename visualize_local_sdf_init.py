#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:  # pragma: no cover - optional runtime dependency
    vtk = None
    numpy_support = None

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - optional runtime dependency
    o3d = None


REPO_ROOT = Path(__file__).resolve().parent


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize saved SLNR local-SDF initialization artifacts.",
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
        "--backend",
        choices=("auto", "vtk", "open3d"),
        default="auto",
        help="Rendering backend. 'vtk' provides higher-quality glyph and splat rendering.",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "points", "surfels", "voxels"),
        default="hybrid",
        help="Visualization style. 'hybrid' combines splats, normal glyphs, support patches, and optional voxels.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="Maximum number of anchor points to render.",
    )
    parser.add_argument(
        "--max-normals",
        type=int,
        default=800,
        help="Maximum number of normal glyphs to render.",
    )
    parser.add_argument(
        "--normal-scale",
        type=float,
        default=0.35,
        help="Length of rendered normal arrows in meters.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Requested point size in viewer units.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Voxel edge length in meters for --mode voxels or hybrid. 0 uses an inferred value.",
    )
    parser.add_argument(
        "--voxel-opacity",
        type=float,
        default=1.0,
        help="Opacity of voxel cells in hybrid/voxel mode.",
    )
    parser.add_argument(
        "--surfel-scale",
        type=float,
        default=1.35,
        help="Multiplier applied to estimated local support patch radii.",
    )
    parser.add_argument(
        "--surfel-opacity",
        type=float,
        default=1.0,
        help="Opacity of support patches in surfel/hybrid mode.",
    )
    parser.add_argument(
        "--max-surfels",
        type=int,
        default=3000,
        help="Maximum number of support patches to render in VTK surfel/hybrid modes.",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=12000,
        help="Maximum number of occupied voxels to render in VTK voxel/hybrid modes.",
    )
    parser.add_argument(
        "--point-style",
        choices=("auto", "simple", "gaussian"),
        default="auto",
        help="VTK anchor point rendering style. 'simple' is the most robust.",
    )
    parser.add_argument(
        "--hide-normals",
        action="store_true",
        help="Render anchors only.",
    )
    parser.add_argument(
        "--hide-support",
        action="store_true",
        help="Do not render local support patches.",
    )
    parser.add_argument(
        "--hide-voxels",
        action="store_true",
        help="Do not render voxel occupancy, even in hybrid mode.",
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
        help="Do not render the coordinate frame/orientation marker.",
    )
    parser.add_argument(
        "--save-screenshot",
        type=str,
        default=None,
        help="Optional PNG path. Supported by the VTK backend.",
    )
    parser.add_argument(
        "--export-voxel-centers",
        type=str,
        default=None,
        help="Optional .ply path for colored voxel centers.",
    )
    parser.add_argument(
        "--export-voxel-mesh",
        type=str,
        default=None,
        help="Optional .ply path for colored cube voxels as a triangle mesh.",
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


def compute_scalar_range(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def infer_voxel_size(positions: np.ndarray, scales_linear: Optional[np.ndarray]) -> float:
    if scales_linear is not None and scales_linear.size > 0:
        xy = scales_linear[:, :2]
        return max(float(np.median(np.max(xy, axis=1))) * 1.8, 0.05)

    extent = compute_extent(positions)
    diag = float(np.linalg.norm(extent))
    if diag <= 1e-6:
        return 0.1
    return max(diag / 80.0, 0.05)


def build_anchor_point_cloud(positions: np.ndarray, normals: np.ndarray):
    if o3d is None:
        raise RuntimeError("Open3D is required for the Open3D backend.")
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return point_cloud


def build_normal_lines(
    positions: np.ndarray,
    normals: np.ndarray,
    normal_scale: float,
):
    if o3d is None:
        raise RuntimeError("Open3D is required for the Open3D backend.")
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


def build_open3d_voxels(
    voxel_centers: np.ndarray,
    voxel_normals: np.ndarray,
    voxel_size: float,
):
    if o3d is None:
        raise RuntimeError("Open3D is required for the Open3D backend.")
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(voxel_centers.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(
        np.clip((voxel_normals + 1.0) * 0.5, 0.0, 1.0).astype(np.float64)
    )
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=float(voxel_size))
    return voxel_grid


def voxel_normals_to_rgb(voxel_normals: np.ndarray) -> np.ndarray:
    return np.clip((voxel_normals + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)


def vtk_array(data: np.ndarray, name: str, deep: bool = True):
    array = numpy_support.numpy_to_vtk(data, deep=deep)
    array.SetName(name)
    return array


def build_vtk_polydata(
    positions: np.ndarray,
    normals: np.ndarray,
    colors_rgb: np.ndarray,
    scalar_values: np.ndarray,
    vector_values: Optional[np.ndarray] = None,
    scale_values: Optional[np.ndarray] = None,
):
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(positions.astype(np.float32), deep=True))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    point_data = polydata.GetPointData()
    point_data.SetNormals(vtk_array(normals.astype(np.float32), "Normals"))
    point_data.SetScalars(vtk_array(colors_rgb.astype(np.uint8), "RGB"))
    point_data.AddArray(vtk_array(scalar_values.astype(np.float32), "Scalar"))

    if vector_values is not None:
        point_data.AddArray(vtk_array(vector_values.astype(np.float32), "Vectors"))

    if scale_values is not None:
        point_data.AddArray(vtk_array(scale_values.astype(np.float32), "Scale"))

    verts = vtk.vtkCellArray()
    count = positions.shape[0]
    verts.AllocateExact(count, count * 2)
    for idx in range(count):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(idx)
    polydata.SetVerts(verts)
    return polydata


def sample_rows(data: np.ndarray, max_count: int) -> np.ndarray:
    if max_count <= 0 or data.shape[0] <= max_count:
        return data
    return data[sample_indices(data.shape[0], max_count)]


def axis_angle_to_quaternion(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    half_angle = angle * 0.5
    sin_half = np.sin(half_angle)
    quat = np.zeros((axis.shape[0], 4), dtype=np.float32)
    quat[:, 0] = axis[:, 0] * sin_half
    quat[:, 1] = axis[:, 1] * sin_half
    quat[:, 2] = axis[:, 2] * sin_half
    quat[:, 3] = np.cos(half_angle)
    return quat


def normals_to_quaternions(normals: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    normals = normalize_vectors(normals.astype(np.float32))
    cross = np.cross(np.tile(z_axis[None, :], (normals.shape[0], 1)), normals)
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)
    dot = np.clip(normals @ z_axis, -1.0, 1.0)
    angle = np.arccos(dot).astype(np.float32)

    fallback = np.zeros_like(cross)
    fallback[:, 0] = 1.0

    axis = np.where(cross_norm > 1e-7, cross / np.clip(cross_norm, 1e-7, None), fallback)
    opposite = dot < -0.9999
    axis[opposite] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    angle[opposite] = np.pi
    return axis_angle_to_quaternion(axis.astype(np.float32), angle)


def make_lookup_table() -> "vtk.vtkLookupTable":
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetHueRange(0.67, 0.0)
    lut.SetSaturationRange(0.85, 0.95)
    lut.SetValueRange(0.95, 1.0)
    lut.Build()
    return lut


def add_point_splats(
    renderer,
    positions: np.ndarray,
    normals: np.ndarray,
    scales_linear: Optional[np.ndarray],
    point_size: float,
):
    colors = np.clip((normals + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    scalar_values = positions[:, 2] if positions.size else np.zeros(0, dtype=np.float32)
    if scales_linear is not None and scales_linear.size > 0:
        support_scale = np.max(scales_linear[:, :2], axis=1)
    else:
        support_scale = np.ones(positions.shape[0], dtype=np.float32)

    polydata = build_vtk_polydata(
        positions=positions,
        normals=normals,
        colors_rgb=colors,
        scalar_values=scalar_values,
        scale_values=support_scale,
    )

    mapper = vtk.vtkPointGaussianMapper()
    mapper.SetInputData(polydata)
    mapper.SetColorModeToDirectScalars()
    mapper.SetScaleArray("Scale")
    mapper.SetScaleFactor(float(point_size) * 0.02)
    mapper.EmissiveOff()
    mapper.SetSplatShaderCode(
        "//VTK::Color::Impl\n"
        "float dist2 = dot(offsetVCVSOutput.xy, offsetVCVSOutput.xy);\n"
        "if (dist2 > 1.0) { discard; }\n"
        "float gaussian = exp(-dist2 * 3.0);\n"
        "ambientColor *= gaussian;\n"
        "diffuseColor *= gaussian;\n"
        "opacity = opacity * gaussian;\n"
    )

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1.0)
    renderer.AddActor(actor)
    return actor


def add_simple_points(
    renderer,
    positions: np.ndarray,
    normals: np.ndarray,
    point_size: float,
):
    colors = np.clip((normals + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    polydata = build_vtk_polydata(
        positions=positions,
        normals=normals,
        colors_rgb=colors,
        scalar_values=np.zeros(positions.shape[0], dtype=np.float32),
    )

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToPoints()
    actor.GetProperty().RenderPointsAsSpheresOff()
    actor.GetProperty().SetPointSize(float(point_size))
    actor.GetProperty().SetAmbient(0.35)
    actor.GetProperty().SetDiffuse(0.65)
    renderer.AddActor(actor)
    return actor


def add_normal_glyphs(
    renderer,
    positions: np.ndarray,
    normals: np.ndarray,
    normal_scale: float,
):
    colors = np.clip((normals + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    scalar_values = np.linalg.norm(normals, axis=1)
    polydata = build_vtk_polydata(
        positions=positions,
        normals=normals,
        colors_rgb=colors,
        scalar_values=scalar_values,
        vector_values=normals,
        scale_values=np.full(positions.shape[0], normal_scale, dtype=np.float32),
    )

    arrow = vtk.vtkArrowSource()
    arrow.SetTipResolution(24)
    arrow.SetShaftResolution(20)
    arrow.SetTipLength(0.28)
    arrow.SetTipRadius(0.11)
    arrow.SetShaftRadius(0.035)

    glyph_mapper = vtk.vtkGlyph3DMapper()
    glyph_mapper.SetInputData(polydata)
    glyph_mapper.SetSourceConnection(arrow.GetOutputPort())
    glyph_mapper.SetOrientationArray("Normals")
    glyph_mapper.SetOrientationModeToDirection()
    glyph_mapper.OrientOn()
    glyph_mapper.SetScaleArray("Scale")
    glyph_mapper.SetScaleModeToScaleByMagnitude()
    glyph_mapper.SetScaleFactor(1.0)
    glyph_mapper.SetColorModeToDirectScalars()
    glyph_mapper.SetInputArrayToProcess(0, 0, 0, 0, "RGB")

    actor = vtk.vtkActor()
    actor.SetMapper(glyph_mapper)
    actor.GetProperty().SetOpacity(1.0)
    renderer.AddActor(actor)
    return actor


def add_support_surfels(
    renderer,
    positions: np.ndarray,
    normals: np.ndarray,
    scales_linear: Optional[np.ndarray],
    surfel_scale: float,
    surfel_opacity: float,
):
    if positions.shape[0] == 0:
        return None

    if scales_linear is not None and scales_linear.size > 0:
        xy_support = np.max(scales_linear[:, :2], axis=1) * surfel_scale
    else:
        xy_support = np.full(positions.shape[0], surfel_scale * 0.1, dtype=np.float32)

    scalar_values = xy_support
    colors = np.zeros((positions.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = 244
    colors[:, 1] = 194
    colors[:, 2] = 96

    polydata = build_vtk_polydata(
        positions=positions,
        normals=normals,
        colors_rgb=colors,
        scalar_values=scalar_values,
        scale_values=xy_support.astype(np.float32),
    )
    quaternions = normals_to_quaternions(normals)
    polydata.GetPointData().AddArray(vtk_array(quaternions, "Orientation"))

    disk = vtk.vtkDiskSource()
    disk.SetInnerRadius(0.0)
    disk.SetOuterRadius(1.0)
    disk.SetCircumferentialResolution(24)
    disk.SetRadialResolution(2)

    glyph_mapper = vtk.vtkGlyph3DMapper()
    glyph_mapper.SetInputData(polydata)
    glyph_mapper.SetSourceConnection(disk.GetOutputPort())
    glyph_mapper.SetOrientationArray("Orientation")
    glyph_mapper.SetOrientationModeToQuaternion()
    glyph_mapper.OrientOn()
    glyph_mapper.SetScaleArray("Scale")
    glyph_mapper.SetScaleModeToScaleByMagnitude()
    glyph_mapper.SetScaleFactor(1.0)
    glyph_mapper.SetColorModeToDirectScalars()
    glyph_mapper.SetInputArrayToProcess(0, 0, 0, 0, "RGB")

    actor = vtk.vtkActor()
    actor.SetMapper(glyph_mapper)
    actor.GetProperty().SetOpacity(float(surfel_opacity))
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetAmbient(0.2)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetSpecular(0.15)
    renderer.AddActor(actor)
    return actor


def voxelize_positions(
    positions: np.ndarray,
    normals: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if positions.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    min_corner = positions.min(axis=0)
    grid_origin = np.floor(min_corner / voxel_size) * voxel_size

    coords = np.floor((positions - grid_origin) / voxel_size).astype(np.int64)
    unique_coords, inverse, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    centers = grid_origin.astype(np.float32) + (unique_coords.astype(np.float32) + 0.5) * float(voxel_size)
    densities = counts.astype(np.float32)
    normal_sums = np.zeros((unique_coords.shape[0], 3), dtype=np.float32)
    np.add.at(normal_sums, inverse, normals.astype(np.float32))
    avg_normals = normalize_vectors(normal_sums)
    return centers, densities, avg_normals


def subsample_voxels(
    centers: np.ndarray,
    densities: np.ndarray,
    avg_normals: np.ndarray,
    max_voxels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_voxels <= 0 or centers.shape[0] <= max_voxels:
        return centers, densities, avg_normals
    keep = np.argsort(densities)[-max_voxels:]
    keep = np.sort(keep)
    return centers[keep], densities[keep], avg_normals[keep]


def prepare_voxel_bundle(
    positions: np.ndarray,
    normals: np.ndarray,
    voxel_size: float,
    max_voxels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers, densities, avg_normals = voxelize_positions(positions, normals, voxel_size)
    centers, densities, avg_normals = subsample_voxels(centers, densities, avg_normals, max_voxels)
    colors = voxel_normals_to_rgb(avg_normals)
    return centers, densities, avg_normals, colors


def write_voxel_centers_ply(
    path: Path,
    voxel_centers: np.ndarray,
    voxel_normals: np.ndarray,
    voxel_colors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {voxel_centers.shape[0]}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        for point, normal, color in zip(voxel_centers, voxel_normals, voxel_colors):
            ply_file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_voxel_mesh_ply(
    path: Path,
    voxel_centers: np.ndarray,
    voxel_size: float,
    voxel_colors: np.ndarray,
) -> None:
    cube_offsets = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    ) * float(voxel_size)
    cube_faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )
    vertex_count = int(voxel_centers.shape[0] * 8)
    face_count = int(voxel_centers.shape[0] * 12)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {vertex_count}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write(f"element face {face_count}\n")
        ply_file.write("property list uchar int vertex_indices\n")
        ply_file.write("end_header\n")

        for center, color in zip(voxel_centers, voxel_colors):
            vertices = center[None, :] + cube_offsets
            for vertex in vertices:
                ply_file.write(
                    f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )

        for voxel_index in range(voxel_centers.shape[0]):
            base = voxel_index * 8
            for face in cube_faces:
                ply_file.write(f"3 {base + face[0]} {base + face[1]} {base + face[2]}\n")


def add_voxel_actor(
    renderer,
    voxel_centers: np.ndarray,
    voxel_densities: np.ndarray,
    voxel_normals: np.ndarray,
    voxel_colors: np.ndarray,
    voxel_size: float,
    opacity: float,
):
    if voxel_centers.shape[0] == 0:
        return None

    polydata = build_vtk_polydata(
        positions=voxel_centers,
        normals=voxel_normals,
        colors_rgb=voxel_colors,
        scalar_values=voxel_densities,
        scale_values=np.full(voxel_centers.shape[0], voxel_size, dtype=np.float32),
    )

    cube = vtk.vtkCubeSource()
    cube.SetXLength(1.0)
    cube.SetYLength(1.0)
    cube.SetZLength(1.0)

    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(polydata)
    mapper.SetSourceConnection(cube.GetOutputPort())
    mapper.SetScaleArray("Scale")
    mapper.SetScaleModeToScaleByMagnitude()
    mapper.SetScaleFactor(1.0)
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(float(opacity))
    actor.GetProperty().EdgeVisibilityOff()
    actor.GetProperty().SetInterpolationToPhong()
    renderer.AddActor(actor)
    return actor


def setup_vtk_camera(renderer, positions: np.ndarray) -> None:
    camera = renderer.GetActiveCamera()
    if positions.shape[0] == 0:
        camera.SetPosition(2.5, -2.5, 2.0)
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetViewUp(0.0, 0.0, 1.0)
        return

    center = positions.mean(axis=0)
    extent = compute_extent(positions)
    diag = float(np.linalg.norm(extent))
    if diag <= 1e-6:
        diag = 1.0

    offset = np.array([1.35, -1.55, 0.95], dtype=np.float64)
    offset = offset / np.linalg.norm(offset) * diag * 1.7

    camera.SetFocalPoint(*center.tolist())
    camera.SetPosition(*(center + offset).tolist())
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetClippingRange(0.01, max(diag * 10.0, 10.0))


def add_orientation_marker(interactor) -> None:
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(0.8, 0.8, 0.8)

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(interactor)
    widget.SetViewport(0.0, 0.0, 0.18, 0.18)
    widget.SetEnabled(1)
    widget.InteractiveOff()


def save_vtk_screenshot(render_window, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_filter = vtk.vtkWindowToImageFilter()
    image_filter.SetInput(render_window)
    image_filter.SetInputBufferTypeToRGB()
    image_filter.ReadFrontBufferOff()
    image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(image_filter.GetOutputPort())
    writer.Write()


def validate_vtk_runtime(render_window, save_screenshot: Optional[str]) -> None:
    class_name = render_window.GetClassName()
    needs_x11 = class_name == "vtkXOpenGLRenderWindow"
    has_display = bool(os.environ.get("DISPLAY"))
    if needs_x11 and not has_display:
        target = "interactive display" if save_screenshot is None else "this VTK screenshot path"
        raise RuntimeError(
            f"VTK runtime requires an X11 display for {target}, but DISPLAY is not set. "
            "Run this script on a machine with a desktop session, or use the Open3D backend there."
        )


def render_vtk_scene(
    positions_view: np.ndarray,
    normals_view: np.ndarray,
    positions_normals: np.ndarray,
    normals_normals: np.ndarray,
    scalings_view: Optional[np.ndarray],
    voxel_bundle: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
    args: argparse.Namespace,
) -> None:
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)

    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("SLNR Local SDF Init")
    render_window.SetSize(1680, 980)
    render_window.AddRenderer(renderer)
    render_window.SetMultiSamples(0)
    if args.save_screenshot is not None:
        render_window.SetOffScreenRendering(1)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    validate_vtk_runtime(render_window, args.save_screenshot)

    light_kit = vtk.vtkLightKit()
    light_kit.SetKeyLightIntensity(0.95)
    light_kit.SetKeyToFillRatio(2.1)
    light_kit.SetKeyToBackRatio(2.8)
    light_kit.SetKeyToHeadRatio(3.2)
    light_kit.SetFillLightWarmth(0.5)
    light_kit.SetBackLightWarmth(0.45)
    light_kit.SetHeadLightWarmth(0.4)
    light_kit.AddLightsToRenderer(renderer)

    point_style = args.point_style
    if point_style == "auto":
        point_style = "simple"

    if args.mode in ("hybrid", "points", "surfels"):
        if point_style == "gaussian":
            add_point_splats(
                renderer,
                positions_view,
                normals_view,
                scalings_view,
                point_size=args.point_size,
            )
        else:
            add_simple_points(
                renderer,
                positions_view,
                normals_view,
                point_size=args.point_size,
            )

    if not args.hide_support and args.mode in ("hybrid", "surfels"):
        support_indices = sample_indices(positions_view.shape[0], args.max_surfels)
        add_support_surfels(
            renderer,
            positions_view[support_indices],
            normals_view[support_indices],
            scales_linear=scalings_view[support_indices] if scalings_view is not None else None,
            surfel_scale=args.surfel_scale,
            surfel_opacity=args.surfel_opacity,
        )

    if not args.hide_normals and positions_normals.shape[0] > 0 and args.mode in ("hybrid", "points", "surfels"):
        add_normal_glyphs(
            renderer,
            positions_normals,
            normals_normals,
            normal_scale=args.normal_scale,
        )

    if not args.hide_voxels and args.mode in ("hybrid", "voxels"):
        if voxel_bundle is None:
            raise RuntimeError("Voxel rendering requested but voxel bundle was not prepared.")
        voxel_centers, voxel_densities, voxel_normals, voxel_colors, voxel_size = voxel_bundle
        add_voxel_actor(
            renderer,
            voxel_centers,
            voxel_densities,
            voxel_normals,
            voxel_colors,
            voxel_size=voxel_size,
            opacity=args.voxel_opacity,
        )
        print(f"Voxel size   : {voxel_size:.4f}")
        print("Voxel color  : normal")
        print("Voxel fill   : bbox-grid occupied by support points")

    if not args.no_frame:
        add_orientation_marker(interactor)

    setup_vtk_camera(renderer, positions_view)
    renderer.ResetCameraClippingRange()
    render_window.Render()

    if args.save_screenshot is not None:
        save_vtk_screenshot(render_window, Path(args.save_screenshot).expanduser().resolve())
        print(f"Screenshot   : {Path(args.save_screenshot).expanduser().resolve()}")
        return

    interactor.Initialize()
    interactor.Start()


def render_open3d_scene(
    positions_view: np.ndarray,
    normals_view: np.ndarray,
    positions_normals: np.ndarray,
    normals_normals: np.ndarray,
    positions_full: np.ndarray,
    voxel_bundle: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
    args: argparse.Namespace,
) -> None:
    if o3d is None:
        raise RuntimeError("Open3D backend requested but Open3D is not installed.")

    geometries = []
    point_cloud = build_anchor_point_cloud(positions_view, normals_view)
    geometries.append(point_cloud)

    if not args.hide_normals and positions_normals.shape[0] > 0:
        geometries.append(build_normal_lines(positions_normals, normals_normals, args.normal_scale))

    if not args.hide_voxels and args.mode in ("hybrid", "voxels"):
        if voxel_bundle is None:
            raise RuntimeError("Voxel rendering requested but voxel bundle was not prepared.")
        voxel_centers, _, voxel_normals, _, voxel_size = voxel_bundle
        geometries.append(build_open3d_voxels(voxel_centers, voxel_normals, voxel_size))
        print(f"Voxel size   : {voxel_size:.4f}")
        print(f"Voxel color  : normal")
        print("Voxel fill   : bbox-grid occupied by support points")

    if not args.no_frame:
        extent = compute_extent(positions_full)
        frame_size = max(float(np.max(extent)) * 0.1, args.normal_scale * 2.0, 0.2)
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SLNR Local SDF Init", width=1600, height=900)
    render_option = vis.get_render_option()
    render_option.point_size = float(args.point_size)
    render_option.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    for geometry in geometries:
        vis.add_geometry(geometry)

    vis.run()
    vis.destroy_window()


def choose_backend(requested: str) -> str:
    if requested == "vtk":
        if vtk is None:
            raise RuntimeError("VTK backend requested but vtk is not installed.")
        return "vtk"
    if requested == "open3d":
        if o3d is None:
            raise RuntimeError("Open3D backend requested but open3d is not installed.")
        return "open3d"
    if vtk is not None:
        return "vtk"
    if o3d is not None:
        return "open3d"
    raise RuntimeError("No supported visualization backend is installed. Install vtk or open3d.")


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
    scales_linear = np.exp(scalings).astype(np.float32) if scalings is not None else None

    point_indices = sample_indices(positions.shape[0], args.max_points)
    positions_view = positions[point_indices]
    normals_view = normals[point_indices]
    scalings_view = scales_linear[point_indices] if scales_linear is not None else None

    normal_indices = sample_indices(positions_view.shape[0], args.max_normals)
    positions_normals = positions_view[normal_indices]
    normals_normals = normals_view[normal_indices]

    need_voxels = (
        (not args.hide_voxels and args.mode in ("hybrid", "voxels"))
        or args.export_voxel_centers is not None
        or args.export_voxel_mesh is not None
    )
    voxel_bundle = None
    if need_voxels:
        voxel_size = float(args.voxel_size) if args.voxel_size > 0.0 else infer_voxel_size(positions_view, scales_linear)
        voxel_centers, voxel_densities, voxel_normals, voxel_colors = prepare_voxel_bundle(
            positions_view,
            normals_view,
            voxel_size,
            args.max_voxels,
        )
        voxel_bundle = (voxel_centers, voxel_densities, voxel_normals, voxel_colors, voxel_size)

    backend = choose_backend(args.backend)

    print(f"Artifact dir : {artifact_dir}")
    print(f"NPZ path     : {npz_path}")
    print(f"Backend      : {backend}")
    print(f"Mode         : {args.mode}")
    print(f"Anchor count : {positions.shape[0]}")
    print(f"Rendered pts : {positions_view.shape[0]}")
    print(f"Rendered nrm : {0 if args.hide_normals else positions_normals.shape[0]}")

    if scales_linear is not None and scales_linear.size > 0:
        print(f"Scale mean   : {scales_linear.mean(axis=0).tolist()}")
        print(f"Scale min    : {scales_linear.min(axis=0).tolist()}")
        print(f"Scale max    : {scales_linear.max(axis=0).tolist()}")

    if summary_path is not None:
        with open(summary_path, encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        print("Summary json : {}".format(summary_path))
        print(json.dumps(summary, indent=2))

    if args.export_voxel_centers is not None:
        if voxel_bundle is None:
            raise RuntimeError("Voxel center export requested but voxel bundle was not prepared.")
        voxel_centers_path = Path(args.export_voxel_centers).expanduser().resolve()
        write_voxel_centers_ply(voxel_centers_path, voxel_bundle[0], voxel_bundle[2], voxel_bundle[3])
        print(f"Voxel centers: {voxel_centers_path}")

    if args.export_voxel_mesh is not None:
        if voxel_bundle is None:
            raise RuntimeError("Voxel mesh export requested but voxel bundle was not prepared.")
        voxel_mesh_path = Path(args.export_voxel_mesh).expanduser().resolve()
        write_voxel_mesh_ply(voxel_mesh_path, voxel_bundle[0], voxel_bundle[4], voxel_bundle[3])
        print(f"Voxel mesh   : {voxel_mesh_path}")

    if backend == "vtk":
        render_vtk_scene(
            positions_view=positions_view,
            normals_view=normals_view,
            positions_normals=positions_normals,
            normals_normals=normals_normals,
            scalings_view=scalings_view,
            voxel_bundle=voxel_bundle,
            args=args,
        )
        return

    render_open3d_scene(
        positions_view=positions_view,
        normals_view=normals_view,
        positions_normals=positions_normals,
        normals_normals=normals_normals,
        positions_full=positions,
        voxel_bundle=voxel_bundle,
        args=args,
    )


if __name__ == "__main__":
    main()
