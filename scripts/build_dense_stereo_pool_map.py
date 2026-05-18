#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENCV_OPTICAL_TO_ROS = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    odom_rows = read_rows(args.odom_csv)
    odom_by_frame = {
        int(row["frame_index"]): row
        for row in odom_rows
        if row.get("frame_index") not in {None, ""}
    }
    if not odom_by_frame:
        raise SystemExit(f"No odometry rows with frame_index in {args.odom_csv}")
    pose_by_frame = build_pose_table(odom_rows, args.orientation_source)
    path_ros_xy = path_xy_from_pose_table(pose_by_frame)

    fx, fy, cx, cy, baseline = camera_params(metadata, args)
    image0_dir = dataset_dir / "image_0"
    image1_dir = dataset_dir / "image_1"
    frame_count = int(metadata.get("frames") or len(list(image0_dir.glob("*.png"))))
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)

    matcher = make_sgbm(args)
    voxel_accumulator: dict[tuple[int, int, int], dict[str, Any]] = {}
    frame_stats: list[dict[str, Any]] = []
    track_id = 0
    for frame_index in range(0, frame_count, max(1, args.frame_stride)):
        row = odom_by_frame.get(frame_index)
        if row is None:
            continue
        if args.success_only and row.get("pose_success") not in {"1", "true", "True"}:
            continue
        left_path = image0_dir / f"{frame_index:06d}.png"
        right_path = image1_dir / f"{frame_index:06d}.png"
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            continue
        left_eq = preprocess(left, args)
        right_eq = preprocess(right, args)
        disparity = matcher.compute(left_eq, right_eq).astype(np.float32) / 16.0
        local_points, pixels = depth_points_from_disparity(disparity, left_eq, fx, fy, cx, cy, baseline, args)
        if len(local_points) == 0:
            frame_stats.append({"frame_index": frame_index, "accepted_points": 0})
            continue
        position, rotation = pose_by_frame.get(frame_index, pose_from_row(row))
        world_points = (rotation @ local_points.T).T + position
        if args.path_corridor_radius > 0.0 and len(path_ros_xy):
            world_points, pixels = filter_by_path_corridor(world_points, pixels, path_ros_xy, args.path_corridor_radius)
        added = 0
        for point, pixel in zip(world_points, pixels):
            key = tuple(np.floor(point / args.voxel_size).astype(np.int64).tolist())
            bucket = voxel_accumulator.get(key)
            if bucket is None:
                voxel_accumulator[key] = {
                    "sum": point.astype(np.float64),
                    "count": 1,
                    "frame_index": frame_index,
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "track_id": track_id,
                    "u": float(pixel[0]),
                    "v": float(pixel[1]),
                }
                track_id += 1
            else:
                bucket["sum"] += point
                bucket["count"] += 1
            added += 1
        frame_stats.append({"frame_index": frame_index, "accepted_points": int(added)})

    raw_dense_rows = dense_rows_from_voxels(voxel_accumulator)
    dense_rows = filter_rows_by_voxel_age(raw_dense_rows, args.min_voxel_age)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    ros_points = rows_to_ros_points(dense_rows)
    initial_rectangle = fit_top_view_rectangle(ros_points, args)
    plane_filter_stats = {"applied": False, "input_points": len(dense_rows), "kept_points": len(dense_rows)}
    if args.pool_plane_z_band > 0.0:
        dense_rows, ros_points, plane_filter_stats = filter_rows_by_pool_plane_z(
            dense_rows,
            ros_points,
            initial_rectangle,
            args,
        )
        initial_rectangle = fit_top_view_rectangle(ros_points, args)
    boundary_filter_stats = {"applied": False, "input_points": len(dense_rows), "kept_points": len(dense_rows)}
    if args.pool_boundary_filter:
        dense_rows, ros_points, boundary_filter_stats = filter_rows_by_pool_boundary(
            dense_rows,
            ros_points,
            initial_rectangle,
            args,
        )
    write_dense_csv(args.output_csv, dense_rows)

    rectangle = fit_top_view_rectangle(ros_points, args)
    write_rectangle_csv(args.rectangle_csv, rectangle)
    preview_png = write_preview(args.preview_png, ros_points, rectangle)
    summary = {
        "mode": "dense_stereo_pool_map",
        "inputs": {
            "dataset_dir": str(dataset_dir),
            "odom_csv": str(args.odom_csv),
            "dvl_usage": "not_used",
        },
        "camera": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "baseline_m": baseline,
            "focal_scale": args.focal_scale,
            "baseline_scale": args.baseline_scale,
            "depth_scale": args.depth_scale,
        },
        "parameters": jsonable_args(args),
        "frames_processed": len(frame_stats),
        "frame_stats": frame_stats,
        "raw_voxel_points": len(raw_dense_rows),
        "voxel_points": len(dense_rows),
        "plane_z_filter": plane_filter_stats,
        "boundary_filter": boundary_filter_stats,
        "ros_bounds": bounds_dict(ros_points),
        "rectangle": rectangle,
        "outputs": {
            "dense_map_csv": str(args.output_csv),
            "rectangle_csv": str(args.rectangle_csv),
            "preview_png": str(preview_png),
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a semi-dense stereo world map from exported stereo frames and pure VINS odometry. DVL is not used."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--odom-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs/live/latest_dense_stereo_map.csv")
    parser.add_argument("--rectangle-csv", type=Path, default=PROJECT_ROOT / "outputs/live/latest_pool_rectangle_ros.csv")
    parser.add_argument("--summary-json", type=Path, default=PROJECT_ROOT / "outputs/evaluation/dense_stereo_pool_map_summary.json")
    parser.add_argument("--preview-png", type=Path, default=PROJECT_ROOT / "outputs/evaluation/dense_stereo_pool_map_preview.png")
    parser.add_argument("--max-frames", type=int, default=0, help="0 uses all frames.")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-points-per-frame", type=int, default=2500)
    parser.add_argument("--voxel-size", type=float, default=0.06)
    parser.add_argument(
        "--min-voxel-age",
        type=int,
        default=1,
        help="Keep only voxelized map points with at least this many stereo samples.",
    )
    parser.add_argument(
        "--path-corridor-radius",
        type=float,
        default=0.0,
        help="Discard world points farther than this top-view distance from the VINS trajectory. 0 disables the filter.",
    )
    parser.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--orientation-source",
        choices=("csv", "tangent"),
        default="csv",
        help=(
            "csv uses the pose quaternion stored by VIO. tangent locks the camera forward axis to the local "
            "trajectory direction, matching the RViz tangent path display and reducing frame-to-frame attitude scatter."
        ),
    )
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--fx-scale", type=float, default=1.0)
    parser.add_argument("--fy-scale", type=float, default=1.0)
    parser.add_argument("--cx-offset-px", type=float, default=0.0)
    parser.add_argument("--cy-offset-px", type=float, default=0.0)
    parser.add_argument("--baseline-scale", type=float, default=1.0)
    parser.add_argument("--baseline-m", type=float, default=0.0)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--min-depth", type=float, default=0.35)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--min-disparity", type=float, default=2.0)
    parser.add_argument("--max-disparity", type=float, default=128.0)
    parser.add_argument("--min-gradient", type=float, default=4.0)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sgbm-num-disparities", type=int, default=128)
    parser.add_argument("--sgbm-block-size", type=int, default=5)
    parser.add_argument("--sgbm-uniqueness", type=int, default=8)
    parser.add_argument("--sgbm-speckle-window", type=int, default=80)
    parser.add_argument("--sgbm-speckle-range", type=int, default=2)
    parser.add_argument("--rectangle-z-percentile", type=float, default=55.0)
    parser.add_argument("--rectangle-trim-percentile", type=float, default=2.0)
    parser.add_argument(
        "--pool-boundary-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After the first rectangle fit, keep only points close to the fitted rectangular pool boundary.",
    )
    parser.add_argument(
        "--pool-plane-z-band",
        type=float,
        default=0.0,
        help=(
            "Keep only ROS-frame points within this vertical distance from the first fitted rectangle z. "
            "Use this to suppress stereo outliers that make the pool map look vertically exploded. 0 disables it."
        ),
    )
    parser.add_argument(
        "--pool-boundary-band",
        type=float,
        default=0.35,
        help="Top-view distance in meters allowed from the fitted rectangular boundary.",
    )
    parser.add_argument(
        "--pool-boundary-min-points",
        type=int,
        default=2500,
        help="Skip boundary filtering if it would keep fewer points than this.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def camera_params(metadata: dict[str, Any], args: argparse.Namespace) -> tuple[float, float, float, float, float]:
    k = metadata["camera0"]["k"]
    fx = float(k[0]) * args.focal_scale * args.fx_scale
    fy = float(k[4]) * args.focal_scale * args.fy_scale
    cx = float(k[2]) + args.cx_offset_px
    cy = float(k[5]) + args.cy_offset_px
    baseline = float(args.baseline_m) if args.baseline_m > 0.0 else float(metadata.get("baseline_m") or 0.0)
    if baseline <= 0.0:
        baseline = abs(float(metadata["camera1"]["p"][3]) / float(k[0]))
    return fx, fy, cx, cy, baseline * args.baseline_scale


def make_sgbm(args: argparse.Namespace) -> cv2.StereoSGBM:
    num_disparities = max(16, int(math.ceil(args.sgbm_num_disparities / 16.0) * 16))
    block_size = int(args.sgbm_block_size)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=2,
        uniquenessRatio=args.sgbm_uniqueness,
        speckleWindowSize=args.sgbm_speckle_window,
        speckleRange=args.sgbm_speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def preprocess(gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if not args.clahe:
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def depth_points_from_disparity(
    disparity: np.ndarray,
    gray: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = disparity.shape
    ys = np.arange(0, height, max(1, args.pixel_stride), dtype=np.int32)
    xs = np.arange(0, width, max(1, args.pixel_stride), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    u = grid_x.reshape(-1).astype(np.float64)
    v = grid_y.reshape(-1).astype(np.float64)
    d = disparity[grid_y, grid_x].reshape(-1).astype(np.float64)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)[grid_y, grid_x].reshape(-1)
    valid = (
        np.isfinite(d)
        & (d >= args.min_disparity)
        & (d <= args.max_disparity)
        & (gradient >= args.min_gradient)
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    u = u[valid]
    v = v[valid]
    d = d[valid]
    z = fx * baseline / d * args.depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.column_stack([x, y, z]).astype(np.float64)
    depth_valid = (points[:, 2] >= args.min_depth) & (points[:, 2] <= args.max_depth)
    points = points[depth_valid]
    pixels = np.column_stack([u, v]).astype(np.float64)[depth_valid]
    if args.max_points_per_frame > 0 and len(points) > args.max_points_per_frame:
        # Deterministic spatial thinning: keep evenly spaced indices after disparity/depth filtering.
        indices = np.linspace(0, len(points) - 1, args.max_points_per_frame).astype(np.int64)
        points = points[indices]
        pixels = pixels[indices]
    return points, pixels


def pose_from_row(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    position = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
    quaternion = np.array(
        [
            float(row.get("qx") or 0.0),
            float(row.get("qy") or 0.0),
            float(row.get("qz") or 0.0),
            float(row.get("qw") or 1.0),
        ],
        dtype=np.float64,
    )
    rotation = quaternion_to_matrix(quaternion)
    return position, rotation


def build_pose_table(rows: list[dict[str, str]], orientation_source: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    sorted_rows = sorted(
        [row for row in rows if row.get("frame_index") not in {None, ""}],
        key=lambda row: int(row["frame_index"]),
    )
    if orientation_source == "csv":
        return {int(row["frame_index"]): pose_from_row(row) for row in sorted_rows}
    positions = [np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64) for row in sorted_rows]
    table: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, row in enumerate(sorted_rows):
        position = positions[index]
        rotation = tangent_camera_rotation(positions, index)
        table[int(row["frame_index"])] = (position, rotation)
    return table


def tangent_camera_rotation(positions: list[np.ndarray], index: int) -> np.ndarray:
    if len(positions) < 2:
        return np.eye(3, dtype=np.float64)
    candidates: list[np.ndarray] = []
    if 0 < index < len(positions) - 1:
        candidates.append(positions[index + 1] - positions[index - 1])
    if index < len(positions) - 1:
        candidates.append(positions[index + 1] - positions[index])
    if index > 0:
        candidates.append(positions[index] - positions[index - 1])
    for radius in range(2, min(12, len(positions))):
        left = max(0, index - radius)
        right = min(len(positions) - 1, index + radius)
        if right > left:
            candidates.append(positions[right] - positions[left])
    forward = None
    for candidate in candidates:
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-5:
            forward = candidate / norm
            break
    if forward is None:
        return np.eye(3, dtype=np.float64)
    down_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, down_ref))) > 0.95:
        down_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = np.cross(down_ref, forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        return np.eye(3, dtype=np.float64)
    right /= right_norm
    down = np.cross(forward, right)
    down /= max(float(np.linalg.norm(down)), 1e-12)
    return np.column_stack([right, down, forward]).astype(np.float64)


def path_xy_from_pose_table(pose_by_frame: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if not pose_by_frame:
        return np.empty((0, 2), dtype=np.float64)
    raw_points = np.array(
        [pose_by_frame[frame_index][0] for frame_index in sorted(pose_by_frame)],
        dtype=np.float64,
    )
    ros_points = raw_points @ OPENCV_OPTICAL_TO_ROS.T
    return ros_points[:, :2].astype(np.float64)


def filter_by_path_corridor(
    world_points: np.ndarray,
    pixels: np.ndarray,
    path_ros_xy: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(world_points) == 0 or len(path_ros_xy) == 0:
        return world_points, pixels
    points_ros_xy = (world_points @ OPENCV_OPTICAL_TO_ROS.T)[:, :2]
    keep = np.zeros(len(points_ros_xy), dtype=bool)
    chunk = 4096
    radius_sq = radius * radius
    for start in range(0, len(points_ros_xy), chunk):
        block = points_ros_xy[start : start + chunk]
        diff = block[:, None, :] - path_ros_xy[None, :, :]
        min_dist_sq = np.min(np.sum(diff * diff, axis=2), axis=1)
        keep[start : start + chunk] = min_dist_sq <= radius_sq
    return world_points[keep], pixels[keep]


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def dense_rows_from_voxels(voxels: dict[tuple[int, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for bucket in voxels.values():
        point = bucket["sum"] / max(1, int(bucket["count"]))
        rows.append(
            {
                "frame_index": int(bucket["frame_index"]),
                "timestamp_sec": float(bucket["timestamp_sec"]),
                "track_id": int(bucket["track_id"]),
                "point": point,
                "u": float(bucket["u"]),
                "v": float(bucket["v"]),
                "age": int(bucket["count"]),
            }
        )
    rows.sort(key=lambda row: (row["frame_index"], row["track_id"]))
    return rows


def filter_rows_by_voxel_age(rows: list[dict[str, Any]], min_age: int) -> list[dict[str, Any]]:
    if min_age <= 1:
        return rows
    return [row for row in rows if int(row["age"]) >= min_age]


def rows_to_ros_points(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    return np.array([row["point"] for row in rows], dtype=np.float64) @ OPENCV_OPTICAL_TO_ROS.T


def filter_rows_by_pool_plane_z(
    rows: list[dict[str, Any]],
    points_ros: np.ndarray,
    rectangle: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    stats = {
        "applied": False,
        "reason": "",
        "input_points": len(rows),
        "kept_points": len(rows),
        "band_m": float(args.pool_plane_z_band),
        "center_z_m": None,
    }
    if len(rows) == 0 or len(points_ros) == 0:
        stats["reason"] = "empty input"
        return rows, points_ros, stats
    corners = rectangle.get("corners", [])
    if not corners:
        stats["reason"] = "rectangle unavailable"
        return rows, points_ros, stats
    center_z = float(np.median(np.array(corners, dtype=np.float64)[:, 2]))
    band = float(args.pool_plane_z_band)
    keep = np.abs(points_ros[:, 2] - center_z) <= band
    kept_count = int(np.sum(keep))
    stats["kept_points"] = kept_count
    stats["center_z_m"] = center_z
    if kept_count < 4:
        stats["reason"] = "below minimum kept points"
        return rows, points_ros, stats
    stats["applied"] = True
    stats["reason"] = "kept points near fitted pool plane z"
    return [row for row, should_keep in zip(rows, keep) if bool(should_keep)], points_ros[keep], stats


def filter_rows_by_pool_boundary(
    rows: list[dict[str, Any]],
    points_ros: np.ndarray,
    rectangle: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    corners = np.array(rectangle.get("corners", [])[:4], dtype=np.float64)
    stats = {
        "applied": False,
        "reason": "",
        "input_points": len(rows),
        "kept_points": len(rows),
        "band_m": float(args.pool_boundary_band),
    }
    if len(rows) == 0 or len(points_ros) == 0:
        stats["reason"] = "empty input"
        return rows, points_ros, stats
    if len(corners) != 4:
        stats["reason"] = "rectangle unavailable"
        return rows, points_ros, stats

    distances = distances_to_rectangle_boundary(points_ros[:, :2], corners[:, :2])
    keep = distances <= float(args.pool_boundary_band)
    kept_count = int(np.sum(keep))
    stats["kept_points"] = kept_count
    if kept_count < int(args.pool_boundary_min_points):
        stats["reason"] = "below minimum kept points"
        return rows, points_ros, stats

    filtered_rows = [row for row, should_keep in zip(rows, keep) if bool(should_keep)]
    stats["applied"] = True
    stats["reason"] = "kept points near fitted rectangular pool boundary"
    return filtered_rows, points_ros[keep], stats


def distances_to_rectangle_boundary(points_xy: np.ndarray, corners_xy: np.ndarray) -> np.ndarray:
    distances = np.full(len(points_xy), np.inf, dtype=np.float64)
    for index in range(4):
        start = corners_xy[index]
        end = corners_xy[(index + 1) % 4]
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            continue
        t = np.clip(((points_xy - start) @ segment) / length_sq, 0.0, 1.0)
        closest = start + t[:, None] * segment
        diff = points_xy - closest
        distances = np.minimum(distances, np.sqrt(np.sum(diff * diff, axis=1)))
    return distances


def write_dense_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "timestamp_sec", "track_id", "x", "y", "z", "u", "v", "age"])
        for row in rows:
            x, y, z = row["point"]
            writer.writerow(
                [
                    row["frame_index"],
                    f"{row['timestamp_sec']:.9f}",
                    row["track_id"],
                    f"{x:.9f}",
                    f"{y:.9f}",
                    f"{z:.9f}",
                    f"{row['u']:.3f}",
                    f"{row['v']:.3f}",
                    row["age"],
                ]
            )


def fit_top_view_rectangle(points_ros: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    if len(points_ros) < 4:
        return {"corners": [], "area_m2": 0.0, "center": [0.0, 0.0, 0.0], "angle_deg": 0.0}
    z_cut = np.percentile(points_ros[:, 2], args.rectangle_z_percentile)
    selected = points_ros[points_ros[:, 2] <= z_cut]
    if len(selected) < 4:
        selected = points_ros
    xy = selected[:, :2].astype(np.float32)
    trim = max(0.0, min(20.0, float(args.rectangle_trim_percentile)))
    if trim > 0 and len(xy) > 20:
        lo = np.percentile(xy, trim, axis=0)
        hi = np.percentile(xy, 100.0 - trim, axis=0)
        mask = np.all((xy >= lo) & (xy <= hi), axis=1)
        if int(np.sum(mask)) >= 4:
            xy = xy[mask]
    rect = cv2.minAreaRect(xy.reshape(-1, 1, 2))
    corners_xy = cv2.boxPoints(rect)
    z = float(np.percentile(points_ros[:, 2], 20.0))
    corners = [[float(x), float(y), z] for x, y in corners_xy]
    corners.append(corners[0])
    (cx, cy), (width, height), angle = rect
    return {
        "corners": corners,
        "area_m2": float(width * height),
        "width_m": float(width),
        "height_m": float(height),
        "center": [float(cx), float(cy), z],
        "angle_deg": float(angle),
        "z_cut_percentile": args.rectangle_z_percentile,
    }


def write_rectangle_csv(path: Path, rectangle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x", "y", "z"])
        for corner in rectangle.get("corners", []):
            writer.writerow([f"{corner[0]:.9f}", f"{corner[1]:.9f}", f"{corner[2]:.9f}"])


def write_preview(path: Path, points_ros: np.ndarray, rectangle: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if len(points_ros):
        step = max(1, len(points_ros) // 60000)
        sample = points_ros[::step]
        axes[0].scatter(sample[:, 0], sample[:, 1], s=0.25, alpha=0.22)
        axes[1].scatter(sample[:, 0], sample[:, 2], s=0.25, alpha=0.22)
    corners = np.array(rectangle.get("corners", []), dtype=np.float64)
    if len(corners):
        axes[0].plot(corners[:, 0], corners[:, 1], color="red", linewidth=2.0, label="min-area rectangle")
        axes[0].legend(loc="best")
    axes[0].set_title("Dense stereo map top view")
    axes[0].set_xlabel("ROS x forward [m]")
    axes[0].set_ylabel("ROS y left [m]")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Dense stereo map side view")
    axes[1].set_xlabel("ROS x forward [m]")
    axes[1].set_ylabel("ROS z up [m]")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    return path


def bounds_dict(points: np.ndarray) -> dict[str, Any]:
    if len(points) == 0:
        return {"min": [], "max": [], "range": []}
    return {
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
        "range": np.ptp(points, axis=0).tolist(),
    }


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


if __name__ == "__main__":
    main()
