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
    points_ros = load_points(args.input_csv, args.input_coordinate_frame)
    if len(points_ros) < 4:
        raise SystemExit(f"Need at least four points in {args.input_csv}")
    points_ros = rotate_yaw(points_ros, args.yaw_offset_deg)
    points_ros = percentile_filter(points_ros, args.trim_percentile)
    rectangle = fit_rectangle(points_ros, args)
    boundary_points, boundary_stats = select_boundary_points(points_ros, rectangle, args.boundary_band)
    if len(boundary_points) >= args.min_boundary_points:
        fit_points = boundary_points
    else:
        fit_points = points_ros
        boundary_stats["fallback"] = "boundary filter kept too few points; using trimmed points"
    projected = project_points_to_rectangle_boundary(fit_points, rectangle)
    z_min, z_max = robust_z_range(projected, args)
    if z_max <= z_min:
        z_min, z_max = -0.3, 0.3
    projected = projected[(projected[:, 2] >= z_min) & (projected[:, 2] <= z_max)]
    wall_grid = sample_rectangle_walls(rectangle, z_min, z_max, args.wall_spacing_xy, args.wall_spacing_z)
    output_points = np.vstack([projected, wall_grid]) if len(projected) else wall_grid
    output_points = voxel_downsample(output_points, args.voxel_size)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_cloud_csv(args.output_csv, output_points)
    write_rectangle_csv(args.rectangle_csv, rectangle, z_min)
    preview_png = write_preview(args.preview_png, points_ros, output_points, rectangle)
    summary = {
        "mode": "pool_wall_cloud",
        "inputs": {
            "input_csv": str(args.input_csv),
            "input_coordinate_frame": args.input_coordinate_frame,
            "dvl_usage": "not_used",
        },
        "parameters": {
            "yaw_offset_deg": args.yaw_offset_deg,
            "trim_percentile": args.trim_percentile,
            "boundary_band_m": args.boundary_band,
            "min_boundary_points": args.min_boundary_points,
            "wall_spacing_xy_m": args.wall_spacing_xy,
            "wall_spacing_z_m": args.wall_spacing_z,
            "voxel_size_m": args.voxel_size,
            "max_width_m": args.max_width,
            "max_height_m": args.max_height,
            "max_vertical_m": args.max_vertical,
        },
        "input_points": int(len(points_ros)),
        "boundary_filter": boundary_stats,
        "projected_points": int(len(projected)),
        "wall_grid_points": int(len(wall_grid)),
        "output_points": int(len(output_points)),
        "z_range_m": [float(z_min), float(z_max)],
        "rectangle": rectangle_to_json(rectangle, z_min),
        "outputs": {
            "wall_cloud_csv": str(args.output_csv),
            "rectangle_csv": str(args.rectangle_csv),
            "preview_png": str(preview_png),
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an RViz-ready wall cloud from vision-derived stereo map points. "
            "The output is in ROS/map coordinates and does not use DVL."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--input-coordinate-frame", choices=("ros", "opencv", "raw", "flip-y"), default="ros")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs/evaluation/current_pool_wall_cloud_ros.csv")
    parser.add_argument("--rectangle-csv", type=Path, default=PROJECT_ROOT / "outputs/evaluation/current_pool_wall_rectangle_ros.csv")
    parser.add_argument("--summary-json", type=Path, default=PROJECT_ROOT / "outputs/evaluation/current_pool_wall_cloud_summary.json")
    parser.add_argument("--preview-png", type=Path, default=PROJECT_ROOT / "outputs/evaluation/current_pool_wall_cloud_preview.png")
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--trim-percentile", type=float, default=2.0)
    parser.add_argument("--boundary-band", type=float, default=0.32)
    parser.add_argument("--min-boundary-points", type=int, default=500)
    parser.add_argument("--wall-spacing-xy", type=float, default=0.04)
    parser.add_argument("--wall-spacing-z", type=float, default=0.04)
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument("--z-low-percentile", type=float, default=5.0)
    parser.add_argument("--z-high-percentile", type=float, default=95.0)
    parser.add_argument("--max-width", type=float, default=5.49)
    parser.add_argument("--max-height", type=float, default=2.74)
    parser.add_argument("--max-vertical", type=float, default=1.32)
    return parser.parse_args()


def load_points(path: Path, coordinate_frame: str) -> np.ndarray:
    points: list[np.ndarray] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            raw = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
            points.append(convert_point(raw, coordinate_frame))
    return np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)


def convert_point(point: np.ndarray, coordinate_frame: str) -> np.ndarray:
    if coordinate_frame in {"opencv", "raw"}:
        return point.astype(np.float64)
    if coordinate_frame == "flip-y":
        return np.array([point[0], -point[1], point[2]], dtype=np.float64)
    if coordinate_frame != "ros":
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    return OPENCV_OPTICAL_TO_ROS @ point


def rotate_yaw(points: np.ndarray, yaw_offset_deg: float) -> np.ndarray:
    if abs(yaw_offset_deg) <= 1e-12:
        return points
    yaw = math.radians(yaw_offset_deg)
    rotation = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return points @ rotation.T


def percentile_filter(points: np.ndarray, trim_percentile: float) -> np.ndarray:
    trim = max(0.0, min(20.0, float(trim_percentile)))
    if trim <= 0.0 or len(points) < 20:
        return points
    low = np.percentile(points, trim, axis=0)
    high = np.percentile(points, 100.0 - trim, axis=0)
    keep = np.all((points >= low) & (points <= high), axis=1)
    return points[keep] if int(np.sum(keep)) >= 4 else points


def fit_rectangle(points_ros: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    xy = points_ros[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(xy.reshape(-1, 1, 2))
    center, size, angle = rect
    width, height = float(size[0]), float(size[1])
    if width <= 1e-6 or height <= 1e-6:
        raise SystemExit("Failed to fit a valid rectangle from vision point cloud")

    corners = cv2.boxPoints(rect).astype(np.float64)
    corners = clamp_rectangle_size(corners, float(args.max_width), float(args.max_height))
    center_xy = corners.mean(axis=0)
    edge_lengths = [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    ]
    return {
        "corners_xy": corners,
        "center_xy": center_xy,
        "edge_lengths": edge_lengths,
        "angle_deg": float(angle),
    }


def clamp_rectangle_size(corners: np.ndarray, max_width: float, max_height: float) -> np.ndarray:
    center = corners.mean(axis=0)
    edge0 = corners[1] - corners[0]
    edge1 = corners[2] - corners[1]
    len0 = float(np.linalg.norm(edge0))
    len1 = float(np.linalg.norm(edge1))
    if len0 <= 1e-9 or len1 <= 1e-9:
        return corners
    axis0 = edge0 / len0
    axis1 = edge1 / len1
    long_limit = max(max_width, max_height)
    short_limit = min(max_width, max_height)
    if len0 >= len1:
        half0 = min(len0, long_limit) * 0.5
        half1 = min(len1, short_limit) * 0.5
    else:
        half0 = min(len0, short_limit) * 0.5
        half1 = min(len1, long_limit) * 0.5
    return np.array(
        [
            center - axis0 * half0 - axis1 * half1,
            center + axis0 * half0 - axis1 * half1,
            center + axis0 * half0 + axis1 * half1,
            center - axis0 * half0 + axis1 * half1,
        ],
        dtype=np.float64,
    )


def select_boundary_points(
    points_ros: np.ndarray,
    rectangle: dict[str, Any],
    boundary_band: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    corners = rectangle["corners_xy"]
    distances = distances_to_rectangle_boundary(points_ros[:, :2], corners)
    keep = distances <= boundary_band
    return points_ros[keep], {
        "input_points": int(len(points_ros)),
        "kept_points": int(np.sum(keep)),
        "band_m": float(boundary_band),
    }


def distances_to_rectangle_boundary(points_xy: np.ndarray, corners_xy: np.ndarray) -> np.ndarray:
    distances = np.full(len(points_xy), np.inf, dtype=np.float64)
    for index in range(4):
        start = corners_xy[index]
        end = corners_xy[(index + 1) % 4]
        closest = closest_points_on_segment(points_xy, start, end)
        diff = points_xy - closest
        distances = np.minimum(distances, np.sqrt(np.sum(diff * diff, axis=1)))
    return distances


def closest_points_on_segment(points_xy: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return np.repeat(start.reshape(1, 2), len(points_xy), axis=0)
    t = np.clip(((points_xy - start) @ segment) / length_sq, 0.0, 1.0)
    return start + t[:, None] * segment


def project_points_to_rectangle_boundary(points_ros: np.ndarray, rectangle: dict[str, Any]) -> np.ndarray:
    corners = rectangle["corners_xy"]
    best_distance = np.full(len(points_ros), np.inf, dtype=np.float64)
    best_xy = np.zeros((len(points_ros), 2), dtype=np.float64)
    for index in range(4):
        closest = closest_points_on_segment(points_ros[:, :2], corners[index], corners[(index + 1) % 4])
        diff = points_ros[:, :2] - closest
        distances = np.sqrt(np.sum(diff * diff, axis=1))
        update = distances < best_distance
        best_distance[update] = distances[update]
        best_xy[update] = closest[update]
    return np.column_stack([best_xy, points_ros[:, 2]])


def robust_z_range(points_ros: np.ndarray, args: argparse.Namespace) -> tuple[float, float]:
    if len(points_ros) == 0:
        return -0.3, 0.3
    low = float(np.percentile(points_ros[:, 2], args.z_low_percentile))
    high = float(np.percentile(points_ros[:, 2], args.z_high_percentile))
    center = 0.5 * (low + high)
    half = min(0.5 * (high - low), 0.5 * float(args.max_vertical))
    half = max(half, min(0.35, 0.5 * float(args.max_vertical)))
    return center - half, center + half


def sample_rectangle_walls(
    rectangle: dict[str, Any],
    z_min: float,
    z_max: float,
    spacing_xy: float,
    spacing_z: float,
) -> np.ndarray:
    corners = rectangle["corners_xy"]
    spacing_xy = max(0.01, float(spacing_xy))
    spacing_z = max(0.01, float(spacing_z))
    z_values = np.arange(z_min, z_max + spacing_z * 0.5, spacing_z, dtype=np.float64)
    wall_points: list[np.ndarray] = []
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        length = float(np.linalg.norm(end - start))
        count = max(2, int(math.ceil(length / spacing_xy)) + 1)
        for t in np.linspace(0.0, 1.0, count, dtype=np.float64):
            xy = start + t * (end - start)
            wall_points.append(np.column_stack([
                np.full_like(z_values, xy[0]),
                np.full_like(z_values, xy[1]),
                z_values,
            ]))
    return np.vstack(wall_points) if wall_points else np.empty((0, 3), dtype=np.float64)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0.0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    buckets: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
    for key, point in zip(keys, points):
        key_tuple = tuple(int(value) for value in key)
        if key_tuple in buckets:
            total, count = buckets[key_tuple]
            buckets[key_tuple] = (total + point, count + 1)
        else:
            buckets[key_tuple] = (point.copy(), 1)
    return np.array([total / count for total, count in buckets.values()], dtype=np.float64)


def write_cloud_csv(path: Path, points: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "timestamp_sec", "track_id", "x", "y", "z", "u", "v", "age"])
        for index, point in enumerate(points):
            writer.writerow([0, "0.000000000", index, f"{point[0]:.9f}", f"{point[1]:.9f}", f"{point[2]:.9f}", "0.000", "0.000", 1])


def write_rectangle_csv(path: Path, rectangle: dict[str, Any], z_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x", "y", "z"])
        for xy in list(rectangle["corners_xy"]) + [rectangle["corners_xy"][0]]:
            writer.writerow([f"{xy[0]:.9f}", f"{xy[1]:.9f}", f"{z_value:.9f}"])


def rectangle_to_json(rectangle: dict[str, Any], z_value: float) -> dict[str, Any]:
    corners_xy = rectangle["corners_xy"]
    return {
        "corners": [[float(x), float(y), float(z_value)] for x, y in corners_xy] + [[float(corners_xy[0, 0]), float(corners_xy[0, 1]), float(z_value)]],
        "edge_lengths_m": [float(value) for value in rectangle["edge_lengths"]],
        "center": [float(rectangle["center_xy"][0]), float(rectangle["center_xy"][1]), float(z_value)],
        "angle_deg": float(rectangle["angle_deg"]),
    }


def write_preview(
    path: Path,
    input_points: np.ndarray,
    wall_points: np.ndarray,
    rectangle: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(input_points):
        ax.scatter(input_points[:, 0], input_points[:, 1], s=1, c="#2288ff", alpha=0.20, label="vision input")
    if len(wall_points):
        ax.scatter(wall_points[:, 0], wall_points[:, 1], s=1, c="#88ff88", alpha=0.55, label="wall cloud")
    corners = np.vstack([rectangle["corners_xy"], rectangle["corners_xy"][0]])
    ax.plot(corners[:, 0], corners[:, 1], c="#ff44dd", linewidth=2.0, label="fit")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_xlabel("ROS/map x [m]")
    ax.set_ylabel("ROS/map y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
