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
    odom_rows = read_rows(args.odom_csv)
    feature_rows = read_rows(args.feature_csv)
    odom_by_frame = {
        int(row["frame_index"]): row
        for row in odom_rows
        if row.get("frame_index") not in {None, ""}
    }
    if not odom_by_frame:
        raise SystemExit(f"No frame_index rows in {args.odom_csv}")
    origin = raw_position(odom_rows[0]) if args.zero_start else np.zeros(3, dtype=np.float64)
    path_display = np.array(
        [
            display_point(raw_position(row), origin, args.coordinate_frame, args.yaw_offset_deg)
            for row in odom_rows
            if row.get("pose_success") in {"1", "true", "True", ""}
        ],
        dtype=np.float64,
    )

    candidates: list[dict[str, Any]] = []
    rejection_counts = {
        "missing_odom": 0,
        "pose_failed": 0,
        "track_age": 0,
        "camera_depth_or_range": 0,
        "path_corridor": 0,
        "nonfinite": 0,
    }
    for row in feature_rows:
        try:
            frame_index = int(row["frame_index"])
            age = int(row.get("age", "1") or "1")
            point_raw = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
        except (KeyError, ValueError):
            rejection_counts["nonfinite"] += 1
            continue
        if not np.all(np.isfinite(point_raw)):
            rejection_counts["nonfinite"] += 1
            continue
        source_row = odom_by_frame.get(frame_index)
        if source_row is None:
            rejection_counts["missing_odom"] += 1
            continue
        if args.success_only and source_row.get("pose_success") not in {"1", "true", "True", ""}:
            rejection_counts["pose_failed"] += 1
            continue
        if age < args.min_track_age:
            rejection_counts["track_age"] += 1
            continue
        camera_point = point_in_source_camera(point_raw, source_row)
        if not passes_camera_filter(camera_point, args):
            rejection_counts["camera_depth_or_range"] += 1
            continue
        point_display = display_point(point_raw, origin, args.coordinate_frame, args.yaw_offset_deg)
        if args.path_corridor_radius > 0.0 and len(path_display):
            distance = min_path_distance_xy(point_display[:2], path_display[:, :2])
            if distance > args.path_corridor_radius:
                rejection_counts["path_corridor"] += 1
                continue
        candidates.append({"row": row, "point_display": point_display})

    if len(candidates) < 4:
        raise SystemExit("Too few VINS feature points survived basic filtering.")

    candidate_points = np.array([entry["point_display"] for entry in candidates], dtype=np.float64)
    trim_keep = percentile_mask(candidate_points, args.trim_percentile)
    trimmed_candidates = [entry for entry, keep in zip(candidates, trim_keep) if bool(keep)]
    trimmed_points = candidate_points[trim_keep]
    if len(trimmed_points) < 4:
        trimmed_candidates = candidates
        trimmed_points = candidate_points

    rectangle = fit_feature_rectangle(trimmed_points, args)
    z_keep = vertical_mask(trimmed_points, args)
    boundary_band_used: float | None = None
    if args.selection_mode == "boundary":
        distances = distances_to_rectangle_boundary(trimmed_points[:, :2], rectangle["corners_xy"])
        boundary_keep = distances <= args.boundary_band
        final_keep = boundary_keep & z_keep
        final_candidates = [entry for entry, keep in zip(trimmed_candidates, final_keep) if bool(keep)]
        if len(final_candidates) < args.min_output_points:
            relaxed_keep = (distances <= args.relaxed_boundary_band) & z_keep
            final_candidates = [entry for entry, keep in zip(trimmed_candidates, relaxed_keep) if bool(keep)]
            boundary_band_used = args.relaxed_boundary_band
        else:
            boundary_band_used = args.boundary_band
    else:
        final_candidates = [entry for entry, keep in zip(trimmed_candidates, z_keep) if bool(keep)]
    if len(final_candidates) < 4:
        raise SystemExit("Too few VINS feature points survived wall-boundary filtering.")

    final_points = np.array([entry["point_display"] for entry in final_candidates], dtype=np.float64)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_feature_rows(args.output_csv, [entry["row"] for entry in final_candidates])
    preview_png = write_preview(args.preview_png, candidate_points, final_points, rectangle, path_display)
    summary = {
        "mode": "vins_feature_cloud_filter",
        "inputs": {
            "odom_csv": str(args.odom_csv),
            "feature_csv": str(args.feature_csv),
            "dvl_usage": "not_used",
        },
        "parameters": {
            "coordinate_frame": args.coordinate_frame,
            "zero_start": args.zero_start,
            "yaw_offset_deg": args.yaw_offset_deg,
            "min_track_age": args.min_track_age,
            "min_depth_m": args.min_depth,
            "max_depth_m": args.max_depth,
            "max_range_m": args.max_range,
            "path_corridor_radius_m": args.path_corridor_radius,
            "trim_percentile": args.trim_percentile,
            "selection_mode": args.selection_mode,
            "boundary_band_m": boundary_band_used,
            "max_vertical_m": args.max_vertical,
        },
        "input_feature_rows": len(feature_rows),
        "basic_filter_candidates": len(candidates),
        "trimmed_candidates": len(trimmed_candidates),
        "output_feature_rows": len(final_candidates),
        "rejection_counts": rejection_counts,
        "display_bounds": bounds(final_points),
        "rectangle": rectangle_to_json(rectangle),
        "outputs": {
            "filtered_feature_csv": str(args.output_csv),
            "preview_png": str(preview_png),
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter VINS-generated feature landmarks for RViz wall-shaped point cloud display. DVL is not used."
    )
    parser.add_argument("--odom-csv", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs/evaluation/current_vins_feature_wall_cloud.csv")
    parser.add_argument("--summary-json", type=Path, default=PROJECT_ROOT / "outputs/evaluation/current_vins_feature_wall_cloud_summary.json")
    parser.add_argument("--preview-png", type=Path, default=PROJECT_ROOT / "outputs/evaluation/current_vins_feature_wall_cloud_preview.png")
    parser.add_argument("--coordinate-frame", choices=("ros", "opencv", "raw", "flip-y"), default="ros")
    parser.add_argument("--zero-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--min-track-age", type=int, default=3)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--max-range", type=float, default=8.0)
    parser.add_argument("--path-corridor-radius", type=float, default=4.5)
    parser.add_argument("--trim-percentile", type=float, default=2.0)
    parser.add_argument(
        "--selection-mode",
        choices=("boundary", "path-corridor"),
        default="boundary",
        help=(
            "boundary keeps points close to a fitted pool rectangle. path-corridor skips rectangle selection "
            "and keeps VINS feature rows that passed the camera/path gates."
        ),
    )
    parser.add_argument("--boundary-band", type=float, default=0.55)
    parser.add_argument("--relaxed-boundary-band", type=float, default=0.85)
    parser.add_argument("--min-output-points", type=int, default=1200)
    parser.add_argument("--max-width", type=float, default=5.49)
    parser.add_argument("--max-height", type=float, default=2.74)
    parser.add_argument("--max-vertical", type=float, default=1.32)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def raw_position(row: dict[str, str]) -> np.ndarray:
    return np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)


def display_point(point_raw: np.ndarray, origin_raw: np.ndarray, coordinate_frame: str, yaw_offset_deg: float) -> np.ndarray:
    point = point_raw - origin_raw
    if coordinate_frame in {"opencv", "raw"}:
        converted = point
    elif coordinate_frame == "flip-y":
        converted = np.array([point[0], -point[1], point[2]], dtype=np.float64)
    elif coordinate_frame == "ros":
        converted = OPENCV_OPTICAL_TO_ROS @ point
    else:
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    return rotate_yaw(converted, yaw_offset_deg)


def rotate_yaw(point: np.ndarray, yaw_offset_deg: float) -> np.ndarray:
    if abs(yaw_offset_deg) <= 1e-12:
        return point
    yaw = math.radians(yaw_offset_deg)
    rotation = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotation @ point


def point_in_source_camera(point_raw: np.ndarray, source_row: dict[str, str]) -> np.ndarray:
    source_position = raw_position(source_row)
    rotation = quaternion_to_matrix(
        np.array(
            [
                float(source_row.get("qx") or 0.0),
                float(source_row.get("qy") or 0.0),
                float(source_row.get("qz") or 0.0),
                float(source_row.get("qw") or 1.0),
            ],
            dtype=np.float64,
        )
    )
    return rotation.T @ (point_raw - source_position)


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


def passes_camera_filter(camera_point: np.ndarray, args: argparse.Namespace) -> bool:
    if camera_point[2] < args.min_depth:
        return False
    if args.max_depth > 0.0 and camera_point[2] > args.max_depth:
        return False
    if args.max_range > 0.0 and float(np.linalg.norm(camera_point)) > args.max_range:
        return False
    return True


def min_path_distance_xy(point_xy: np.ndarray, path_xy: np.ndarray) -> float:
    diff = path_xy - point_xy.reshape(1, 2)
    return float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))


def percentile_mask(points: np.ndarray, trim_percentile: float) -> np.ndarray:
    trim = max(0.0, min(20.0, float(trim_percentile)))
    if trim <= 0.0 or len(points) < 20:
        return np.ones(len(points), dtype=bool)
    low = np.percentile(points, trim, axis=0)
    high = np.percentile(points, 100.0 - trim, axis=0)
    return np.all((points >= low) & (points <= high), axis=1)


def vertical_mask(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    low = float(np.percentile(points[:, 2], 5.0))
    high = float(np.percentile(points[:, 2], 95.0))
    center = 0.5 * (low + high)
    half = min(0.5 * (high - low), 0.5 * args.max_vertical)
    half = max(half, 0.15)
    return np.abs(points[:, 2] - center) <= half


def fit_feature_rectangle(points: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    xy = points[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(xy.reshape(-1, 1, 2))
    corners = cv2.boxPoints(rect).astype(np.float64)
    corners = clamp_rectangle_size(corners, args.max_width, args.max_height)
    edge_lengths = [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    ]
    return {
        "corners_xy": corners,
        "edge_lengths": edge_lengths,
        "angle_deg": float(rect[2]),
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


def write_feature_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["frame_index", "timestamp_sec", "track_id", "x", "y", "z", "u", "v", "age"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def rectangle_to_json(rectangle: dict[str, Any]) -> dict[str, Any]:
    corners = rectangle["corners_xy"]
    return {
        "corners_xy": [[float(x), float(y)] for x, y in corners] + [[float(corners[0, 0]), float(corners[0, 1])]],
        "edge_lengths_m": [float(value) for value in rectangle["edge_lengths"]],
        "angle_deg": float(rectangle["angle_deg"]),
    }


def bounds(points: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": [float(value) for value in np.min(points, axis=0)],
        "max": [float(value) for value in np.max(points, axis=0)],
        "range": [float(value) for value in np.ptp(points, axis=0)],
    }


def write_preview(
    path: Path,
    candidates: np.ndarray,
    selected: np.ndarray,
    rectangle: dict[str, Any],
    path_display: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(candidates[:, 0], candidates[:, 1], s=2, c="#66aaff", alpha=0.18, label="VINS feature candidates")
    ax.scatter(selected[:, 0], selected[:, 1], s=4, c="#ffe650", alpha=0.75, label="selected VINS features")
    if len(path_display):
        ax.plot(path_display[:, 0], path_display[:, 1], c="#00aaff", linewidth=1.2, alpha=0.8, label="VINS odom path")
    corners = np.vstack([rectangle["corners_xy"], rectangle["corners_xy"][0]])
    ax.plot(corners[:, 0], corners[:, 1], c="#ff44dd", linewidth=2.0, label="feature boundary fit")
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
