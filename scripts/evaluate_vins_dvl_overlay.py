#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

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
    vins_rows = read_rows(args.vins_csv)
    dvl_rows = read_rows(args.dvl_csv)
    if args.success_only_vins:
        vins_rows = [row for row in vins_rows if row.get("pose_success") in {"1", "true", "True", ""}]
    if len(vins_rows) < 2 or len(dvl_rows) < 2:
        raise SystemExit("Need at least two rows in both VINS and DVL CSVs.")

    vins_times = np.array([float(row.get("timestamp_sec", index)) for index, row in enumerate(vins_rows)], dtype=np.float64)
    dvl_times = np.array([float(row.get("timestamp_sec", index)) for index, row in enumerate(dvl_rows)], dtype=np.float64)
    vins_positions = np.array([row_position(row, args.vins_coordinate_frame) for row in vins_rows], dtype=np.float64)
    dvl_positions = np.array([row_position(row, args.dvl_coordinate_frame) for row in dvl_rows], dtype=np.float64)

    if args.sample_reference_at_vins_times:
        times = vins_times.copy()
        vins_eval = vins_positions.copy()
        dvl_eval = interpolate(dvl_times, dvl_positions, np.clip(times, dvl_times[0], dvl_times[-1]))
    elif args.match_by == "index":
        count = min(len(vins_positions), len(dvl_positions))
        times = np.arange(count, dtype=np.float64)
        vins_eval = vins_positions[:count]
        dvl_eval = dvl_positions[:count]
    else:
        start = max(float(vins_times[0]), float(dvl_times[0]))
        end = min(float(vins_times[-1]), float(dvl_times[-1]))
        if end <= start:
            raise SystemExit("VINS and DVL time ranges do not overlap.")
        count = max(2, min(len(vins_rows), len(dvl_rows)))
        times = np.linspace(start, end, count)
        vins_eval = interpolate(vins_times, vins_positions, times)
        dvl_eval = interpolate(dvl_times, dvl_positions, times)

    if args.zero_start:
        vins_eval = vins_eval - vins_eval[0]
        dvl_eval = dvl_eval - dvl_eval[0]

    vins_eval = rotate_yaw(vins_eval, args.vins_yaw_offset_deg)
    dvl_eval = rotate_yaw(dvl_eval, args.dvl_yaw_offset_deg)

    metrics = trajectory_metrics(vins_eval, dvl_eval)
    summary: dict[str, Any] = {
        "vins_csv": str(args.vins_csv),
        "dvl_csv": str(args.dvl_csv),
        "match_by": args.match_by,
        "success_only_vins": args.success_only_vins,
        "sample_reference_at_vins_times": args.sample_reference_at_vins_times,
        "zero_start": args.zero_start,
        "vins_yaw_offset_deg": args.vins_yaw_offset_deg,
        "dvl_yaw_offset_deg": args.dvl_yaw_offset_deg,
        "sample_count": int(len(times)),
        "targets": {
            "max_axis_rmse_m": args.max_axis_rmse,
            "min_axis_corr": args.min_axis_corr,
        },
        "passed": bool(
            max(metrics["axis_rmse_m"].values()) <= args.max_axis_rmse
            and min(metrics["axis_corr"].values()) >= args.min_axis_corr
        ),
        **metrics,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VINS and DVL overlay agreement in the RViz/map frame.")
    parser.add_argument("--vins-csv", type=Path, required=True)
    parser.add_argument("--dvl-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "vins_dvl_overlay_strict_summary.json")
    parser.add_argument("--vins-coordinate-frame", choices=("ros", "raw", "opencv", "flip-y"), default="ros")
    parser.add_argument("--dvl-coordinate-frame", choices=("ros", "raw", "opencv", "flip-y"), default="ros")
    parser.add_argument("--match-by", choices=("time", "index"), default="time")
    parser.add_argument("--success-only-vins", action="store_true")
    parser.add_argument("--sample-reference-at-vins-times", action="store_true")
    parser.add_argument("--zero-start", action="store_true")
    parser.add_argument(
        "--vins-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Rotate the VINS trajectory around the evaluation/RViz Z axis after zero-start alignment.",
    )
    parser.add_argument(
        "--dvl-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Rotate the DVL reference trajectory around the evaluation/RViz Z axis. Keep at 0 for fixed-reference scoring.",
    )
    parser.add_argument("--max-axis-rmse", type=float, default=1e-5)
    parser.add_argument("--min-axis-corr", type=float, default=0.9999)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def row_position(row: dict[str, str], coordinate_frame: str) -> tuple[float, float, float]:
    point = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
    if coordinate_frame in {"raw", "opencv"}:
        return tuple(point.tolist())
    if coordinate_frame == "flip-y":
        return (float(point[0]), float(-point[1]), float(point[2]))
    if coordinate_frame != "ros":
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    return tuple((OPENCV_OPTICAL_TO_ROS @ point).tolist())


def interpolate(times: np.ndarray, positions: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    out = np.empty((len(query_times), 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(query_times, times, positions[:, axis])
    return out


def rotate_yaw(positions: np.ndarray, yaw_offset_deg: float) -> np.ndarray:
    if abs(yaw_offset_deg) <= 1e-12:
        return positions
    yaw = math.radians(yaw_offset_deg)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return positions @ rotation.T


def trajectory_metrics(vins: np.ndarray, dvl: np.ndarray) -> dict[str, Any]:
    errors = vins - dvl
    axis_rmse = np.sqrt(np.mean(errors * errors, axis=0))
    axis_max_abs = np.max(np.abs(errors), axis=0)
    axis_corr = np.array([pearson(vins[:, axis], dvl[:, axis]) for axis in range(3)], dtype=np.float64)
    point_errors = np.linalg.norm(errors, axis=1)
    dvl_length = path_length(dvl)
    rmse_3d = float(math.sqrt(float(np.mean(point_errors * point_errors))))
    rmse_ratio = rmse_3d / max(dvl_length, 1e-12)
    direction = direction_metrics(vins, dvl)
    return {
        "axis_rmse_m": {
            "x": float(axis_rmse[0]),
            "y": float(axis_rmse[1]),
            "z": float(axis_rmse[2]),
        },
        "axis_max_abs_error_m": {
            "x": float(axis_max_abs[0]),
            "y": float(axis_max_abs[1]),
            "z": float(axis_max_abs[2]),
        },
        "axis_corr": {
            "x": float(axis_corr[0]),
            "y": float(axis_corr[1]),
            "z": float(axis_corr[2]),
        },
        "mean_axis_corr": float(np.mean(axis_corr)),
        "rmse_3d_m": rmse_3d,
        "max_3d_error_m": float(np.max(point_errors)),
        "dvl_path_length_m": float(dvl_length),
        "rmse_ratio": float(rmse_ratio),
        "rmse_match_percent": float(max(0.0, 1.0 - rmse_ratio) * 100.0),
        **direction,
    }


def direction_metrics(vins: np.ndarray, dvl: np.ndarray) -> dict[str, Any]:
    vins_steps = np.diff(vins, axis=0)
    dvl_steps = np.diff(dvl, axis=0)
    vins_norms = np.linalg.norm(vins_steps, axis=1)
    dvl_norms = np.linalg.norm(dvl_steps, axis=1)
    valid = (vins_norms > 1e-9) & (dvl_norms > 1e-9)
    if not np.any(valid):
        return {
            "direction_sample_count": 0,
            "mean_direction_cosine": 0.0,
            "mean_direction_error_deg": 180.0,
            "median_direction_error_deg": 180.0,
            "p95_direction_error_deg": 180.0,
        }
    vins_unit = vins_steps[valid] / vins_norms[valid, None]
    dvl_unit = dvl_steps[valid] / dvl_norms[valid, None]
    cosines = np.sum(vins_unit * dvl_unit, axis=1)
    cosines = np.clip(cosines, -1.0, 1.0)
    errors_deg = np.degrees(np.arccos(cosines))
    return {
        "direction_sample_count": int(len(errors_deg)),
        "mean_direction_cosine": float(np.mean(cosines)),
        "mean_direction_error_deg": float(np.mean(errors_deg)),
        "median_direction_error_deg": float(np.median(errors_deg)),
        "p95_direction_error_deg": float(np.percentile(errors_deg, 95)),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm <= 1e-12 and right_norm <= 1e-12:
        return 1.0 if np.allclose(left, right, atol=1e-12, rtol=0.0) else 0.0
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / (left_norm * right_norm))


def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


if __name__ == "__main__":
    main()
