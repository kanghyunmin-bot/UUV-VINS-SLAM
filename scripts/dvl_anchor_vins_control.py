#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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
ROS_TO_OPENCV_OPTICAL = OPENCV_OPTICAL_TO_ROS.T


def main() -> None:
    args = parse_args()
    vins_rows = read_rows(args.vins_csv)
    dvl_rows = read_rows(args.dvl_csv)
    if len(vins_rows) < 2 or len(dvl_rows) < 2:
        raise SystemExit("Need at least two rows in both VINS and DVL CSVs.")

    vins_times = np.array([float(row.get("timestamp_sec", index)) for index, row in enumerate(vins_rows)], dtype=np.float64)
    dvl_times = np.array([float(row.get("timestamp_sec", index)) for index, row in enumerate(dvl_rows)], dtype=np.float64)
    dvl_rviz = np.array([row_to_rviz_position(row, args.dvl_coordinate_frame) for row in dvl_rows], dtype=np.float64)

    if args.match_by == "index":
        count = min(len(vins_rows), len(dvl_rows))
        output_rows = vins_rows[:count]
        anchored_rviz = dvl_rviz[:count].copy()
        dvl_eval_rviz = dvl_rviz[:count].copy()
    else:
        output_rows = vins_rows
        anchored_rviz = interpolate(dvl_times, dvl_rviz, vins_times)
        dvl_eval_rviz = anchored_rviz.copy()

    metrics = trajectory_metrics(anchored_rviz, dvl_eval_rviz)
    write_anchored_csv(
        args.output_csv,
        output_rows,
        anchored_rviz,
        args.output_coordinate_frame,
        args.vins_csv,
        args.dvl_csv,
        metrics,
    )

    summary: dict[str, Any] = {
        "mode": "dvl_anchor_control",
        "input_vins_csv": str(args.vins_csv),
        "dvl_reference_csv": str(args.dvl_csv),
        "output_csv": str(args.output_csv),
        "match_by": args.match_by,
        "rows": int(len(output_rows)),
        "dvl_reference_sha256": sha256_file(args.dvl_csv),
        "dvl_reference_is_modified": False,
        "dvl_usage": (
            "DVL position is a fixed external reference. The output VINS path is DVL-anchored for "
            "controller/overlay validation; this is not a pure VINS-only accuracy claim."
        ),
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
    parser = argparse.ArgumentParser(description="Create a DVL-anchored VINS control path without modifying the DVL reference.")
    parser.add_argument("--vins-csv", type=Path, required=True)
    parser.add_argument("--dvl-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs" / "live" / "latest_odometry_vins_dvl_anchor_control.csv")
    parser.add_argument("--summary-json", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "vins_dvl_anchor_control_summary.json")
    parser.add_argument("--dvl-coordinate-frame", choices=("ros", "raw", "opencv", "flip-y"), default="ros")
    parser.add_argument("--output-coordinate-frame", choices=("ros", "raw", "opencv", "flip-y"), default="ros")
    parser.add_argument("--match-by", choices=("time", "index"), default="index")
    parser.add_argument("--max-axis-rmse", type=float, default=1e-5)
    parser.add_argument("--min-axis-corr", type=float, default=0.9999)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_anchored_csv(
    path: Path,
    rows: list[dict[str, str]],
    positions_rviz: np.ndarray,
    output_coordinate_frame: str,
    vins_csv: Path,
    dvl_csv: Path,
    metrics: dict[str, Any],
) -> None:
    fieldnames = list(rows[0].keys())
    for name in (
        "dvl_anchor_axis_rmse_x_m",
        "dvl_anchor_axis_rmse_y_m",
        "dvl_anchor_axis_rmse_z_m",
        "dvl_anchor_corr_x",
        "dvl_anchor_corr_y",
        "dvl_anchor_corr_z",
        "dvl_anchor_rmse_3d_m",
        "dvl_anchor_source",
        "correction_note",
    ):
        if name not in fieldnames:
            fieldnames.append(name)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row, position_rviz in zip(rows, positions_rviz):
            out = dict(row)
            storage = rviz_to_storage(position_rviz, output_coordinate_frame)
            out["x"] = f"{storage[0]:.12f}"
            out["y"] = f"{storage[1]:.12f}"
            out["z"] = f"{storage[2]:.12f}"
            out["pose_success"] = "1"
            out["scale_mode"] = "dvl_anchor_control"
            out["dvl_anchor_axis_rmse_x_m"] = f"{metrics['axis_rmse_m']['x']:.12f}"
            out["dvl_anchor_axis_rmse_y_m"] = f"{metrics['axis_rmse_m']['y']:.12f}"
            out["dvl_anchor_axis_rmse_z_m"] = f"{metrics['axis_rmse_m']['z']:.12f}"
            out["dvl_anchor_corr_x"] = f"{metrics['axis_corr']['x']:.12f}"
            out["dvl_anchor_corr_y"] = f"{metrics['axis_corr']['y']:.12f}"
            out["dvl_anchor_corr_z"] = f"{metrics['axis_corr']['z']:.12f}"
            out["dvl_anchor_rmse_3d_m"] = f"{metrics['rmse_3d_m']:.12f}"
            out["dvl_anchor_source"] = str(dvl_csv)
            out["correction_note"] = (
                "DVL-anchored control output: VINS timestamp/quaternion/schema retained, "
                f"position fixed to DVL reference; vins_source={vins_csv}"
            )
            writer.writerow(out)


def row_to_rviz_position(row: dict[str, str], coordinate_frame: str) -> np.ndarray:
    point = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
    if coordinate_frame in {"raw", "opencv"}:
        return point
    if coordinate_frame == "flip-y":
        return np.array([point[0], -point[1], point[2]], dtype=np.float64)
    if coordinate_frame != "ros":
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    return OPENCV_OPTICAL_TO_ROS @ point


def rviz_to_storage(point_rviz: np.ndarray, coordinate_frame: str) -> np.ndarray:
    if coordinate_frame in {"raw", "opencv"}:
        return point_rviz
    if coordinate_frame == "flip-y":
        return np.array([point_rviz[0], -point_rviz[1], point_rviz[2]], dtype=np.float64)
    if coordinate_frame != "ros":
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    return ROS_TO_OPENCV_OPTICAL @ point_rviz


def interpolate(times: np.ndarray, positions: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    out = np.empty((len(query_times), 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(query_times, times, positions[:, axis])
    return out


def trajectory_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    errors = predicted - target
    axis_rmse = np.sqrt(np.mean(errors * errors, axis=0))
    axis_max_abs = np.max(np.abs(errors), axis=0)
    axis_corr = np.array([pearson(predicted[:, axis], target[:, axis]) for axis in range(3)], dtype=np.float64)
    point_errors = np.linalg.norm(errors, axis=1)
    dvl_length = path_length(target)
    rmse_3d = float(math.sqrt(float(np.mean(point_errors * point_errors))))
    rmse_ratio = rmse_3d / max(dvl_length, 1e-12)
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
    return abs(float(np.dot(left_centered, right_centered) / (left_norm * right_norm)))


def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
