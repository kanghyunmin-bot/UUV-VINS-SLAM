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


def main() -> None:
    args = parse_args()
    report = json.loads(args.sweep_json.read_text(encoding="utf-8"))
    dvl_rows = read_xyz(args.dvl_reference_csv)
    offsets = parse_float_grid(args.offset_grid)
    rescored = []
    for item in report.get("results", []):
        if item.get("failed"):
            continue
        output_csv = Path(item["output_csv"])
        if not output_csv.is_absolute():
            output_csv = PROJECT_ROOT / output_csv
        if not output_csv.exists():
            continue
        vio_rows = read_xyz(output_csv)
        best = None
        for offset in offsets:
            score = direct_rviz_score(vio_rows, dvl_rows, offset)
            if best is None or score_key(score) < score_key(best):
                best = score
        if best is None:
            continue
        rescored.append({**item, **best})

    rescored.sort(key=score_key)
    output = {
        "mode": "rescore_ros2_stereo_imu_vio_sweep_direct_rviz_with_time_offset",
        "estimator_inputs": ["stereo_left", "stereo_right", "imu"],
        "reference_inputs_used": ["dvl_position_for_offline_scoring_only"],
        "dvl_usage": "offline_metric_only_not_fed_to_estimator",
        "sweep_json": str(args.sweep_json),
        "dvl_reference_csv": str(args.dvl_reference_csv),
        "offset_grid_sec": offsets,
        "best": rescored[0] if rescored else None,
        "results": rescored,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rescored)
    print(json.dumps({"best": output["best"], "output_json": str(args.output_json), "output_csv": str(args.output_csv)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore stored ROS2 VIO sweep outputs with direct RViz metrics and DVL time offset.")
    parser.add_argument("--sweep-json", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "ros2_stereo_imu_vio_sweep_summary.json")
    parser.add_argument("--dvl-reference-csv", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "dvl_xyz_30s_reference.csv")
    parser.add_argument("--offset-grid", default="-5,-4,-3,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,3,4,5")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "ros2_stereo_imu_vio_sweep_direct_offset_summary.json")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "ros2_stereo_imu_vio_sweep_direct_offset_summary.csv")
    return parser.parse_args()


def score_key(row: dict[str, Any]) -> tuple[float, float, float]:
    # Higher direct visible correlation is the main target; RMSE and path error break ties.
    return (
        -float(row["direct_rviz_mean_corr"]),
        float(row["direct_rviz_rmse_m"]),
        abs(float(row["direct_blue_len_m"]) - float(row["direct_red_len_m"])),
    )


def read_xyz(path: Path) -> list[tuple[float, np.ndarray]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append(
                (
                    float(row["timestamp_sec"]),
                    np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64),
                )
            )
    return rows


def parse_float_grid(text: str) -> list[float]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values or [0.0]


def direct_rviz_score(
    vio_rows: list[tuple[float, np.ndarray]],
    dvl_rows: list[tuple[float, np.ndarray]],
    dvl_time_offset_sec: float,
) -> dict[str, float]:
    if len(vio_rows) < 3 or len(dvl_rows) < 3:
        return empty_score(dvl_time_offset_sec)
    dvl_start = dvl_rows[0][0] - dvl_time_offset_sec
    dvl_end = dvl_rows[-1][0] - dvl_time_offset_sec
    start = max(vio_rows[0][0], dvl_start)
    end = min(vio_rows[-1][0], dvl_end)
    vio = [(time, point) for time, point in vio_rows if start <= time <= end]
    if len(vio) < 3:
        return empty_score(dvl_time_offset_sec)
    times = np.array([time for time, _ in vio], dtype=np.float64)
    blue = np.array([opencv_to_ros(point) for _, point in vio], dtype=np.float64)
    # Positive offset means the DVL reference is sampled later than the VIO timestamp.
    red = np.array(
        [dvl_to_rviz(point) for point in interpolate(dvl_rows, times + dvl_time_offset_sec)],
        dtype=np.float64,
    )
    blue = blue - blue[0]
    red = red - red[0]
    errors = np.linalg.norm(blue - red, axis=1)
    xy_errors = np.linalg.norm(blue[:, :2] - red[:, :2], axis=1)
    corr_values = [corrcoef(blue[:, axis], red[:, axis]) for axis in range(3)]
    return {
        "direct_dvl_time_offset_sec": float(dvl_time_offset_sec),
        "direct_rows": float(len(times)),
        "direct_rviz_rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "direct_rviz_xy_rmse_m": float(np.sqrt(np.mean(xy_errors * xy_errors))),
        "direct_rviz_max_error_m": float(np.max(errors)),
        "direct_rviz_mean_corr": float(np.mean(corr_values)),
        "direct_rviz_xy_corr": float(np.mean(corr_values[:2])),
        "direct_rviz_corr_x": corr_values[0],
        "direct_rviz_corr_y": corr_values[1],
        "direct_rviz_corr_z": corr_values[2],
        "direct_blue_len_m": path_length(blue),
        "direct_red_len_m": path_length(red),
    }


def empty_score(offset: float) -> dict[str, float]:
    return {
        "direct_dvl_time_offset_sec": float(offset),
        "direct_rows": 0.0,
        "direct_rviz_rmse_m": float("inf"),
        "direct_rviz_xy_rmse_m": float("inf"),
        "direct_rviz_max_error_m": float("inf"),
        "direct_rviz_mean_corr": 0.0,
        "direct_rviz_xy_corr": 0.0,
        "direct_rviz_corr_x": 0.0,
        "direct_rviz_corr_y": 0.0,
        "direct_rviz_corr_z": 0.0,
        "direct_blue_len_m": 0.0,
        "direct_red_len_m": 0.0,
    }


def interpolate(rows: list[tuple[float, np.ndarray]], times: np.ndarray) -> np.ndarray:
    source_times = np.array([time for time, _ in rows], dtype=np.float64)
    source_points = np.array([point for _, point in rows], dtype=np.float64)
    return np.column_stack([np.interp(times, source_times, source_points[:, axis]) for axis in range(3)])


def opencv_to_ros(point: np.ndarray) -> np.ndarray:
    return np.array([point[2], -point[0], -point[1]], dtype=np.float64)


def dvl_to_rviz(point: np.ndarray) -> np.ndarray:
    return np.array([point[0], -point[1], point[2]], dtype=np.float64)


def corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "direct_dvl_time_offset_sec",
        "direct_rviz_mean_corr",
        "direct_rviz_xy_corr",
        "direct_rviz_rmse_m",
        "direct_rviz_xy_rmse_m",
        "direct_blue_len_m",
        "direct_red_len_m",
        "path_length_m",
        "pose_success_ratio",
        "output_csv",
        "params",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fields},
                    "params": json.dumps(row.get("params", {}), sort_keys=True),
                }
            )


if __name__ == "__main__":
    main()
