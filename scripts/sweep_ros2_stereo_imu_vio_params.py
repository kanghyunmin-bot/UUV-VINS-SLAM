#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    name: str
    params: dict[str, Any]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(args)
    dvl_rows = read_xyz(args.dvl_reference_csv)
    if len(dvl_rows) < 3:
        raise SystemExit(f"DVL reference is too short: {args.dvl_reference_csv}")

    results = []
    for index, candidate in enumerate(candidates, start=1):
        output_csv = args.output_dir / f"{index:03d}_{candidate.name}.csv"
        summary_json = args.output_dir / f"{index:03d}_{candidate.name}.json"
        debug_csv = args.output_dir / f"{index:03d}_{candidate.name}_debug.csv"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ros2_stereo_imu_vio.py"),
            "--dataset-dir",
            str(args.dataset_dir),
            "--max-frames",
            str(args.max_frames),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--debug-csv",
            str(debug_csv),
        ]
        for key, value in candidate.params.items():
            command.extend([f"--{key.replace('_', '-')}", str(value)])
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            results.append(
                {
                    "name": candidate.name,
                    "params": candidate.params,
                    "failed": True,
                    "error": "\n".join(completed.stdout.splitlines()[-20:]),
                }
            )
            continue

        vio_rows = read_xyz(output_csv)
        score = evaluate(vio_rows, dvl_rows)
        direct_score = evaluate_direct_rviz(vio_rows, dvl_rows)
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        result = {
            "name": candidate.name,
            "params": candidate.params,
            "failed": False,
            "output_csv": str(output_csv),
            "summary_json": str(summary_json),
            "debug_csv": str(debug_csv),
            "pose_success_ratio": summary.get("pose_success_ratio", 0.0),
            "path_length_m": summary["trajectory_stats"]["path_length_m"],
            "max_step_m": summary["trajectory_stats"]["max_step_m"],
            "mean_pnp_inliers": summary["tracking_stats"]["mean_pnp_inliers"],
            **score,
            **direct_score,
        }
        results.append(result)
        print(
            f"[{index:02d}/{len(candidates):02d}] {candidate.name} "
            f"direct_corr={result['direct_rviz_mean_corr']:.3f} direct_rmse={result['direct_rviz_rmse_m']:.3f} "
            f"aligned_corr={result['mean_corr']:.3f} aligned_rmse={result['rmse_m']:.3f} "
            f"path={result['path_length_m']:.2f} success={result['pose_success_ratio']:.3f}",
            flush=True,
        )

    valid_results = [row for row in results if not row.get("failed")]
    valid_results.sort(
        key=lambda row: (
            -float(row["direct_rviz_mean_corr"]),
            float(row["direct_rviz_rmse_m"]),
            -float(row["pose_success_ratio"]),
        )
    )
    best = valid_results[0] if valid_results else None
    report = {
        "mode": "ros2_stereo_imu_vio_parameter_sweep",
        "estimator_inputs": ["stereo_left", "stereo_right", "imu"],
        "reference_inputs_used": ["dvl_position_for_offline_scoring_only"],
        "dvl_usage": "offline_metric_only_not_fed_to_estimator",
        "dataset_dir": str(args.dataset_dir),
        "dvl_reference_csv": str(args.dvl_reference_csv),
        "candidate_count": len(candidates),
        "best": best,
        "results": valid_results + [row for row in results if row.get("failed")],
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_csv(args.report_csv, valid_results)
    print(json.dumps({"best": best, "report_json": str(args.report_json), "report_csv": str(args.report_csv)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune ROS2-native stereo+IMU VIO params against DVL offline metrics.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dvl-reference-csv", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "dvl_xyz_30s_reference.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ros2_stereo_imu_vio" / "sweeps" / "latest")
    parser.add_argument("--report-json", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "ros2_stereo_imu_vio_sweep_summary.json")
    parser.add_argument("--report-csv", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "ros2_stereo_imu_vio_sweep_summary.csv")
    parser.add_argument("--max-frames", type=int, default=316)
    parser.add_argument("--profile", choices=["quick", "wide"], default="quick")
    return parser.parse_args()


def build_candidates(args: argparse.Namespace) -> list[Candidate]:
    base = {
        "max_features": 220,
        "min_features": 100,
        "min_distance": 18,
        "pnp_threshold": 2.5,
        "min_pnp_inliers": 18,
        "max_step_m": 0.20,
        "max_speed_mps": 1.0,
        "max_rotation_rad": 0.45,
        "imu_rotation_gate_rad": 0.75,
        "translation_smoothing": 0.15,
    }
    candidates = [Candidate("baseline", base)]
    if args.profile == "quick":
        grid = {
            "max_features": [220, 320],
            "min_distance": [12, 18, 24],
            "pnp_threshold": [2.0, 2.5, 3.5],
            "max_step_m": [0.15, 0.20, 0.30],
            "translation_smoothing": [0.05, 0.15],
        }
        keys = list(grid.keys())
        for values in itertools.product(*(grid[key] for key in keys)):
            params = dict(base)
            params.update(dict(zip(keys, values)))
            params["min_features"] = max(80, int(params["max_features"] * 0.45))
            candidates.append(Candidate("_".join(f"{key}{value}" for key, value in zip(keys, values)), params))
    else:
        grid = {
            "max_features": [180, 220, 320, 450],
            "min_distance": [10, 14, 18, 24],
            "pnp_threshold": [1.8, 2.5, 3.5, 5.0],
            "min_pnp_inliers": [12, 18, 24],
            "max_step_m": [0.12, 0.20, 0.35],
            "translation_smoothing": [0.0, 0.10, 0.25],
        }
        keys = list(grid.keys())
        for values in itertools.product(*(grid[key] for key in keys)):
            params = dict(base)
            params.update(dict(zip(keys, values)))
            params["min_features"] = max(70, int(params["max_features"] * 0.42))
            candidates.append(Candidate("_".join(f"{key}{value}" for key, value in zip(keys, values)), params))
    # Preserve order but remove duplicates caused by baseline/grid overlap.
    unique: dict[str, Candidate] = {}
    for candidate in candidates:
        key = json.dumps(candidate.params, sort_keys=True)
        unique.setdefault(key, candidate)
    return list(unique.values())


def read_xyz(path: Path) -> list[tuple[float, np.ndarray]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append((float(row["timestamp_sec"]), np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)))
    return rows


def evaluate(vio_rows: list[tuple[float, np.ndarray]], dvl_rows: list[tuple[float, np.ndarray]]) -> dict[str, float]:
    if len(vio_rows) < 3:
        return empty_score()
    start = max(vio_rows[0][0], dvl_rows[0][0])
    end = min(vio_rows[-1][0], dvl_rows[-1][0])
    vio = [(t, p) for t, p in vio_rows if start <= t <= end]
    if len(vio) < 3:
        return empty_score()
    times = np.array([t for t, _ in vio], dtype=np.float64)
    est = np.array([p for _, p in vio], dtype=np.float64)
    ref = interpolate(dvl_rows, times)
    est_aligned, scale = align_sim3(est, ref)
    errors = np.linalg.norm(est_aligned - ref, axis=1)
    corr_values = [corrcoef(est_aligned[:, axis], ref[:, axis]) for axis in range(3)]
    return {
        "rows": float(len(times)),
        "rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "max_error_m": float(np.max(errors)),
        "mean_corr": float(np.mean(corr_values)),
        "corr_x": corr_values[0],
        "corr_y": corr_values[1],
        "corr_z": corr_values[2],
        "sim3_scale": float(scale),
        "dvl_path_length_m": path_length(ref),
        "aligned_path_length_m": path_length(est_aligned),
    }


def evaluate_direct_rviz(vio_rows: list[tuple[float, np.ndarray]], dvl_rows: list[tuple[float, np.ndarray]]) -> dict[str, float]:
    if len(vio_rows) < 3:
        return {
            "direct_rviz_rmse_m": float("inf"),
            "direct_rviz_max_error_m": float("inf"),
            "direct_rviz_mean_corr": 0.0,
            "direct_rviz_xy_corr": 0.0,
            "direct_rviz_corr_x": 0.0,
            "direct_rviz_corr_y": 0.0,
            "direct_rviz_corr_z": 0.0,
        }
    start = max(vio_rows[0][0], dvl_rows[0][0])
    end = min(vio_rows[-1][0], dvl_rows[-1][0])
    vio = [(t, p) for t, p in vio_rows if start <= t <= end]
    if len(vio) < 3:
        return {
            "direct_rviz_rmse_m": float("inf"),
            "direct_rviz_max_error_m": float("inf"),
            "direct_rviz_mean_corr": 0.0,
            "direct_rviz_xy_corr": 0.0,
            "direct_rviz_corr_x": 0.0,
            "direct_rviz_corr_y": 0.0,
            "direct_rviz_corr_z": 0.0,
        }
    times = np.array([t for t, _ in vio], dtype=np.float64)
    est = np.array([opencv_to_ros(p) for _, p in vio], dtype=np.float64)
    ref = np.array([dvl_to_rviz(p) for p in interpolate(dvl_rows, times)], dtype=np.float64)
    est = est - est[0]
    ref = ref - ref[0]
    errors = np.linalg.norm(est - ref, axis=1)
    corr_values = [corrcoef(est[:, axis], ref[:, axis]) for axis in range(3)]
    return {
        "direct_rviz_rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "direct_rviz_max_error_m": float(np.max(errors)),
        "direct_rviz_mean_corr": float(np.mean(corr_values)),
        "direct_rviz_xy_corr": float(np.mean(corr_values[:2])),
        "direct_rviz_corr_x": corr_values[0],
        "direct_rviz_corr_y": corr_values[1],
        "direct_rviz_corr_z": corr_values[2],
    }


def opencv_to_ros(point: np.ndarray) -> np.ndarray:
    return np.array([point[2], -point[0], -point[1]], dtype=np.float64)


def dvl_to_rviz(point: np.ndarray) -> np.ndarray:
    return np.array([point[0], -point[1], point[2]], dtype=np.float64)


def interpolate(rows: list[tuple[float, np.ndarray]], times: np.ndarray) -> np.ndarray:
    ref_times = np.array([t for t, _ in rows], dtype=np.float64)
    ref_points = np.array([p for _, p in rows], dtype=np.float64)
    out = np.column_stack(
        [np.interp(times, ref_times, ref_points[:, axis]) for axis in range(3)]
    )
    return out.astype(np.float64)


def align_sim3(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered / max(1, len(source))
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    rotated = source_centered @ rotation
    denom = float(np.sum(rotated * rotated))
    scale = float(np.sum(rotated * target_centered) / denom) if denom > 1e-12 else 1.0
    aligned = scale * rotated + target_mean
    return aligned, scale


def corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def empty_score() -> dict[str, float]:
    return {
        "rows": 0.0,
        "rmse_m": float("inf"),
        "max_error_m": float("inf"),
        "mean_corr": 0.0,
        "corr_x": 0.0,
        "corr_y": 0.0,
        "corr_z": 0.0,
        "sim3_scale": 1.0,
        "dvl_path_length_m": 0.0,
        "aligned_path_length_m": 0.0,
    }


def write_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "mean_corr",
        "direct_rviz_mean_corr",
        "direct_rviz_xy_corr",
        "direct_rviz_rmse_m",
        "rmse_m",
        "max_error_m",
        "corr_x",
        "corr_y",
        "corr_z",
        "path_length_m",
        "pose_success_ratio",
        "mean_pnp_inliers",
        "output_csv",
        "params",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fieldnames},
                    "params": json.dumps(row.get("params", {}), sort_keys=True),
                }
            )


if __name__ == "__main__":
    main()
