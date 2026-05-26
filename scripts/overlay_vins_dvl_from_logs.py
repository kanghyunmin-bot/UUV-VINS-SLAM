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
    vins = read_vins_log(args.vins_log)
    dvl = read_dvl_csv(args.dvl_csv, flip_y=args.dvl_coordinate_frame == "flip-y")
    if args.max_vins_duration_sec > 0.0 and len(vins) > 0:
        relative_t = vins[:, 0] - vins[0, 0]
        vins = vins[relative_t <= args.max_vins_duration_sec + 1e-9]
    if len(vins) < 2 or len(dvl) < 2:
        raise SystemExit("Need at least two VINS and DVL samples.")

    yaw_offset_deg = args.vins_yaw_offset_deg
    yaw_sweep: dict[str, Any] | None = None
    if args.auto_vins_yaw_sweep:
        yaw_offset_deg, yaw_sweep = find_best_yaw_offset(
            vins,
            dvl,
            min(args.shape_samples, len(vins), len(dvl)),
            args.yaw_sweep_step_deg,
            args.yaw_sweep_objective,
        )

    time_overlay = time_align(vins, dvl, yaw_offset_deg)
    shape_overlay = arclength_align(vins, dvl, min(args.shape_samples, len(vins), len(dvl)), yaw_offset_deg)

    args.overlay_csv.parent.mkdir(parents=True, exist_ok=True)
    write_overlay_csv(args.overlay_csv, time_overlay)

    summary = {
        "mode": "vins_log_to_dvl_reference_overlay",
        "vins_log": str(args.vins_log),
        "dvl_csv": str(args.dvl_csv),
        "dvl_usage": "reference_only_not_estimator_input",
        "coordinate_note": "VINS raw world frame zero-start; optional fixed yaw frame correction; DVL fixed reference with optional flip-y zero-start.",
        "max_vins_duration_sec": float(args.max_vins_duration_sec),
        "vins_yaw_offset_deg": float(yaw_offset_deg),
        "auto_vins_yaw_sweep": bool(args.auto_vins_yaw_sweep),
        "yaw_sweep": yaw_sweep,
        "time_aligned": metrics(time_overlay),
        "shape_arclength_aligned": metrics(shape_overlay),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_plot and args.plot_png:
        write_plot(args.plot_png, time_overlay, shape_overlay, summary)
    if not args.no_plot and args.single_xy_plot_png:
        write_single_xy_plot(args.single_xy_plot_png, shape_overlay, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay current VINS-Fusion pose log with fixed DVL reference.")
    parser.add_argument(
        "--vins-log",
        type=Path,
        default=PROJECT_ROOT / "outputs/ros2_vins_fusion_live/vio.csv",
    )
    parser.add_argument(
        "--dvl-csv",
        type=Path,
        default=PROJECT_ROOT / "data/rosbag_active/localization bag/dvl_reference_34_85s.csv",
    )
    parser.add_argument(
        "--overlay-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/evaluation/current_vins_dvl_log_overlay.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs/evaluation/current_vins_dvl_log_overlay_summary.json",
    )
    parser.add_argument(
        "--plot-png",
        type=Path,
        default=PROJECT_ROOT / "outputs/evaluation/current_vins_dvl_log_overlay.png",
    )
    parser.add_argument("--dvl-coordinate-frame", choices=("raw", "flip-y"), default="flip-y")
    parser.add_argument("--shape-samples", type=int, default=400)
    parser.add_argument(
        "--max-vins-duration-sec",
        type=float,
        default=0.0,
        help="Crop VINS log to this duration from its first pose before scoring. 0 disables cropping.",
    )
    parser.add_argument(
        "--vins-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Fixed RViz/world yaw correction applied to VINS after zero-start. DVL remains reference only.",
    )
    parser.add_argument(
        "--auto-vins-yaw-sweep",
        action="store_true",
        help="Find a fixed VINS yaw correction from the logs for offline overlay diagnostics only.",
    )
    parser.add_argument("--yaw-sweep-step-deg", type=float, default=1.0)
    parser.add_argument("--yaw-sweep-objective", choices=("time", "shape"), default="shape")
    parser.add_argument(
        "--single-xy-plot-png",
        type=Path,
        help="Optional single XY overlay plot for presentation-style comparison.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Write CSV and JSON metrics only. Useful on hosts without matplotlib.",
    )
    return parser.parse_args()


def read_vins_log(path: Path) -> np.ndarray:
    rows: list[tuple[float, float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) < 4:
                continue
            try:
                timestamp = float(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
            except ValueError:
                continue
            rows.append((timestamp, x, y, z))
    return np.asarray(rows, dtype=np.float64)


def read_dvl_csv(path: Path, flip_y: bool) -> np.ndarray:
    rows: list[tuple[float, float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                timestamp = float(row["timestamp_sec"])
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
            except (KeyError, ValueError):
                continue
            rows.append((timestamp, x, -y if flip_y else y, z))
    return np.asarray(rows, dtype=np.float64)


def time_align(vins: np.ndarray, dvl: np.ndarray, vins_yaw_offset_deg: float = 0.0) -> dict[str, np.ndarray]:
    vins_t = vins[:, 0] - vins[0, 0]
    dvl_t = dvl[:, 0] - dvl[0, 0]
    query_t = np.clip(vins_t, dvl_t[0], dvl_t[-1])
    vins_pos = rotate_yaw(zero_start(vins[:, 1:4]), vins_yaw_offset_deg)
    dvl_pos = interpolate(dvl_t, zero_start(dvl[:, 1:4]), query_t)
    return {"t": vins_t, "vins": vins_pos, "dvl": dvl_pos}


def arclength_align(vins: np.ndarray, dvl: np.ndarray, samples: int, vins_yaw_offset_deg: float = 0.0) -> dict[str, np.ndarray]:
    vins_pos = rotate_yaw(resample_by_arclength(zero_start(vins[:, 1:4]), samples), vins_yaw_offset_deg)
    dvl_pos = resample_by_arclength(zero_start(dvl[:, 1:4]), samples)
    return {"t": np.arange(samples, dtype=np.float64), "vins": vins_pos, "dvl": dvl_pos}


def find_best_yaw_offset(
    vins: np.ndarray,
    dvl: np.ndarray,
    shape_samples: int,
    step_deg: float,
    objective: str,
) -> tuple[float, dict[str, Any]]:
    if step_deg <= 0:
        raise ValueError("yaw sweep step must be positive")
    best_yaw = 0.0
    best_metrics: dict[str, Any] | None = None
    best_score = -math.inf
    yaw = -180.0
    tested = 0
    while yaw <= 180.0 + 1e-9:
        overlay = time_align(vins, dvl, yaw) if objective == "time" else arclength_align(vins, dvl, shape_samples, yaw)
        result = metrics(overlay)
        score = result["rmse_match_percent"]
        if score > best_score:
            best_score = score
            best_yaw = yaw
            best_metrics = result
        yaw += step_deg
        tested += 1
    return best_yaw, {
        "objective": objective,
        "step_deg": float(step_deg),
        "tested_count": tested,
        "best_yaw_offset_deg": float(best_yaw),
        "best_metrics": best_metrics,
        "usage": "offline_frame_diagnostic_only_not_estimator_input",
    }


def interpolate(times: np.ndarray, positions: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    out = np.empty((len(query_times), 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(query_times, times, positions[:, axis])
    return out


def zero_start(positions: np.ndarray) -> np.ndarray:
    return positions - positions[0]


def rotate_yaw(positions: np.ndarray, yaw_offset_deg: float) -> np.ndarray:
    if abs(yaw_offset_deg) <= 1e-12:
        return positions
    yaw = math.radians(yaw_offset_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return positions @ rotation.T


def resample_by_arclength(positions: np.ndarray, samples: int) -> np.ndarray:
    if len(positions) < 2:
        return np.repeat(positions[:1], samples, axis=0)
    dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    total = float(dist[-1])
    if total <= 1e-12:
        return np.repeat(positions[:1], samples, axis=0)
    query = np.linspace(0.0, total, samples)
    out = np.empty((samples, 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(query, dist, positions[:, axis])
    return out


def metrics(overlay: dict[str, np.ndarray]) -> dict[str, Any]:
    vins = overlay["vins"]
    dvl = overlay["dvl"]
    errors = vins - dvl
    point_errors = np.linalg.norm(errors, axis=1)
    axis_rmse = np.sqrt(np.mean(errors * errors, axis=0))
    axis_corr = [pearson(vins[:, axis], dvl[:, axis]) for axis in range(3)]
    dvl_length = path_length(dvl)
    vins_length = path_length(vins)
    rmse3 = float(math.sqrt(float(np.mean(point_errors * point_errors))))
    return {
        "sample_count": int(len(vins)),
        "vins_path_length_m": float(vins_length),
        "dvl_path_length_m": float(dvl_length),
        "rmse_3d_m": rmse3,
        "rmse_match_percent": float(max(0.0, 1.0 - rmse3 / max(dvl_length, 1e-12)) * 100.0),
        "max_3d_error_m": float(np.max(point_errors)),
        "axis_rmse_m": {"x": float(axis_rmse[0]), "y": float(axis_rmse[1]), "z": float(axis_rmse[2])},
        "axis_corr": {"x": float(axis_corr[0]), "y": float(axis_corr[1]), "z": float(axis_corr[2])},
        **direction_metrics(vins, dvl),
    }


def direction_metrics(vins: np.ndarray, dvl: np.ndarray) -> dict[str, Any]:
    vins_steps = np.diff(vins, axis=0)
    dvl_steps = np.diff(dvl, axis=0)
    vins_norm = np.linalg.norm(vins_steps, axis=1)
    dvl_norm = np.linalg.norm(dvl_steps, axis=1)
    valid = (vins_norm > 1e-9) & (dvl_norm > 1e-9)
    if not np.any(valid):
        return {"direction_sample_count": 0, "mean_direction_error_deg": 180.0}
    cosines = np.sum(vins_steps[valid] * dvl_steps[valid], axis=1) / (vins_norm[valid] * dvl_norm[valid])
    errors = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return {
        "direction_sample_count": int(len(errors)),
        "mean_direction_cosine": float(np.mean(cosines)),
        "mean_direction_error_deg": float(np.mean(errors)),
        "median_direction_error_deg": float(np.median(errors)),
        "p95_direction_error_deg": float(np.percentile(errors, 95)),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def write_overlay_csv(path: Path, overlay: dict[str, np.ndarray]) -> None:
    fields = [
        "t",
        "vins_x",
        "vins_y",
        "vins_z",
        "dvl_x",
        "dvl_y",
        "dvl_z",
        "err_x",
        "err_y",
        "err_z",
        "err_3d",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for t, vins, dvl in zip(overlay["t"], overlay["vins"], overlay["dvl"]):
            err = vins - dvl
            writer.writerow(
                {
                    "t": f"{float(t):.9f}",
                    "vins_x": f"{float(vins[0]):.9f}",
                    "vins_y": f"{float(vins[1]):.9f}",
                    "vins_z": f"{float(vins[2]):.9f}",
                    "dvl_x": f"{float(dvl[0]):.9f}",
                    "dvl_y": f"{float(dvl[1]):.9f}",
                    "dvl_z": f"{float(dvl[2]):.9f}",
                    "err_x": f"{float(err[0]):.9f}",
                    "err_y": f"{float(err[1]):.9f}",
                    "err_z": f"{float(err[2]):.9f}",
                    "err_3d": f"{float(np.linalg.norm(err)):.9f}",
                }
            )


def write_plot(path: Path, time_overlay: dict[str, np.ndarray], shape_overlay: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("VINS log vs fixed DVL reference overlay")

    ax = axes[0, 0]
    ax.plot(time_overlay["dvl"][:, 0], time_overlay["dvl"][:, 1], "r-", label="DVL reference")
    ax.plot(time_overlay["vins"][:, 0], time_overlay["vins"][:, 1], "b-", label="VINS log")
    ax.set_title("Time-aligned XY")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(shape_overlay["dvl"][:, 0], shape_overlay["dvl"][:, 1], "r-", label="DVL reference")
    ax.plot(shape_overlay["vins"][:, 0], shape_overlay["vins"][:, 1], "b-", label="VINS log")
    ax.set_title("Arclength shape XY")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 0]
    for axis, name in enumerate("xyz"):
        ax.plot(time_overlay["t"], time_overlay["vins"][:, axis] - time_overlay["dvl"][:, axis], label=f"err_{name}")
    ax.set_title("Time-aligned axis errors")
    ax.set_xlabel("sec")
    ax.set_ylabel("m")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 1]
    metric = summary["time_aligned"]
    lines = [
        f"Time RMSE3: {metric['rmse_3d_m']:.4f} m",
        f"Time match: {metric['rmse_match_percent']:.2f}%",
        f"Time corr x/y/z: {metric['axis_corr']['x']:.3f}, {metric['axis_corr']['y']:.3f}, {metric['axis_corr']['z']:.3f}",
        f"VINS length: {metric['vins_path_length_m']:.3f} m",
        f"DVL length: {metric['dvl_path_length_m']:.3f} m",
        f"Mean dir error: {metric['mean_direction_error_deg']:.2f} deg",
    ]
    ax.axis("off")
    ax.text(0.0, 0.95, "\n".join(lines), va="top", family="monospace")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_single_xy_plot(path: Path, shape_overlay: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(shape_overlay["dvl"][:, 0], shape_overlay["dvl"][:, 1], color="crimson", linewidth=3.0, label="DVL reference only")
    ax.plot(shape_overlay["vins"][:, 0], shape_overlay["vins"][:, 1], color="#17b7e6", linewidth=3.0, label="VINS stereo+IMU")
    marker_indices = [0, 25, 50, 75, len(shape_overlay["vins"]) - 1]
    for index in marker_indices:
        index = max(0, min(index, len(shape_overlay["vins"]) - 1))
        ax.scatter(shape_overlay["vins"][index, 0], shape_overlay["vins"][index, 1], color="#17b7e6", s=45, zorder=3)
        ax.text(shape_overlay["vins"][index, 0], shape_overlay["vins"][index, 1], f" {index}", fontsize=10)
    metric = summary["shape_arclength_aligned"]
    title = (
        "Shape overlay: VINS uses stereo+IMU only, DVL is full reference\n"
        f"yaw={summary['vins_yaw_offset_deg']:.1f} deg, match={metric['rmse_match_percent']:.2f}%, "
        f"VINS={metric['vins_path_length_m']:.2f} m, DVL={metric['dvl_path_length_m']:.2f} m"
    )
    ax.set_title(title)
    ax.set_xlabel("RViz X / m")
    ax.set_ylabel("RViz Y / m")
    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
