#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BAG = (
    Path.home()
    / "Desktop"
    / "uuv_sim"
    / "real_robot_ros_bag"
    / "extracted_2026_04_01"
    / "bag_2026-04-01_20-20-30"
    / "bag_2026-04-01_20-20-30_0.db3"
)
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "vins_fusion" / "gui_runs" / "run_20260507_205427_901" / "vins_odometry.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "live" / "latest_odometry_dvl_fit.csv"
DEFAULT_DVL_CSV = PROJECT_ROOT / "outputs" / "evaluation" / "dvl_position_reference.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "evaluation" / "dvl_fit_summary.json"
PHYSICAL_FIT_MODES = {"rigid", "sim3"}


def main() -> None:
    args = parse_args()
    rows = read_odometry(args.input_csv)
    rows = filter_rows_by_time(rows, args.start_sec, args.duration_sec)
    times = np.array([float(row["timestamp_sec"]) for row in rows], dtype=float)
    raw_csv_positions = np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows], dtype=float)
    raw_positions = convert_vins_positions(raw_csv_positions, args.metric_frame)
    if len(rows) < 3:
        raise SystemExit("Need at least three VINS poses.")

    if args.dvl_mode == "position":
        dvl_samples = read_dvl_position_path(
            args.bag,
            args.left_topic,
            args.dvl_position_topic,
            args.reference_time_source,
            args.reference_origin_ns,
            args.reference_time_offset_sec,
        )
    else:
        dvl_samples = read_dvl_twist_path(
            args.bag,
            args.left_topic,
            args.dvl_twist_topic,
            args.reference_time_source,
            args.reference_origin_ns,
            args.reference_time_offset_sec,
        )
    sampled_dvl_positions = convert_dvl_positions(sample_path(dvl_samples, times), args.metric_frame)

    candidates = []
    fit_modes = parse_names(args.fit_modes, allowed={"rigid", "sim3", "affine", "oracle"})
    for dvl_clean_max_step in parse_grid(args.dvl_clean_max_step_grid):
        dvl_positions, dvl_clamped_steps = clean_reference_path(sampled_dvl_positions, dvl_clean_max_step)
        dvl_length = path_length(dvl_positions)
        if dvl_length <= 1e-9:
            continue
        for max_step in parse_grid(args.max_step_grid):
            candidate_positions, clamped_steps, clamp_limit, clamped_length = clamp_and_rescale_path(
                raw_positions,
                target_length=dvl_length,
                max_step_m=max_step,
                jump_factor=args.jump_factor,
            )
            for fit_mode in fit_modes:
                fitted, transform = align_positions(candidate_positions, dvl_positions, fit_mode)
                metrics = trajectory_metrics(fitted, dvl_positions)
                candidate = {
                    "fit_mode": fit_mode,
                    "dvl_clean_max_step_m": dvl_clean_max_step,
                    "dvl_clamped_steps": dvl_clamped_steps,
                    "dvl_path_length_m": dvl_length,
                    "max_step_m": max_step,
                    "clamp_limit_m": clamp_limit,
                    "clamped_steps": clamped_steps,
                    "clamped_path_length_m": clamped_length,
                    "scale_to_dvl_length": dvl_length / max(clamped_length, 1e-9),
                    "rmse_m": metrics["rmse_m"],
                    "rigid_rmse_m": metrics["rmse_m"],
                    "mean_corr": metrics["mean_corr"],
                    "corr_x": metrics["corr_x"],
                    "corr_y": metrics["corr_y"],
                    "corr_z": metrics["corr_z"],
                    "max_error_m": metrics["max_error_m"],
                    "transform": transform,
                    "_positions": fitted,
                    "_dvl_positions": dvl_positions,
                }
                apply_control_gates(
                    candidate,
                    row_count=len(rows),
                    min_mean_corr=args.min_mean_corr,
                    max_rmse_ratio=args.max_rmse_ratio,
                    max_error_ratio=args.max_error_ratio,
                )
                candidates.append(candidate)

    if not candidates:
        raise SystemExit("No DVL fit candidates could be generated.")
    best, controller = select_candidate(candidates, args.min_mean_corr)
    ranked_candidates = sorted(candidates, key=lambda item: candidate_rank(item, args.min_mean_corr))
    write_reference_csv(args.dvl_reference_csv, times, best["_dvl_positions"], args.metric_frame)
    write_fitted_csv(args.output_csv, rows, best["_positions"], best, best["dvl_path_length_m"], args.metric_frame)
    summary = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "dvl_reference_csv": str(args.dvl_reference_csv),
        "start_sec": args.start_sec,
        "duration_sec": args.duration_sec,
        "metric_frame": args.metric_frame,
        "reference_time_source": args.reference_time_source,
        "reference_origin_ns": args.reference_origin_ns,
        "reference_time_offset_sec": args.reference_time_offset_sec,
        "segment_start_timestamp_sec": float(times[0]),
        "segment_end_timestamp_sec": float(times[-1]),
        "segment_duration_sec": float(times[-1] - times[0]),
        "rows": len(rows),
        "raw_vins_path_length_m": path_length(raw_positions),
        "dvl_path_length_m": best["dvl_path_length_m"],
        "dvl_clean_max_step_m": best["dvl_clean_max_step_m"],
        "dvl_clamped_steps": best["dvl_clamped_steps"],
        "min_mean_corr": args.min_mean_corr,
        "max_rmse_ratio": args.max_rmse_ratio,
        "max_error_ratio": args.max_error_ratio,
        "reached_min_corr": best["mean_corr"] >= args.min_mean_corr,
        "controller": controller,
        "best": public_candidate(best),
        "top_candidates": [
            public_candidate(candidate) for candidate in ranked_candidates[: min(args.top_k, len(ranked_candidates))]
        ],
        "note": (
            f"DVL-fit is an offline evaluation/post-process using DVL {args.dvl_mode}. "
            "VINS shape is scaled to DVL path length, then aligned to the DVL path. "
            "Affine mode can raise correlation for diagnostics, but it is not a physically valid VIO backend factor. "
            "Oracle mode publishes the DVL reference itself and is only for visual reference/upper-bound checks."
        ),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit VINS odometry to DVL twist odometry by RMSE/correlation sweep.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dvl-reference-csv", type=Path, default=DEFAULT_DVL_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--left-topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 disables time slicing.")
    parser.add_argument("--dvl-mode", choices=("position", "twist"), default="position")
    parser.add_argument("--dvl-position-topic", default="/dvl/position")
    parser.add_argument("--dvl-twist-topic", default="/dvl/twist")
    parser.add_argument(
        "--reference-time-source",
        choices=("header", "db"),
        default="db",
        help="Use ROS header stamps or rosbag DB timestamps for the DVL reference time axis.",
    )
    parser.add_argument(
        "--reference-origin-ns",
        type=int,
        default=0,
        help="Optional explicit reference origin in nanoseconds. 0 uses the first left image time.",
    )
    parser.add_argument(
        "--reference-time-offset-sec",
        type=float,
        default=0.0,
        help="Offset added to DVL sample times. Use 1.0 for ROS1 bags written with a +1s playback stamp offset.",
    )
    parser.add_argument(
        "--metric-frame",
        choices=("rviz", "raw"),
        default="rviz",
        help=(
            "rviz: compare VINS and DVL in the same ROS/RViz frame "
            "(VINS x,y,z -> z,-x,-y; DVL x,y,z -> x,-y,z). "
            "raw: keep legacy direct x,y,z comparison."
        ),
    )
    parser.add_argument(
        "--dvl-clean-max-step-grid",
        default="0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50,0",
        help="Comma-separated DVL reference jump clamp values. 0 disables cleaning.",
    )
    parser.add_argument(
        "--max-step-grid",
        default="0.05,0.08,0.10,0.15,0.20,0.30,0.40,0.50,0.75,1.0,1.5,2.0,3.0,5.0,1000.0",
    )
    parser.add_argument("--fit-modes", default="rigid,sim3,affine")
    parser.add_argument("--min-mean-corr", type=float, default=0.70)
    parser.add_argument(
        "--max-rmse-ratio",
        type=float,
        default=0.08,
        help="Reject candidates whose RMSE is larger than this fraction of DVL path length.",
    )
    parser.add_argument(
        "--max-error-ratio",
        type=float,
        default=0.60,
        help="Reject candidates whose max point error is larger than this fraction of DVL path length.",
    )
    parser.add_argument("--jump-factor", type=float, default=8.0)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def read_odometry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def convert_vins_positions(positions: np.ndarray, metric_frame: str) -> np.ndarray:
    if metric_frame == "raw":
        return positions
    converted = np.empty_like(positions)
    converted[:, 0] = positions[:, 2]
    converted[:, 1] = -positions[:, 0]
    converted[:, 2] = -positions[:, 1]
    return converted


def convert_dvl_positions(positions: np.ndarray, metric_frame: str) -> np.ndarray:
    if metric_frame == "raw":
        return positions
    converted = np.empty_like(positions)
    converted[:, 0] = positions[:, 0]
    converted[:, 1] = -positions[:, 1]
    converted[:, 2] = positions[:, 2]
    return converted


def metric_position_to_csv_storage(position: np.ndarray, metric_frame: str) -> tuple[float, float, float]:
    if metric_frame == "raw":
        return (float(position[0]), float(position[1]), float(position[2]))
    # Store so ros2_publish_odometry.py --coordinate-frame ros reconstructs the same RViz point.
    return (-float(position[1]), -float(position[2]), float(position[0]))


def metric_quaternion_to_csv_storage(
    quaternion_metric: tuple[float, float, float, float],
    metric_frame: str,
) -> tuple[float, float, float, float]:
    if metric_frame == "raw":
        return normalize_quaternion(quaternion_metric)
    rotation = quaternion_to_matrix(quaternion_metric)
    optical_to_ros = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    # ros2_publish_odometry.py reconstructs R_metric = C * R_csv * C.T.
    converted = optical_to_ros.T @ rotation @ optical_to_ros
    return matrix_to_quaternion(converted)


def filter_rows_by_time(rows: list[dict[str, str]], start_sec: float, duration_sec: float) -> list[dict[str, str]]:
    if not rows:
        return rows
    origin = float(rows[0]["timestamp_sec"])
    start = origin + max(0.0, start_sec)
    end = math.inf if duration_sec <= 0.0 else start + duration_sec
    filtered = [row for row in rows if start <= float(row["timestamp_sec"]) <= end]
    if len(filtered) < 3:
        raise SystemExit(
            f"Time slice produced too few rows: start_sec={start_sec}, duration_sec={duration_sec}, rows={len(filtered)}"
        )
    offset = float(filtered[0]["timestamp_sec"])
    normalized = []
    for index, row in enumerate(filtered):
        out = dict(row)
        out["frame_index"] = str(index)
        out["timestamp_sec"] = f"{float(row['timestamp_sec']) - offset:.9f}"
        normalized.append(out)
    return normalized


def read_dvl_twist_path(
    bag: Path,
    left_topic: str,
    dvl_topic: str,
    time_source: str,
    reference_origin_ns: int,
    reference_time_offset_sec: float,
) -> list[tuple[float, np.ndarray]]:
    try:
        from geometry_msgs.msg import TwistWithCovarianceStamped
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise SystemExit("Run this script inside the ros2_h311 conda environment for DVL deserialization.") from exc

    conn = sqlite3.connect(str(bag))
    try:
        topic_ids = {name: topic_id for topic_id, name in conn.execute("select id, name from topics")}
        if left_topic not in topic_ids:
            raise SystemExit(f"Missing image topic: {left_topic}")
        if dvl_topic not in topic_ids:
            raise SystemExit(f"Missing DVL topic: {dvl_topic}")
        origin_ns = resolve_reference_origin_ns(
            conn,
            topic_ids[left_topic],
            Image,
            time_source,
            reference_origin_ns,
            left_topic,
        )

        samples: list[tuple[float, np.ndarray]] = []
        position = np.zeros(3, dtype=float)
        prev_t: float | None = None
        prev_v: np.ndarray | None = None
        for timestamp_ns, data in conn.execute(
            "select timestamp, data from messages where topic_id = ? order by timestamp",
            (topic_ids[dvl_topic],),
        ):
            msg = deserialize_message(data, TwistWithCovarianceStamped)
            sample_ns = timestamp_ns if time_source == "db" else stamp_to_ns(msg.header.stamp)
            t = (sample_ns - origin_ns) / 1e9 + reference_time_offset_sec
            v = msg.twist.twist.linear
            velocity = np.array([v.x, v.y, v.z], dtype=float)
            if prev_t is not None and prev_v is not None:
                dt = max(0.0, t - prev_t)
                position = position + 0.5 * (prev_v + velocity) * dt
            samples.append((t, position.copy()))
            prev_t = t
            prev_v = velocity
        if len(samples) < 3:
            raise SystemExit(f"Too few DVL samples from {dvl_topic}")
        return samples
    finally:
        conn.close()


def read_dvl_position_path(
    bag: Path,
    left_topic: str,
    dvl_topic: str,
    time_source: str,
    reference_origin_ns: int,
    reference_time_offset_sec: float,
) -> list[tuple[float, np.ndarray]]:
    try:
        from dvl_msgs.msg import DVLDR
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise SystemExit(
            "DVL position source requires dvl_msgs. Build/source ros2_overlay_ws/install/setup.bash, "
            "or run this script with --dvl-mode twist."
        ) from exc

    conn = sqlite3.connect(str(bag))
    try:
        topic_ids = {name: topic_id for topic_id, name in conn.execute("select id, name from topics")}
        if left_topic not in topic_ids:
            raise SystemExit(f"Missing image topic: {left_topic}")
        if dvl_topic not in topic_ids:
            raise SystemExit(f"Missing DVL topic: {dvl_topic}")
        origin_ns = resolve_reference_origin_ns(
            conn,
            topic_ids[left_topic],
            Image,
            time_source,
            reference_origin_ns,
            left_topic,
        )

        samples: list[tuple[float, np.ndarray]] = []
        origin_position: np.ndarray | None = None
        for timestamp_ns, data in conn.execute(
            "select timestamp, data from messages where topic_id = ? order by timestamp",
            (topic_ids[dvl_topic],),
        ):
            msg = deserialize_message(data, DVLDR)
            sample_ns = timestamp_ns if time_source == "db" else stamp_to_ns(msg.header.stamp)
            t = (sample_ns - origin_ns) / 1e9 + reference_time_offset_sec
            position = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=float)
            if origin_position is None:
                origin_position = position.copy()
            samples.append((t, position - origin_position))
        if len(samples) < 3:
            raise SystemExit(f"Too few DVL position samples from {dvl_topic}")
        return samples
    finally:
        conn.close()


def resolve_reference_origin_ns(
    conn: sqlite3.Connection,
    left_topic_id: int,
    message_type: Any,
    time_source: str,
    reference_origin_ns: int,
    left_topic: str,
) -> int:
    if reference_origin_ns:
        return reference_origin_ns
    row = conn.execute(
        "select timestamp, data from messages where topic_id = ? order by timestamp limit 1",
        (left_topic_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"No messages for image topic: {left_topic}")
    if time_source == "db":
        return int(row[0])
    first_image = deserialize_message(row[1], message_type)
    return stamp_to_ns(first_image.header.stamp)


def sample_path(samples: list[tuple[float, np.ndarray]], query_times: np.ndarray) -> np.ndarray:
    sample_times = np.array([item[0] for item in samples], dtype=float)
    sample_positions = np.array([item[1] for item in samples], dtype=float)
    sampled = np.empty((len(query_times), 3), dtype=float)
    for axis in range(3):
        sampled[:, axis] = np.interp(query_times, sample_times, sample_positions[:, axis])
    return sampled


def clean_reference_path(positions: np.ndarray, max_step_m: float) -> tuple[np.ndarray, int]:
    if max_step_m <= 0.0 or len(positions) < 2:
        return positions, 0
    cleaned = np.empty_like(positions)
    cleaned[0] = positions[0]
    clamped = 0
    for index, current in enumerate(positions[1:], start=1):
        delta = current - cleaned[index - 1]
        length = float(np.linalg.norm(delta))
        if length > max_step_m:
            delta = delta * (max_step_m / length)
            clamped += 1
        cleaned[index] = cleaned[index - 1] + delta
    return cleaned, clamped


def clamp_and_rescale_path(
    positions: np.ndarray,
    *,
    target_length: float,
    max_step_m: float,
    jump_factor: float,
) -> tuple[np.ndarray, int, float, float]:
    deltas = np.diff(positions, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    clamp_limit = float(max_step_m)

    clamped = deltas.copy()
    clamped_steps = 0
    for index, length in enumerate(lengths):
        if length > clamp_limit > 0.0:
            clamped[index] *= clamp_limit / length
            clamped_steps += 1
    clamped_length = float(np.linalg.norm(clamped, axis=1).sum())
    scale = target_length / clamped_length if clamped_length > 1e-9 else 1.0
    reconstructed = np.empty_like(positions)
    reconstructed[0] = positions[0]
    for index, delta in enumerate(clamped, start=1):
        reconstructed[index] = reconstructed[index - 1] + delta * scale
    return reconstructed, clamped_steps, clamp_limit, clamped_length


def rigid_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean
    covariance = src_centered.T @ dst_centered / len(source)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = dst_mean - rotation @ src_mean
    aligned = (rotation @ source.T).T + translation
    return aligned, {"rotation": rotation.tolist(), "translation": translation.tolist()}


def sim3_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean
    covariance = src_centered.T @ dst_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        sign[-1] = -1.0
    rotation = vt.T @ np.diag(sign) @ u.T
    variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    scale = float(np.sum(singular_values * sign) / variance) if variance > 1e-12 else 1.0
    translation = dst_mean - scale * (rotation @ src_mean)
    aligned = scale * (rotation @ source.T).T + translation
    return aligned, {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }


def affine_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    design = np.column_stack([source, np.ones(len(source), dtype=float)])
    coeff, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    aligned = design @ coeff
    return aligned, {
        "linear": coeff[:3, :].T.tolist(),
        "translation": coeff[3, :].tolist(),
    }


def align_positions(source: np.ndarray, target: np.ndarray, fit_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    if fit_mode == "rigid":
        aligned, transform = rigid_align(source, target)
    elif fit_mode == "sim3":
        aligned, transform = sim3_align(source, target)
    elif fit_mode == "affine":
        aligned, transform = affine_align(source, target)
    elif fit_mode == "oracle":
        aligned = target.copy()
        transform = {"source": "dvl_reference_exact"}
    else:
        raise SystemExit(f"Unsupported fit mode: {fit_mode}")
    transform["fit_mode"] = fit_mode
    return aligned, transform


def trajectory_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(predicted - target, axis=1)
    corr = []
    for axis in range(3):
        corr.append(abs(pearson(predicted[:, axis], target[:, axis])))
    return {
        "rmse_m": float(math.sqrt(float(np.mean(errors * errors)))),
        "max_error_m": float(np.max(errors)),
        "corr_x": corr[0],
        "corr_y": corr[1],
        "corr_z": corr[2],
        "mean_corr": float(sum(corr) / 3.0),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm <= 1e-12 and right_norm <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    denom = left_norm * right_norm
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denom)


def write_fitted_csv(
    path: Path,
    rows: list[dict[str, str]],
    positions: np.ndarray,
    best: dict[str, Any],
    dvl_length: float,
    metric_frame: str,
) -> None:
    fieldnames = list(rows[0].keys())
    for name in ("qx", "qy", "qz", "qw"):
        if name not in fieldnames:
            fieldnames.append(name)
    for name in (
        "dvl_fit_rmse_m",
        "dvl_fit_rmse_ratio",
        "dvl_fit_rmse_match_percent",
        "dvl_fit_mean_corr",
        "dvl_fit_source",
        "dvl_fit_orientation_source",
        "correction_note",
    ):
        if name not in fieldnames:
            fieldnames.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, position) in enumerate(zip(rows, positions)):
            out = dict(row)
            csv_x, csv_y, csv_z = metric_position_to_csv_storage(position, metric_frame)
            qx, qy, qz, qw = metric_quaternion_to_csv_storage(
                tangent_quaternion_for_index(positions, index),
                metric_frame,
            )
            out["x"] = f"{csv_x:.9f}"
            out["y"] = f"{csv_y:.9f}"
            out["z"] = f"{csv_z:.9f}"
            out["qx"] = f"{qx:.9f}"
            out["qy"] = f"{qy:.9f}"
            out["qz"] = f"{qz:.9f}"
            out["qw"] = f"{qw:.9f}"
            out["scale_mode"] = "dvl_fit_metric"
            out["dvl_fit_rmse_m"] = f"{best['rmse_m']:.9f}"
            out["dvl_fit_rmse_ratio"] = f"{best['rmse_ratio']:.9f}"
            out["dvl_fit_rmse_match_percent"] = f"{max(0.0, 1.0 - best['rmse_ratio']) * 100.0:.6f}"
            out["dvl_fit_mean_corr"] = f"{best['mean_corr']:.9f}"
            out["dvl_fit_source"] = f"dvl_position_path_length:{dvl_length:.6f}m"
            out["dvl_fit_orientation_source"] = "fitted_trajectory_tangent"
            out["correction_note"] = (
                f"fit_mode={best['fit_mode']}, dvl_clean_max_step={best['dvl_clean_max_step_m']}, "
                f"max_step={best['max_step_m']}, clamp_limit={best['clamp_limit_m']:.6f}, "
                "orientation=fitted_trajectory_tangent"
            )
            writer.writerow(out)


def write_reference_csv(path: Path, times: np.ndarray, positions: np.ndarray, metric_frame: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp_sec", "x", "y", "z", "qx", "qy", "qz", "qw", "orientation_source"])
        for index, (t, p) in enumerate(zip(times, positions)):
            x, y, z = metric_position_to_csv_storage(p, metric_frame)
            qx, qy, qz, qw = metric_quaternion_to_csv_storage(tangent_quaternion_for_index(positions, index), metric_frame)
            writer.writerow([
                f"{t:.9f}",
                f"{x:.9f}",
                f"{y:.9f}",
                f"{z:.9f}",
                f"{qx:.9f}",
                f"{qy:.9f}",
                f"{qz:.9f}",
                f"{qw:.9f}",
                "dvl_trajectory_tangent",
            ])


def tangent_quaternion_for_index(positions: np.ndarray, index: int) -> tuple[float, float, float, float]:
    if len(positions) < 2:
        return (0.0, 0.0, 0.0, 1.0)
    if index == 0:
        delta = positions[1] - positions[0]
    elif index == len(positions) - 1:
        delta = positions[index] - positions[index - 1]
    else:
        delta = positions[index + 1] - positions[index - 1]
    return direction_quaternion(delta)


def direction_quaternion(delta: np.ndarray) -> tuple[float, float, float, float]:
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    x_axis = delta / norm
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(x_axis @ up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    y_axis = np.cross(up, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return matrix_to_quaternion(rotation)


def quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    qx, qy, qz, qw = normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    return normalize_quaternion((float(qx), float(qy), float(qz), float(qw)))


def normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(value) / norm for value in quaternion)


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if not key.startswith("_") and key != "transform"
    }


def parse_grid(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        stripped = part.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise SystemExit("Empty max-step grid.")
    return values


def parse_names(text: str, *, allowed: set[str]) -> list[str]:
    names = []
    for part in text.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in allowed:
            raise SystemExit(f"Unsupported fit mode '{name}'. Allowed: {', '.join(sorted(allowed))}")
        names.append(name)
    if not names:
        raise SystemExit("Empty fit mode list.")
    return names


def apply_control_gates(
    candidate: dict[str, Any],
    *,
    row_count: int,
    min_mean_corr: float,
    max_rmse_ratio: float,
    max_error_ratio: float,
) -> None:
    dvl_length = max(float(candidate["dvl_path_length_m"]), 1e-9)
    denom_steps = max(row_count - 1, 1)
    candidate["is_oracle_fit"] = candidate["fit_mode"] == "oracle"
    candidate["is_physical_fit"] = candidate["fit_mode"] in PHYSICAL_FIT_MODES
    candidate["rmse_ratio"] = float(candidate["rmse_m"]) / dvl_length
    candidate["max_error_ratio"] = float(candidate["max_error_m"]) / dvl_length
    candidate["dvl_clamp_ratio"] = float(candidate["dvl_clamped_steps"]) / denom_steps
    candidate["vins_clamp_ratio"] = float(candidate["clamped_steps"]) / denom_steps

    gates = {
        "mean_corr": float(candidate["mean_corr"]) >= min_mean_corr,
        "rmse_ratio": float(candidate["rmse_ratio"]) <= max_rmse_ratio,
        "max_error_ratio": float(candidate["max_error_ratio"]) <= max_error_ratio,
    }
    candidate["control_gates"] = gates
    candidate["control_pass"] = all(gates.values())
    candidate["control_failures"] = [name for name, passed in gates.items() if not passed]
    candidate["control_score"] = float(
        candidate["mean_corr"] * 100.0
        - candidate["rmse_ratio"] * 35.0
        - candidate["max_error_ratio"] * 10.0
        - candidate["dvl_clamp_ratio"] * 4.0
        - candidate["vins_clamp_ratio"] * 2.0
    )


def select_candidate(
    candidates: list[dict[str, Any]],
    min_mean_corr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    physical_pass = [item for item in candidates if item["control_pass"] and item["is_physical_fit"]]
    oracle_pass = [item for item in candidates if item["control_pass"] and item.get("is_oracle_fit")]
    diagnostic_pass = [item for item in candidates if item["control_pass"]]
    corr_only = [item for item in candidates if item["mean_corr"] >= min_mean_corr]

    if oracle_pass:
        stage = "oracle_reference"
        pool = oracle_pass
        reason = "Selected the exact DVL reference path as an oracle upper-bound visualization."
        warning = "This is 100% reference playback, not a VINS/VIO estimate."
    elif physical_pass:
        stage = "physical_pass"
        pool = physical_pass
        reason = "Selected a physically valid rigid/sim3 candidate that passed all quality gates."
        warning = ""
    elif diagnostic_pass:
        stage = "diagnostic_affine_pass"
        pool = diagnostic_pass
        reason = "Rigid/sim3 could not reach the target, so the controller selected the best gated diagnostic fit."
        warning = "Selected result may include affine post-fit. Treat it as DVL similarity analysis, not backend VIO accuracy."
    elif corr_only:
        stage = "corr_only_fallback"
        pool = corr_only
        reason = "No candidate passed all gates, so the controller preserved the requested correlation target first."
        warning = "Quality gates failed. Inspect RMSE/max error before using this as a navigation result."
    else:
        stage = "best_available_fallback"
        pool = candidates
        reason = "No candidate reached the requested correlation target, so the controller returned the least bad candidate."
        warning = "Target correlation was not reached."

    ranked_pool = sorted(pool, key=lambda item: candidate_rank(item, min_mean_corr))
    best = ranked_pool[0]
    return best, {
        "stage": stage,
        "reason": reason,
        "warning": warning,
        "selected_fit_mode": best["fit_mode"],
        "selected_is_physical_fit": best["is_physical_fit"],
        "selected_control_pass": best["control_pass"],
        "candidate_count": len(candidates),
        "physical_pass_count": len(physical_pass),
        "oracle_pass_count": len(oracle_pass),
        "diagnostic_pass_count": len(diagnostic_pass),
        "corr_only_count": len(corr_only),
    }


def candidate_rank(candidate: dict[str, Any], min_mean_corr: float) -> tuple[float, ...]:
    stage_rank = 0.0 if candidate.get("control_pass") else 1.0
    corr_rank = 0.0 if candidate["mean_corr"] >= min_mean_corr else 1.0
    physical_penalty = 0.0 if candidate.get("is_physical_fit") else 0.25
    return (
        stage_rank,
        corr_rank,
        physical_penalty,
        float(candidate["rmse_m"]),
        float(candidate.get("max_error_ratio", 1e9)),
        -float(candidate["mean_corr"]),
        -float(candidate.get("control_score", 0.0)),
    )




def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


if __name__ == "__main__":
    main()
