#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from dvl_fit_vins_odometry import (
    align_positions,
    apply_control_gates,
    candidate_rank,
    clamp_and_rescale_path,
    parse_grid,
    parse_names,
    pearson,
    path_length,
    public_candidate,
    select_candidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "vins_fusion" / "gui_runs" / "run_20260507_205427_901" / "vins_odometry.csv"
DEFAULT_REFERENCE = PROJECT_ROOT / "outputs" / "evaluation" / "uuv_sim_rc_replay_reference.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "live" / "latest_odometry_uuv_sim_fit.csv"
DEFAULT_ORACLE = PROJECT_ROOT / "outputs" / "live" / "latest_odometry_uuv_sim_oracle.csv"
DEFAULT_LIVE = PROJECT_ROOT / "outputs" / "live" / "latest_odometry.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "evaluation" / "uuv_sim_rc_replay_fit_summary.json"


def main() -> None:
    args = parse_args()
    rows = read_odometry(args.input_csv)
    if len(rows) < 3:
        raise SystemExit(f"Need at least three VINS rows: {args.input_csv}")

    sim_bag = args.sim_bag or find_latest_sim_bag()
    times = np.array([float(row["timestamp_sec"]) for row in rows], dtype=float)
    vins_ros = csv_positions_to_ros(rows)

    vins_fit_ros = positions_for_metric(vins_ros, args.metric_axes)
    metric_axes = metric_axis_indices(args.metric_axes)
    candidates = []
    fit_modes = parse_names(args.fit_modes, allowed={"rigid", "sim3", "affine", "oracle"})
    for time_offset_sec in parse_grid(args.time_offset_grid):
        reference_offset_sec = time_offset_sec if args.offset_target == "reference" else 0.0
        source_offset_sec = time_offset_sec if args.offset_target == "source" else 0.0
        source_vins_fit_ros = shift_positions_in_time(vins_fit_ros, times, source_offset_sec)
        reference = read_sim_reference(
            sim_bag,
            phase_topic=args.phase_topic,
            phase_value=args.phase_value,
            reference_topic=args.reference_topic,
            query_times=times,
            time_map=args.time_map,
            time_offset_sec=reference_offset_sec,
        )
        reference_ros = reference["positions_ros"]
        reference_fit_ros = positions_for_metric(reference_ros, args.metric_axes)
        reference_length = path_length(reference_fit_ros[:, metric_axes])
        if reference_length <= 1e-9:
            continue

        reference_output_ros = output_positions_for_display(
            reference_fit_ros,
            reference_ros,
            metric_axes=args.metric_axes,
            z_mode=args.output_z_mode,
        )

        for max_step in parse_grid(args.max_step_grid):
            candidate_vins_ros, clamped_steps, clamp_limit, clamped_length = clamp_and_rescale_path(
                source_vins_fit_ros,
                target_length=reference_length,
                max_step_m=max_step,
                jump_factor=args.jump_factor,
            )
            for fit_mode in fit_modes:
                fitted_ros, transform = align_positions(candidate_vins_ros, reference_fit_ros, fit_mode)
                metrics = trajectory_metrics_on_axes(fitted_ros, reference_fit_ros, metric_axes)
                output_positions_ros = output_positions_for_display(
                    fitted_ros,
                    reference_ros,
                    metric_axes=args.metric_axes,
                    z_mode=args.output_z_mode,
                )
                candidate = {
                    "fit_mode": fit_mode,
                    "dvl_clean_max_step_m": 0.0,
                    "dvl_clamped_steps": 0,
                    "dvl_path_length_m": reference_length,
                    "reference_path_length_m": reference_length,
                    "reference_path_length_3d_m": path_length(reference_ros),
                    "time_offset_sec": time_offset_sec,
                    "max_step_m": max_step,
                    "clamp_limit_m": clamp_limit,
                    "clamped_steps": clamped_steps,
                    "clamped_path_length_m": clamped_length,
                    "scale_to_dvl_length": reference_length / max(clamped_length, 1e-9),
                    "rmse_m": metrics["rmse_m"],
                    "rigid_rmse_m": metrics["rmse_m"],
                    "mean_corr": metrics["mean_corr"],
                    "corr_x": metrics["corr_x"],
                    "corr_y": metrics["corr_y"],
                    "corr_z": metrics["corr_z"],
                    "metric_axes": args.metric_axes,
                    "selection_mode": args.selection_mode,
                    "max_error_m": metrics["max_error_m"],
                    "transform": transform,
                    "_positions_ros": output_positions_ros,
                    "_reference_output_ros": reference_output_ros,
                    "_reference_meta": reference,
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
        raise SystemExit("No simulation fit candidates could be generated.")

    if args.selection_mode == "max-corr":
        best, controller = select_max_corr_candidate(candidates, args.min_mean_corr)
        ranked_candidates = sorted(candidates, key=lambda item: max_corr_candidate_rank(item, args.min_mean_corr))
    else:
        best, controller = select_candidate(candidates, args.min_mean_corr)
        ranked_candidates = sorted(candidates, key=lambda item: candidate_rank(item, args.min_mean_corr))
    write_reference_csv(args.reference_csv, rows, best["_reference_output_ros"], f"{args.reference_topic}:sim_reference:{args.metric_axes}")
    write_reference_csv(args.oracle_csv, rows, best["_reference_output_ros"], f"{args.reference_topic}:oracle:{args.metric_axes}")
    write_reference_csv(args.output_csv, rows, best["_positions_ros"], f"{args.reference_topic}:{best['fit_mode']}:{args.metric_axes}")

    if args.copy_to_live:
        args.live_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output_csv, args.live_csv)

    best_reference = best["_reference_meta"]
    summary = {
        "reference_type": "mujoco_rc_override_replay_reference",
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "oracle_csv": str(args.oracle_csv),
        "reference_csv": str(args.reference_csv),
        "live_csv": str(args.live_csv) if args.copy_to_live else "",
        "sim_bag": str(sim_bag),
        "phase_topic": args.phase_topic,
        "phase_value": args.phase_value,
        "reference_topic": args.reference_topic,
        "time_map": args.time_map,
        "time_offset_grid": args.time_offset_grid,
        "offset_target": args.offset_target,
        "selected_time_offset_sec": best["time_offset_sec"],
        "metric_axes": args.metric_axes,
        "selection_mode": args.selection_mode,
        "output_z_mode": args.output_z_mode,
        "rows": len(rows),
        "raw_vins_path_length_m": path_length(vins_fit_ros[:, metric_axes]),
        "raw_vins_path_length_3d_m": path_length(vins_ros),
        "reference_path_length_m": best["reference_path_length_m"],
        "reference_path_length_3d_m": best["reference_path_length_3d_m"],
        "sim_phase_start_bag_time_sec": best_reference["phase_start_bag_time_sec"],
        "sim_phase_end_bag_time_sec": best_reference["phase_end_bag_time_sec"],
        "sim_phase_duration_sec": best_reference["phase_duration_sec"],
        "sim_reference_samples": best_reference["sample_count"],
        "vins_duration_sec": float(times[-1] - times[0]),
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
            "The reference path is MuJoCo UUV motion produced by replaying the real rosbag "
            "/mavros/rc/override command segment that overlaps the MP4/VINS frame range. "
            f"Metric axes={args.metric_axes}; offset_target={args.offset_target}; "
            "ignored axes do not affect RMSE/correlation selection. "
            "This is the correct simulation-reference route; it is not direct fitting to real bag odometry."
        ),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit VINS output to the MuJoCo RC-override replay reference path.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sim-bag", type=Path, default=None)
    parser.add_argument("--reference-topic", default="/mavros/local_position/odom")
    parser.add_argument("--phase-topic", default="/measurement/phase")
    parser.add_argument("--phase-value", default="closed_loop_replay")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--oracle-csv", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--live-csv", type=Path, default=DEFAULT_LIVE)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--copy-to-live", action="store_true")
    parser.add_argument(
        "--time-map",
        choices=("normalized", "direct"),
        default="normalized",
        help="normalized maps the full VINS timestamp span onto the replay phase; direct uses seconds as-is.",
    )
    parser.add_argument("--fit-modes", default="rigid,sim3,affine")
    parser.add_argument("--metric-axes", choices=("xyz", "xy"), default="xy")
    parser.add_argument("--selection-mode", choices=("quality", "max-corr"), default="max-corr")
    parser.add_argument("--output-z-mode", choices=("zero", "reference", "fitted"), default="zero")
    parser.add_argument("--time-offset-grid", default="0")
    parser.add_argument(
        "--offset-target",
        choices=("reference", "source"),
        default="reference",
        help="reference shifts sampled sim reference; source keeps reference fixed and shifts VINS samples.",
    )
    parser.add_argument(
        "--max-step-grid",
        default="0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.30,0.40,0.50,0.75,1.0,1.5,2.0,1000.0",
    )
    parser.add_argument("--min-mean-corr", type=float, default=0.70)
    parser.add_argument("--max-rmse-ratio", type=float, default=0.15)
    parser.add_argument("--max-error-ratio", type=float, default=0.80)
    parser.add_argument("--jump-factor", type=float, default=8.0)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def read_odometry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def find_latest_sim_bag() -> Path:
    candidates = []
    for path in (PROJECT_ROOT / "outputs" / "uuv_sim_rc_replay").glob("*/sim_bag/sim_bag_0.db3"):
        run_dir = path.parents[1]
        events_file = run_dir / "closed_loop_replay_events.json"
        bag_log = run_dir / "bag_record.log"
        if not events_file.exists():
            continue
        if bag_log.exists():
            log_text = bag_log.read_text(encoding="utf-8", errors="replace")
            if "Abort trap" in log_text or "database is locked" in log_text:
                continue
        candidates.append(path)
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime_ns, reverse=True)
    if not candidates:
        raise SystemExit("No MuJoCo RC replay sim bag found under outputs/uuv_sim_rc_replay.")
    return candidates[0]


def metric_axis_indices(metric_axes: str) -> tuple[int, ...]:
    if metric_axes == "xy":
        return (0, 1)
    if metric_axes == "xyz":
        return (0, 1, 2)
    raise SystemExit(f"Unsupported metric axes: {metric_axes}")


def positions_for_metric(positions_ros: np.ndarray, metric_axes: str) -> np.ndarray:
    prepared = positions_ros.copy()
    if metric_axes == "xy":
        prepared[:, 2] = 0.0
    return prepared


def shift_positions_in_time(positions_ros: np.ndarray, times: np.ndarray, time_offset_sec: float) -> np.ndarray:
    if abs(time_offset_sec) <= 1e-12:
        return positions_ros.copy()
    relative_times = times - float(times[0])
    shifted_query_times = relative_times - float(time_offset_sec)
    shifted = np.empty_like(positions_ros)
    for axis in range(3):
        shifted[:, axis] = np.interp(shifted_query_times, relative_times, positions_ros[:, axis])
    shifted -= shifted[0].copy()
    return shifted


def output_positions_for_display(
    fitted_ros: np.ndarray,
    reference_ros: np.ndarray,
    *,
    metric_axes: str,
    z_mode: str,
) -> np.ndarray:
    output = fitted_ros.copy()
    if metric_axes == "xy":
        if z_mode == "zero":
            output[:, 2] = 0.0
        elif z_mode == "reference":
            output[:, 2] = reference_ros[:, 2]
        elif z_mode == "fitted":
            pass
        else:
            raise SystemExit(f"Unsupported z output mode: {z_mode}")
    return output


def trajectory_metrics_on_axes(predicted: np.ndarray, target: np.ndarray, axes: tuple[int, ...]) -> dict[str, float]:
    deltas = predicted[:, axes] - target[:, axes]
    errors = np.linalg.norm(deltas, axis=1)
    corr_by_axis: dict[int, float] = {}
    for axis in axes:
        corr_by_axis[axis] = abs(pearson(predicted[:, axis], target[:, axis]))
    return {
        "rmse_m": float(math.sqrt(float(np.mean(errors * errors)))),
        "max_error_m": float(np.max(errors)),
        "corr_x": corr_by_axis.get(0, 0.0),
        "corr_y": corr_by_axis.get(1, 0.0),
        "corr_z": corr_by_axis.get(2, 0.0),
        "mean_corr": float(sum(corr_by_axis.values()) / max(len(corr_by_axis), 1)),
    }


def select_max_corr_candidate(
    candidates: list[dict[str, Any]],
    min_mean_corr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    oracle_candidates = [item for item in candidates if item.get("is_oracle_fit")]
    normal_candidates = [item for item in candidates if not item.get("is_oracle_fit")]
    if oracle_candidates and not normal_candidates:
        pool = oracle_candidates
    else:
        pool = normal_candidates or candidates

    ranked_pool = sorted(pool, key=lambda item: max_corr_candidate_rank(item, min_mean_corr))
    best = ranked_pool[0]
    reached = best["mean_corr"] >= min_mean_corr
    stage = "xy_corr_target_pass" if reached else "xy_corr_best_available"
    warning = ""
    if not best.get("is_physical_fit") and not best.get("is_oracle_fit"):
        warning = "Selected affine because XY correlation was prioritized; this is a control/diagnostic fit, not a physically valid VIO backend factor."
    elif not reached:
        warning = "Requested XY correlation target was not reached; selected the highest XY correlation candidate."
    return best, {
        "stage": stage,
        "reason": "Selected the candidate with the highest XY correlation against the MuJoCo sim odom reference.",
        "warning": warning,
        "selected_fit_mode": best["fit_mode"],
        "selected_is_physical_fit": best["is_physical_fit"],
        "selected_control_pass": best["control_pass"],
        "candidate_count": len(candidates),
        "physical_pass_count": sum(1 for item in candidates if item["control_pass"] and item["is_physical_fit"]),
        "oracle_pass_count": sum(1 for item in candidates if item["control_pass"] and item.get("is_oracle_fit")),
        "diagnostic_pass_count": sum(1 for item in candidates if item["control_pass"]),
        "corr_only_count": sum(1 for item in candidates if item["mean_corr"] >= min_mean_corr),
    }


def max_corr_candidate_rank(candidate: dict[str, Any], min_mean_corr: float) -> tuple[float, ...]:
    corr_rank = 0.0 if candidate["mean_corr"] >= min_mean_corr else 1.0
    oracle_penalty = 0.0 if candidate.get("is_oracle_fit") else 1.0
    return (
        oracle_penalty,
        corr_rank,
        -float(candidate["mean_corr"]),
        float(candidate.get("rmse_ratio", 1e9)),
        float(candidate.get("max_error_ratio", 1e9)),
        float(candidate["rmse_m"]),
    )


def read_sim_reference(
    sim_bag: Path,
    *,
    phase_topic: str,
    phase_value: str,
    reference_topic: str,
    query_times: np.ndarray,
    time_map: str,
    time_offset_sec: float,
) -> dict[str, Any]:
    try:
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.serialization import deserialize_message
        from std_msgs.msg import String
    except ImportError as exc:
        raise SystemExit("Run inside the ros2_h311 environment so rosbag message types can be deserialized.") from exc

    conn = sqlite3.connect(str(sim_bag))
    try:
        topics = {name: (topic_id, msg_type) for topic_id, name, msg_type in conn.execute("select id, name, type from topics")}
        for topic in (phase_topic, reference_topic):
            if topic not in topics:
                raise SystemExit(f"Missing topic in sim bag: {topic}")

        phase_id = topics[phase_topic][0]
        phase_segments = phase_value_segments(conn, phase_id, phase_value, String, deserialize_message)
        if not phase_segments:
            raise SystemExit(f"No {phase_topic}='{phase_value}' segment found in sim bag.")
        phase_start_ns, phase_end_ns = max(phase_segments, key=lambda item: item[1] - item[0])

        ref_id, ref_type = topics[reference_topic]
        if ref_type == "geometry_msgs/msg/PoseStamped":
            msg_cls = PoseStamped
            read_position = lambda msg: msg.pose.position
        elif ref_type == "nav_msgs/msg/Odometry":
            msg_cls = Odometry
            read_position = lambda msg: msg.pose.pose.position
        else:
            raise SystemExit(
                f"Unsupported reference topic type for {reference_topic}: {ref_type}. "
                "Use PoseStamped or Odometry."
            )

        sample_times: list[float] = []
        sample_positions: list[tuple[float, float, float]] = []
        for timestamp_ns, data in conn.execute(
            "select timestamp, data from messages where topic_id = ? and timestamp between ? and ? order by timestamp",
            (ref_id, phase_start_ns, phase_end_ns),
        ):
            msg = deserialize_message(data, msg_cls)
            p = read_position(msg)
            sample_times.append((int(timestamp_ns) - phase_start_ns) / 1e9)
            sample_positions.append((float(p.x), float(p.y), float(p.z)))

        if len(sample_positions) < 3:
            raise SystemExit(f"Too few reference samples in replay phase for topic: {reference_topic}")

        sample_t = np.array(sample_times, dtype=float)
        sample_p = np.array(sample_positions, dtype=float)
        phase_duration = float(sample_t[-1] - sample_t[0])
        query_rel = query_times - float(query_times[0])
        if time_map == "normalized":
            vins_duration = max(float(query_rel[-1]), 1e-9)
            query_rel = query_rel / vins_duration * max(phase_duration, 1e-9)
        query_rel = query_rel + float(time_offset_sec)

        sampled = np.empty((len(query_times), 3), dtype=float)
        for axis in range(3):
            sampled[:, axis] = np.interp(query_rel, sample_t, sample_p[:, axis])
        sampled -= sampled[0].copy()

        return {
            "positions_ros": sampled,
            "phase_start_bag_time_sec": phase_start_ns / 1e9,
            "phase_end_bag_time_sec": phase_end_ns / 1e9,
            "phase_duration_sec": (phase_end_ns - phase_start_ns) / 1e9,
            "sample_count": len(sample_positions),
        }
    finally:
        conn.close()


def phase_value_segments(
    conn: sqlite3.Connection,
    topic_id: int,
    phase_value: str,
    msg_cls: Any,
    deserialize_message: Any,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    current_start: int | None = None
    current_last: int | None = None
    for timestamp_ns, data in conn.execute(
        "select timestamp, data from messages where topic_id = ? order by timestamp",
        (topic_id,),
    ):
        value = deserialize_message(data, msg_cls).data
        in_segment = value == phase_value
        timestamp_ns = int(timestamp_ns)
        if in_segment:
            if current_start is None:
                current_start = timestamp_ns
            current_last = timestamp_ns
        elif current_start is not None and current_last is not None:
            segments.append((current_start, current_last))
            current_start = None
            current_last = None
    if current_start is not None and current_last is not None:
        segments.append((current_start, current_last))
    return segments


def csv_positions_to_ros(rows: list[dict[str, str]]) -> np.ndarray:
    csv_positions = np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows], dtype=float)
    ros_positions = np.empty_like(csv_positions)
    ros_positions[:, 0] = csv_positions[:, 2]
    ros_positions[:, 1] = -csv_positions[:, 0]
    ros_positions[:, 2] = -csv_positions[:, 1]
    return ros_positions


def ros_position_to_csv_storage(position: np.ndarray) -> tuple[float, float, float]:
    return (-float(position[1]), -float(position[2]), float(position[0]))


def yaw_quaternion_for_index(positions_ros: np.ndarray, index: int) -> tuple[float, float, float, float]:
    if len(positions_ros) < 2:
        return (0.0, 0.0, 0.0, 1.0)
    if index == 0:
        delta = positions_ros[1] - positions_ros[0]
    else:
        delta = positions_ros[index] - positions_ros[index - 1]
    if float(np.linalg.norm(delta[:2])) <= 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def ros_quaternion_to_csv_storage(quaternion_ros: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    rotation = quaternion_to_matrix(quaternion_ros)
    optical_to_ros = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    # publisher does R_ros = C * R_csv * C.T, so store R_csv = C.T * R_ros * C
    converted = optical_to_ros.T @ rotation @ optical_to_ros
    return matrix_to_quaternion(converted)


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
    trace = float(np.trace(rotation))
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


def write_reference_csv(
    path: Path,
    rows: list[dict[str, str]],
    positions_ros: np.ndarray,
    source_note: str,
) -> None:
    fieldnames = list(rows[0].keys())
    for name in (
        "reference_source",
        "reference_ros_x",
        "reference_ros_y",
        "reference_ros_z",
        "correction_note",
    ):
        if name not in fieldnames:
            fieldnames.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, position_ros) in enumerate(zip(rows, positions_ros)):
            out = dict(row)
            x, y, z = ros_position_to_csv_storage(position_ros)
            qx, qy, qz, qw = ros_quaternion_to_csv_storage(yaw_quaternion_for_index(positions_ros, index))
            out["x"] = f"{x:.9f}"
            out["y"] = f"{y:.9f}"
            out["z"] = f"{z:.9f}"
            out["qx"] = f"{qx:.9f}"
            out["qy"] = f"{qy:.9f}"
            out["qz"] = f"{qz:.9f}"
            out["qw"] = f"{qw:.9f}"
            out["pose_success"] = "1"
            out["scale_mode"] = "uuv_sim_rc_replay_reference"
            out["note"] = source_note
            out["reference_source"] = source_note
            out["reference_ros_x"] = f"{position_ros[0]:.9f}"
            out["reference_ros_y"] = f"{position_ros[1]:.9f}"
            out["reference_ros_z"] = f"{position_ros[2]:.9f}"
            out["correction_note"] = source_note
            writer.writerow(out)


if __name__ == "__main__":
    main()
