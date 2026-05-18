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

from dvl_fit_vins_odometry import (
    align_positions,
    apply_control_gates,
    candidate_rank,
    clamp_and_rescale_path,
    parse_names,
    parse_grid,
    path_length,
    public_candidate,
    select_candidate,
    trajectory_metrics,
)


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
DEFAULT_REFERENCE = PROJECT_ROOT / "outputs" / "evaluation" / "rc_odom_filtered_reference.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "live" / "latest_odometry_rc_odom_fit.csv"
DEFAULT_ORACLE = PROJECT_ROOT / "outputs" / "live" / "latest_odometry_rc_odom_oracle.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "evaluation" / "rc_odom_reference_summary.json"


def main() -> None:
    args = parse_args()
    rows = read_odometry(args.input_csv)
    if len(rows) < 3:
        raise SystemExit(f"Need at least three VINS rows: {args.input_csv}")

    times = np.array([float(row["timestamp_sec"]) for row in rows], dtype=float)
    raw_vins_ros = csv_positions_to_ros(rows)

    bag_data = read_bag_reference(
        args.bag,
        image_topic=args.image_topic,
        rc_topic=args.rc_topic,
        odom_topic=args.odom_topic,
        times=times,
        rc_padding_sec=args.rc_padding_sec,
    )
    selected_mask = (times >= bag_data["segment_start_sec"]) & (times <= bag_data["segment_end_sec"])
    if selected_mask.sum() < 3:
        raise SystemExit(
            f"RC/MP4 segment has too few frames: {selected_mask.sum()} "
            f"from {bag_data['segment_start_sec']:.3f}..{bag_data['segment_end_sec']:.3f}s"
        )

    selected_rows = [row for row, keep in zip(rows, selected_mask) if bool(keep)]
    selected_times = times[selected_mask]
    vins_ros = raw_vins_ros[selected_mask]
    reference_ros = bag_data["reference_ros"][selected_mask]
    write_reference_csv(args.reference_csv, selected_rows, reference_ros, args.odom_topic)
    write_reference_csv(args.oracle_csv, selected_rows, reference_ros, f"{args.odom_topic}:oracle")

    candidates = []
    fit_modes = parse_names(args.fit_modes, allowed={"rigid", "sim3", "affine", "oracle"})
    reference_length = path_length(reference_ros)
    for max_step in parse_grid(args.max_step_grid):
        candidate_vins_ros, clamped_steps, clamp_limit, clamped_length = clamp_and_rescale_path(
            vins_ros,
            target_length=reference_length,
            max_step_m=max_step,
            jump_factor=args.jump_factor,
        )
        for fit_mode in fit_modes:
            fitted_ros, transform = align_positions(candidate_vins_ros, reference_ros, fit_mode)
            metrics = trajectory_metrics(fitted_ros, reference_ros)
            candidate = {
                "fit_mode": fit_mode,
                "dvl_clean_max_step_m": 0.0,
                "dvl_clamped_steps": 0,
                "dvl_path_length_m": reference_length,
                "reference_path_length_m": reference_length,
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
                "max_error_m": metrics["max_error_m"],
                "transform": transform,
                "_positions_ros": fitted_ros,
            }
            apply_control_gates(
                candidate,
                row_count=len(selected_rows),
                min_mean_corr=args.min_mean_corr,
                max_rmse_ratio=args.max_rmse_ratio,
                max_error_ratio=args.max_error_ratio,
            )
            candidates.append(candidate)

    best, controller = select_candidate(candidates, args.min_mean_corr)
    ranked_candidates = sorted(candidates, key=lambda item: candidate_rank(item, args.min_mean_corr))
    write_reference_csv(args.output_csv, selected_rows, best["_positions_ros"], f"{args.odom_topic}:{best['fit_mode']}")

    summary = {
        "reference_type": "uuv_sim_rc_odom_reference",
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "oracle_csv": str(args.oracle_csv),
        "dvl_reference_csv": str(args.reference_csv),
        "reference_csv": str(args.reference_csv),
        "bag": str(args.bag),
        "image_topic": args.image_topic,
        "rc_topic": args.rc_topic,
        "odom_topic": args.odom_topic,
        "rows": len(selected_rows),
        "source_rows": len(rows),
        "segment_start_sec": bag_data["segment_start_sec"],
        "segment_end_sec": bag_data["segment_end_sec"],
        "segment_duration_sec": float(selected_times[-1] - selected_times[0]),
        "rc_messages_in_segment": bag_data["rc_messages_in_segment"],
        "odom_messages_in_segment": bag_data["odom_messages_in_segment"],
        "raw_vins_path_length_m": path_length(vins_ros),
        "dvl_path_length_m": path_length(reference_ros),
        "reference_path_length_m": path_length(reference_ros),
        "min_mean_corr": args.min_mean_corr,
        "max_rmse_ratio": args.max_rmse_ratio,
        "max_error_ratio": args.max_error_ratio,
        "reached_min_corr": best["mean_corr"] >= args.min_mean_corr,
        "controller": controller,
        "best": public_candidate(best),
        "top_candidates": [public_candidate(candidate) for candidate in ranked_candidates],
        "note": (
            "Reference is sampled from the UUV sim/real rosbag MP4 time section where /mavros/rc/override exists. "
            "Default odom reference is /odometry/filtered. Oracle mode publishes that odom path itself."
        ),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit VINS odometry to the UUV sim RC-override odom segment.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--oracle-csv", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--image-topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--rc-topic", default="/mavros/rc/override")
    parser.add_argument("--odom-topic", default="/odometry/filtered")
    parser.add_argument("--fit-modes", default="rigid,sim3,affine")
    parser.add_argument(
        "--max-step-grid",
        default="0.05,0.08,0.10,0.15,0.20,0.30,0.40,0.50,0.75,1.0,1.5,2.0,3.0,5.0,1000.0",
    )
    parser.add_argument("--min-mean-corr", type=float, default=0.70)
    parser.add_argument("--max-rmse-ratio", type=float, default=0.10)
    parser.add_argument("--max-error-ratio", type=float, default=0.80)
    parser.add_argument("--jump-factor", type=float, default=8.0)
    parser.add_argument("--rc-padding-sec", type=float, default=0.20)
    return parser.parse_args()


def read_odometry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_bag_reference(
    bag: Path,
    *,
    image_topic: str,
    rc_topic: str,
    odom_topic: str,
    times: np.ndarray,
    rc_padding_sec: float,
) -> dict[str, Any]:
    try:
        from mavros_msgs.msg import OverrideRCIn
        from nav_msgs.msg import Odometry
        from rclpy.serialization import deserialize_message
    except ImportError as exc:
        raise SystemExit("Run inside the ros2_h311 environment with mavros_msgs available.") from exc

    conn = sqlite3.connect(str(bag))
    try:
        topics = {name: topic_id for topic_id, name in conn.execute("select id, name from topics")}
        for topic in (image_topic, rc_topic, odom_topic):
            if topic not in topics:
                raise SystemExit(f"Missing topic in bag: {topic}")

        image_rows = conn.execute(
            "select timestamp from messages where topic_id = ? order by timestamp",
            (topics[image_topic],),
        ).fetchall()
        if not image_rows:
            raise SystemExit(f"No image messages for topic: {image_topic}")
        image_origin_db_ns = int(image_rows[0][0])
        video_start_db_ns = image_origin_db_ns + int(round(float(times[0]) * 1e9))
        video_end_db_ns = image_origin_db_ns + int(round(float(times[-1]) * 1e9))

        rc_rows = conn.execute(
            "select timestamp, data from messages where topic_id = ? and timestamp between ? and ? order by timestamp",
            (topics[rc_topic], video_start_db_ns, video_end_db_ns),
        ).fetchall()
        if not rc_rows:
            raise SystemExit(f"No RC override messages in MP4 time window for topic: {rc_topic}")
        # Deserialize once so missing message definitions fail early and the summary is not based on a wrong topic.
        _ = deserialize_message(rc_rows[0][1], OverrideRCIn)
        rc_start_sec = (int(rc_rows[0][0]) - image_origin_db_ns) / 1e9
        rc_end_sec = (int(rc_rows[-1][0]) - image_origin_db_ns) / 1e9
        segment_start_sec = max(float(times[0]), rc_start_sec - rc_padding_sec)
        segment_end_sec = min(float(times[-1]), rc_end_sec + rc_padding_sec)
        segment_start_db_ns = image_origin_db_ns + int(round(segment_start_sec * 1e9))
        segment_end_db_ns = image_origin_db_ns + int(round(segment_end_sec * 1e9))

        odom_samples: list[tuple[float, tuple[float, float, float]]] = []
        for timestamp_ns, data in conn.execute(
            "select timestamp, data from messages where topic_id = ? and timestamp between ? and ? order by timestamp",
            (topics[odom_topic], segment_start_db_ns, segment_end_db_ns),
        ):
            msg = deserialize_message(data, Odometry)
            p = msg.pose.pose.position
            odom_samples.append(((int(timestamp_ns) - image_origin_db_ns) / 1e9, (float(p.x), float(p.y), float(p.z))))
        if len(odom_samples) < 2:
            raise SystemExit(f"Too few odom samples in RC segment for topic: {odom_topic}")

        sample_times = np.array([item[0] for item in odom_samples], dtype=float)
        sample_positions = np.array([item[1] for item in odom_samples], dtype=float)
        reference_ros = np.empty((len(times), 3), dtype=float)
        for axis in range(3):
            reference_ros[:, axis] = np.interp(times, sample_times, sample_positions[:, axis])
        origin = np.array(
            [
                np.interp(segment_start_sec, sample_times, sample_positions[:, axis])
                for axis in range(3)
            ],
            dtype=float,
        )
        reference_ros -= origin
        return {
            "reference_ros": reference_ros,
            "segment_start_sec": segment_start_sec,
            "segment_end_sec": segment_end_sec,
            "rc_messages_in_segment": len(rc_rows),
            "odom_messages_in_segment": len(odom_samples),
        }
    finally:
        conn.close()


def csv_positions_to_ros(rows: list[dict[str, str]]) -> np.ndarray:
    csv_positions = np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows], dtype=float)
    ros_positions = np.empty_like(csv_positions)
    ros_positions[:, 0] = csv_positions[:, 2]
    ros_positions[:, 1] = -csv_positions[:, 0]
    ros_positions[:, 2] = -csv_positions[:, 1]
    return ros_positions


def ros_position_to_csv_storage(position: np.ndarray) -> tuple[float, float, float]:
    return (-float(position[1]), -float(position[2]), float(position[0]))


def write_reference_csv(
    path: Path,
    rows: list[dict[str, str]],
    positions_ros: np.ndarray,
    source_note: str,
) -> None:
    fieldnames = list(rows[0].keys())
    for name in ("reference_source", "reference_ros_x", "reference_ros_y", "reference_ros_z"):
        if name not in fieldnames:
            fieldnames.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row, position_ros in zip(rows, positions_ros):
            out = dict(row)
            x, y, z = ros_position_to_csv_storage(position_ros)
            out["x"] = f"{x:.9f}"
            out["y"] = f"{y:.9f}"
            out["z"] = f"{z:.9f}"
            out["qx"] = "0.000000000"
            out["qy"] = "0.000000000"
            out["qz"] = "0.000000000"
            out["qw"] = "1.000000000"
            out["pose_success"] = "1"
            out["scale_mode"] = "uuv_sim_rc_odom_reference"
            out["note"] = source_note
            out["reference_source"] = source_note
            out["reference_ros_x"] = f"{position_ros[0]:.9f}"
            out["reference_ros_y"] = f"{position_ros[1]:.9f}"
            out["reference_ros_z"] = f"{position_ros[2]:.9f}"
            writer.writerow(out)


if __name__ == "__main__":
    main()
