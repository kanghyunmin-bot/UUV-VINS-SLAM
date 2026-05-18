#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any


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
DEFAULT_REFERENCE = PROJECT_ROOT / "outputs" / "evaluation" / "rosbag_odometry_filtered_reference.csv"
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "live" / "latest_odometry.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "live" / "latest_odometry_aqua_corrected.csv"


def main() -> None:
    args = parse_args()
    rows = read_odometry(args.input_csv)
    if len(rows) < 2:
        raise SystemExit(f"Need at least two odometry rows: {args.input_csv}")

    target_length, source_note = resolve_target_length(args, rows)
    corrected_positions, raw_length, clamped_length, clamped_steps = reconstruct_positions(
        rows,
        target_length=target_length,
        max_step_m=args.max_step_m,
        jump_factor=args.jump_factor,
    )
    write_corrected(args.output_csv, rows, corrected_positions, source_note)
    result = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "source": source_note,
        "rows": len(rows),
        "raw_path_length_m": raw_length,
        "jump_clamped_path_length_m": clamped_length,
        "target_path_length_m": target_length,
        "scale": target_length / clamped_length if clamped_length > 1e-9 else 1.0,
        "clamped_steps": clamped_steps,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply an AQUA-SLAM-inspired metric correction layer to VINS odometry. "
            "This is not a replacement for tightly-coupled DVL factors; it is a practical "
            "post-process that uses DVL/reference motion length to correct visual path scale."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source",
        choices=("reference", "dvl", "target"),
        default="reference",
        help="Metric source for path length correction.",
    )
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--left-topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--dvl-topic", default="/dvl/twist")
    parser.add_argument("--target-path-m", type=float, default=100.0)
    parser.add_argument(
        "--max-step-m",
        type=float,
        default=2.0,
        help="Clamp single-frame visual jumps above this value when reconstructing the corrected path.",
    )
    parser.add_argument(
        "--jump-factor",
        type=float,
        default=8.0,
        help="Also clamp steps larger than median_step * jump_factor.",
    )
    return parser.parse_args()


def read_odometry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def resolve_target_length(args: argparse.Namespace, rows: list[dict[str, str]]) -> tuple[float, str]:
    start_t = float(rows[0]["timestamp_sec"])
    end_t = float(rows[-1]["timestamp_sec"])
    if args.source == "target":
        return float(args.target_path_m), f"manual_target:{args.target_path_m:.3f}m"
    if args.source == "reference":
        length = reference_path_length(args.reference_csv, start_t, end_t)
        return length, f"odometry_filtered_reference:{length:.3f}m"
    length = dvl_path_length(args.bag, args.left_topic, args.dvl_topic, start_t, end_t)
    return length, f"dvl_twist_integrated:{length:.3f}m"


def reconstruct_positions(
    rows: list[dict[str, str]],
    *,
    target_length: float,
    max_step_m: float,
    jump_factor: float,
) -> tuple[list[tuple[float, float, float]], float, float, int]:
    positions = [(float(row["x"]), float(row["y"]), float(row["z"])) for row in rows]
    deltas = [
        (
            positions[i][0] - positions[i - 1][0],
            positions[i][1] - positions[i - 1][1],
            positions[i][2] - positions[i - 1][2],
        )
        for i in range(1, len(positions))
    ]
    lengths = [norm(delta) for delta in deltas]
    raw_length = sum(lengths)
    positive_lengths = [value for value in lengths if value > 1e-9]
    median_step = median(positive_lengths) if positive_lengths else 0.0
    clamp_limit = max_step_m
    if median_step > 0.0:
        clamp_limit = max(max_step_m, median_step * jump_factor)

    clamped_deltas = []
    clamped_steps = 0
    for delta, length in zip(deltas, lengths):
        if length > clamp_limit > 0.0:
            scale = clamp_limit / length
            clamped_deltas.append((delta[0] * scale, delta[1] * scale, delta[2] * scale))
            clamped_steps += 1
        else:
            clamped_deltas.append(delta)

    clamped_length = sum(norm(delta) for delta in clamped_deltas)
    metric_scale = target_length / clamped_length if clamped_length > 1e-9 else 1.0
    corrected = [positions[0]]
    for delta in clamped_deltas:
        prev = corrected[-1]
        corrected.append(
            (
                prev[0] + delta[0] * metric_scale,
                prev[1] + delta[1] * metric_scale,
                prev[2] + delta[2] * metric_scale,
            )
        )
    return corrected, raw_length, clamped_length, clamped_steps


def write_corrected(
    path: Path,
    rows: list[dict[str, str]],
    positions: list[tuple[float, float, float]],
    source_note: str,
) -> None:
    fieldnames = list(rows[0].keys())
    for name in ("correction_source", "correction_note"):
        if name not in fieldnames:
            fieldnames.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row, position in zip(rows, positions):
            output = dict(row)
            output["x"] = f"{position[0]:.9f}"
            output["y"] = f"{position[1]:.9f}"
            output["z"] = f"{position[2]:.9f}"
            output["scale_mode"] = "aqua_metric_corrected"
            output["correction_source"] = source_note
            output["correction_note"] = "AQUA-style DVL/reference path-length correction with jump clamping"
            writer.writerow(output)


def reference_path_length(path: Path, start_t: float, end_t: float) -> float:
    if not path.exists():
        raise SystemExit(f"Missing reference CSV: {path}")
    points = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            t = float(row["timestamp_sec_from_infra1"])
            if start_t <= t <= end_t:
                points.append((float(row["x"]), float(row["y"]), float(row["z"])))
    if len(points) < 2:
        raise SystemExit(f"No reference points in time range {start_t:.3f}..{end_t:.3f}s")
    return sum(distance(prev, cur) for prev, cur in zip(points, points[1:]))


def dvl_path_length(bag: Path, left_topic: str, dvl_topic: str, start_t: float, end_t: float) -> float:
    try:
        from geometry_msgs.msg import TwistWithCovarianceStamped
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise SystemExit(
            "DVL source requires the ROS2 Python environment. Run with ros2_h311 conda, "
            "or use --source reference."
        ) from exc

    conn = sqlite3.connect(str(bag))
    try:
        topic_ids = {name: topic_id for topic_id, name in conn.execute("select id, name from topics")}
        if left_topic not in topic_ids:
            raise SystemExit(f"Missing image topic in bag: {left_topic}")
        if dvl_topic not in topic_ids:
            raise SystemExit(f"Missing DVL topic in bag: {dvl_topic}")

        first_image_row = conn.execute(
            "select data from messages where topic_id = ? order by timestamp limit 1",
            (topic_ids[left_topic],),
        ).fetchone()
        if first_image_row is None:
            raise SystemExit(f"No image messages for topic: {left_topic}")
        first_image = deserialize_message(first_image_row[0], Image)
        origin_ns = stamp_to_ns(first_image.header.stamp)

        samples: list[tuple[float, float]] = []
        for (data,) in conn.execute(
            "select data from messages where topic_id = ? order by timestamp",
            (topic_ids[dvl_topic],),
        ):
            msg = deserialize_message(data, TwistWithCovarianceStamped)
            t = (stamp_to_ns(msg.header.stamp) - origin_ns) / 1e9
            if start_t <= t <= end_t:
                v = msg.twist.twist.linear
                samples.append((t, math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)))
        if len(samples) < 2:
            raise SystemExit(f"No DVL twist samples in time range {start_t:.3f}..{end_t:.3f}s")
        total = 0.0
        for (prev_t, prev_speed), (cur_t, cur_speed) in zip(samples, samples[1:]):
            dt = max(0.0, cur_t - prev_t)
            total += 0.5 * (prev_speed + cur_speed) * dt
        return total
    finally:
        conn.close()


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return norm((right[0] - left[0], right[1] - left[1], right[2] - left[2]))


if __name__ == "__main__":
    main()
