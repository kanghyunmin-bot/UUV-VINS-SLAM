#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


FIELDNAMES = [
    "frame_index",
    "timestamp_sec",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "pose_success",
    "pose_inliers",
    "pose_inlier_ratio",
    "scale_mode",
    "note",
]


def main() -> None:
    args = parse_args()
    if not args.vio_txt.exists() and args.vio_txt.name == "vio.txt":
        csv_fallback = args.vio_txt.with_name("vio.csv")
        if csv_fallback.exists():
            args.vio_txt = csv_fallback
    if not args.vio_txt.exists():
        raise SystemExit(
            f"Missing VINS-Fusion output: {args.vio_txt}\n"
            "Run VINS-Fusion first with scripts/vins_run_stereo_kitti.sh on Ubuntu ROS1 "
            "or scripts/vins_docker_run_stereo_kitti.sh after building the Docker image."
        )
    if not args.times_txt.exists():
        raise SystemExit(f"Missing exported timestamps: {args.times_txt}")
    poses = read_vio(args.vio_txt)
    times = read_times(args.times_txt)
    count = min(len(poses), len(times))
    if count == 0:
        raise SystemExit("No VINS poses could be converted.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index in range(count):
            pose = poses[index]
            translation = pose["translation"]
            qx, qy, qz, qw = pose["quaternion"]
            timestamp_sec = pose["timestamp_sec"]
            if timestamp_sec is None:
                timestamp_sec = times[index]
            writer.writerow(
                {
                    "frame_index": index,
                    "timestamp_sec": f"{timestamp_sec:.9f}",
                    "x": f"{translation[0]:.9f}",
                    "y": f"{translation[1]:.9f}",
                    "z": f"{translation[2]:.9f}",
                    "qx": f"{qx:.9f}",
                    "qy": f"{qy:.9f}",
                    "qz": f"{qz:.9f}",
                    "qw": f"{qw:.9f}",
                    "pose_success": 1,
                    "pose_inliers": 0,
                    "pose_inlier_ratio": "0.000000",
                    "scale_mode": args.scale_mode,
                    "note": pose["note"],
                }
            )
    print(args.output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VINS-Fusion vio.txt output to this project's odometry CSV.")
    parser.add_argument(
        "--vio-txt",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "vins_output" / "vio.txt",
    )
    parser.add_argument(
        "--times-txt",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "underwater_stereo_kitti" / "times.txt",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "vins_odometry.csv",
    )
    parser.add_argument("--scale-mode", default="vins_fusion_stereo")
    return parser.parse_args()


def read_vio(path: Path):
    poses = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            values = parse_float_values(text)
            if len(values) == 12:
                rotation = (
                    (values[0], values[1], values[2]),
                    (values[4], values[5], values[6]),
                    (values[8], values[9], values[10]),
                )
                translation = (values[3], values[7], values[11])
                poses.append(
                    {
                        "timestamp_sec": None,
                        "translation": translation,
                        "quaternion": matrix_to_quaternion(rotation),
                        "note": "vins_fusion_kitti_odom_test",
                    }
                )
            elif len(values) >= 8:
                timestamp_raw = values[0]
                timestamp_sec = timestamp_raw / 1e9
                translation = (values[1], values[2], values[3])
                qw, qx, qy, qz = values[4], values[5], values[6], values[7]
                poses.append(
                    {
                        "timestamp_sec": timestamp_sec,
                        "translation": translation,
                        "quaternion": normalize((qx, qy, qz, qw)),
                        "note": "vins_fusion_vins_node",
                    }
                )
    return poses


def parse_float_values(line: str) -> list[float]:
    if "," in line:
        parts = [part for part in line.replace(",", " ").split() if part]
    else:
        parts = line.split()
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            return []
    return values


def read_times(path: Path) -> list[float]:
    return [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def matrix_to_quaternion(rotation) -> tuple[float, float, float, float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2][1] - rotation[1][2]) / scale
        qy = (rotation[0][2] - rotation[2][0]) / scale
        qz = (rotation[1][0] - rotation[0][1]) / scale
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        qw = (rotation[2][1] - rotation[1][2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0][1] + rotation[1][0]) / scale
        qz = (rotation[0][2] + rotation[2][0]) / scale
    elif rotation[1][1] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        qw = (rotation[0][2] - rotation[2][0]) / scale
        qx = (rotation[0][1] + rotation[1][0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1][2] + rotation[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        qw = (rotation[1][0] - rotation[0][1]) / scale
        qx = (rotation[0][2] + rotation[2][0]) / scale
        qy = (rotation[1][2] + rotation[2][1]) / scale
        qz = 0.25 * scale
    return normalize((qx, qy, qz, qw))


def normalize(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(value / norm for value in quaternion)


if __name__ == "__main__":
    main()
