#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray


OPENCV_OPTICAL_TO_ROS = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)


@dataclass
class LoadedPath:
    path: RosPath
    csv_path: Path


class PathOverlayPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("vins_dvl_path_overlay_publisher")
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.frame_id = args.frame_id
        reject_dvl_derived_vins_csv(args.vins_csv)
        vins_time_window = (
            path_time_window(args.vins_csv, success_only=args.success_only_vins)
            if args.crop_reference_to_vins_time
            else None
        )
        vins_sample_times = (
            path_sample_times(args.vins_csv, success_only=args.success_only_vins)
            if args.sample_reference_at_vins_times
            else None
        )
        self.vins_path = load_path(
            args.vins_csv,
            self.frame_id,
            args.vins_coordinate_frame,
            args.zero_start,
            args.vins_orientation_source,
            args.vins_yaw_offset_deg,
            success_only=args.success_only_vins,
        )
        self.dvl_path = load_path(
            args.dvl_csv,
            self.frame_id,
            args.dvl_coordinate_frame,
            args.zero_start,
            args.dvl_orientation_source,
            args.dvl_yaw_offset_deg,
            success_only=False,
            time_window=vins_time_window,
            sample_times=vins_sample_times,
        )
        self.control_path = (
            load_path(
                args.control_csv,
                self.frame_id,
                args.control_coordinate_frame,
                args.zero_start,
                args.control_orientation_source,
                args.control_yaw_offset_deg,
                success_only=False,
                time_window=vins_time_window,
                sample_times=vins_sample_times,
            )
            if args.control_csv is not None
            else None
        )
        self.direction_arrow_stride = max(1, args.direction_arrow_stride)
        self.direction_arrow_length = max(0.01, args.direction_arrow_length)
        self.vins_pub = self.create_publisher(RosPath, args.vins_topic, qos)
        self.dvl_pub = self.create_publisher(RosPath, args.dvl_topic, qos)
        self.vins_direction_pub = self.create_publisher(MarkerArray, args.vins_direction_topic, qos)
        self.dvl_direction_pub = self.create_publisher(MarkerArray, args.dvl_direction_topic, qos)
        self.control_pub = (
            self.create_publisher(RosPath, args.control_topic, qos)
            if self.control_path is not None
            else None
        )
        self.control_direction_pub = (
            self.create_publisher(MarkerArray, args.control_direction_topic, qos)
            if self.control_path is not None
            else None
        )
        self.timer = self.create_timer(1.0 / max(args.rate, 0.1), self.publish_paths)

        self.get_logger().info(
            f"Loaded VINS path: {len(self.vins_path.path.poses)} poses from {args.vins_csv}"
        )
        self.get_logger().info(
            f"Loaded DVL path: {len(self.dvl_path.path.poses)} poses from {args.dvl_csv}"
        )
        if self.control_path is not None:
            self.get_logger().info(
                f"Loaded control path: {len(self.control_path.path.poses)} poses from {args.control_csv}"
            )
        self.get_logger().info(
            f"Publishing overlay paths to {args.vins_topic} and {args.dvl_topic}"
        )

    def publish_paths(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.vins_path.path.header.stamp = stamp
        self.dvl_path.path.header.stamp = stamp
        for pose in self.vins_path.path.poses:
            pose.header.stamp = stamp
        for pose in self.dvl_path.path.poses:
            pose.header.stamp = stamp
        self.vins_pub.publish(self.vins_path.path)
        self.dvl_pub.publish(self.dvl_path.path)
        self.vins_direction_pub.publish(
            make_direction_markers(
                self.vins_path.path,
                "vins_path_direction",
                (0.1, 0.67, 1.0, 0.9),
                stamp,
                self.direction_arrow_stride,
                self.direction_arrow_length,
            )
        )
        self.dvl_direction_pub.publish(
            make_direction_markers(
                self.dvl_path.path,
                "dvl_reference_direction",
                (1.0, 0.18, 0.18, 0.95),
                stamp,
                self.direction_arrow_stride,
                self.direction_arrow_length,
            )
        )
        if self.control_path is not None and self.control_pub is not None:
            self.control_path.path.header.stamp = stamp
            for pose in self.control_path.path.poses:
                pose.header.stamp = stamp
            self.control_pub.publish(self.control_path.path)
            if self.control_direction_pub is not None:
                self.control_direction_pub.publish(
                    make_direction_markers(
                        self.control_path.path,
                        "control_path_direction",
                        (0.28, 1.0, 0.47, 0.9),
                        stamp,
                        self.direction_arrow_stride,
                        self.direction_arrow_length,
                    )
                )


def load_path(
    csv_path: Path,
    frame_id: str,
    coordinate_frame: str,
    zero_start: bool,
    orientation_source: str,
    yaw_offset_deg: float = 0.0,
    success_only: bool = False,
    time_window: tuple[float, float] | None = None,
    sample_times: list[float] | None = None,
) -> LoadedPath:
    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows in CSV: {csv_path}")
    if success_only:
        rows = [row for row in rows if row.get("pose_success") in {"1", "true", "True", ""}]
        if not rows:
            raise RuntimeError(f"No successful pose rows in CSV: {csv_path}")
    if time_window is not None and sample_times is None:
        start_time, end_time = time_window
        rows = [
            row
            for row in rows
            if start_time <= float(row.get("timestamp_sec", start_time)) <= end_time
        ]
        if not rows:
            raise RuntimeError(f"No rows in {csv_path} inside time window {start_time:.3f}-{end_time:.3f}s")

    converted: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] = []
    if sample_times is not None:
        converted = interpolate_path_rows(rows, coordinate_frame, sample_times)
        rows = [{"timestamp_sec": f"{time_value:.9f}"} for time_value in sample_times]
    else:
        for row in rows:
            converted.append(row_pose(row, coordinate_frame))

    origin = None
    zeroed: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] = []
    for position, quaternion in converted:
        if origin is None:
            origin = position
        if zero_start and origin is not None:
            position = (
                position[0] - origin[0],
                position[1] - origin[1],
                position[2] - origin[2],
            )
        zeroed.append((position, quaternion))
    converted = zeroed

    yaw_offset_rad = math.radians(yaw_offset_deg)
    if abs(yaw_offset_rad) > 1e-12:
        converted = [
            apply_yaw_offset(position, quaternion, yaw_offset_rad)
            for position, quaternion in converted
        ]

    poses = []
    positions = [position for position, _ in converted]
    for index, (row, (position, csv_quaternion)) in enumerate(zip(rows, converted)):
        if should_use_tangent_orientation(row, orientation_source):
            quaternion = tangent_quaternion_for_index(positions, index)
        else:
            quaternion = csv_quaternion
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        poses.append(pose)

    msg = RosPath()
    msg.header.frame_id = frame_id
    msg.poses = poses
    return LoadedPath(path=msg, csv_path=csv_path)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def path_time_window(path: Path, success_only: bool = False) -> tuple[float, float]:
    times = path_sample_times(path, success_only=success_only)
    return min(times), max(times)


def path_sample_times(path: Path, success_only: bool = False) -> list[float]:
    rows = load_rows(path)
    if success_only:
        rows = [row for row in rows if row.get("pose_success") in {"1", "true", "True", ""}]
    times = [float(row["timestamp_sec"]) for row in rows if row.get("timestamp_sec") not in {None, ""}]
    if not times:
        raise RuntimeError(f"No timestamp_sec values available for time-window crop: {path}")
    return times


def interpolate_path_rows(
    rows: list[dict[str, str]],
    coordinate_frame: str,
    sample_times: list[float],
) -> list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    timed: list[tuple[float, tuple[float, float, float]]] = []
    for row in rows:
        if row.get("timestamp_sec") in {None, ""}:
            continue
        position, _ = row_pose(row, coordinate_frame)
        timed.append((float(row["timestamp_sec"]), position))
    if len(timed) < 2:
        raise RuntimeError("Need at least two timestamped rows to resample reference path")
    timed.sort(key=lambda item: item[0])
    times = [item[0] for item in timed]
    positions = [item[1] for item in timed]
    return [
        (interpolate_position(times, positions, sample_time), (0.0, 0.0, 0.0, 1.0))
        for sample_time in sample_times
    ]


def interpolate_position(
    times: list[float],
    positions: list[tuple[float, float, float]],
    sample_time: float,
) -> tuple[float, float, float]:
    if sample_time <= times[0]:
        return positions[0]
    if sample_time >= times[-1]:
        return positions[-1]
    for right_index in range(1, len(times)):
        if times[right_index] >= sample_time:
            left_index = right_index - 1
            span = times[right_index] - times[left_index]
            ratio = 0.0 if span <= 0.0 else (sample_time - times[left_index]) / span
            left = positions[left_index]
            right = positions[right_index]
            return (
                left[0] + (right[0] - left[0]) * ratio,
                left[1] + (right[1] - left[1]) * ratio,
                left[2] + (right[2] - left[2]) * ratio,
            )
    return positions[-1]


def reject_dvl_derived_vins_csv(path: Path) -> None:
    lowered = str(path).lower()
    blocked_name_tokens = (
        "dvl_fit",
        "dvl_anchor",
        "dvl_guided",
        "dvl_oracle",
        "oracle",
        "reference_only",
    )
    if any(token in lowered for token in blocked_name_tokens):
        raise RuntimeError(
            f"Refusing to publish DVL-derived CSV as VINS path: {path}. "
            "Use only a pure stereo/IMU VINS output for --vins-csv."
        )
    rows = load_rows(path)
    if not rows:
        return
    blocked_fields = {
        "dvl_anchor_source",
        "dvl_fit_source",
        "dvl_reference_csv",
        "dvl_reference_sha256",
    }
    if blocked_fields.intersection(rows[0]):
        raise RuntimeError(
            f"Refusing to publish CSV with DVL-derived fields as VINS path: {path}."
        )
    for row in rows[:10]:
        scale_mode = (row.get("scale_mode") or "").lower()
        note = (row.get("correction_note") or "").lower()
        if "dvl" in scale_mode or "dvl" in note:
            raise RuntimeError(
                f"Refusing to publish DVL-derived row metadata as VINS path: {path}."
            )


def row_pose(row: dict[str, str], coordinate_frame: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    position = (float(row["x"]), float(row["y"]), float(row["z"]))
    quaternion = (
        float(row.get("qx") or 0.0),
        float(row.get("qy") or 0.0),
        float(row.get("qz") or 0.0),
        float(row.get("qw") or 1.0),
    )
    if coordinate_frame in {"opencv", "raw"}:
        return position, normalize_quaternion(quaternion)
    if coordinate_frame == "flip-y":
        return (position[0], -position[1], position[2]), normalize_quaternion(quaternion)
    if coordinate_frame != "ros":
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")

    ros_position = (position[2], -position[0], -position[1])
    rotation = quaternion_to_matrix(quaternion)
    converted = matmul(matmul(OPENCV_OPTICAL_TO_ROS, rotation), transpose(OPENCV_OPTICAL_TO_ROS))
    return ros_position, matrix_to_quaternion(converted)


def apply_yaw_offset(
    position: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    yaw_offset_rad: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    rotation = yaw_rotation_matrix(yaw_offset_rad)
    rotated_position = matvec(rotation, position)
    rotated_quaternion = matrix_to_quaternion(matmul(rotation, quaternion_to_matrix(quaternion)))
    return rotated_position, rotated_quaternion


def yaw_rotation_matrix(yaw_rad: float) -> tuple[tuple[float, float, float], ...]:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    return (
        (cos_yaw, -sin_yaw, 0.0),
        (sin_yaw, cos_yaw, 0.0),
        (0.0, 0.0, 1.0),
    )


def should_use_tangent_orientation(row: dict[str, str], orientation_source: str) -> bool:
    if orientation_source == "tangent":
        return True
    if orientation_source == "csv":
        return False
    if orientation_source != "auto":
        raise ValueError(f"Unsupported orientation source: {orientation_source}")
    return not row_has_quaternion(row)


def row_has_quaternion(row: dict[str, str]) -> bool:
    return all(row.get(name) not in {None, ""} for name in ("qx", "qy", "qz", "qw"))


def tangent_quaternion_for_index(
    positions: list[tuple[float, float, float]],
    index: int,
) -> tuple[float, float, float, float]:
    if len(positions) < 2:
        return (0.0, 0.0, 0.0, 1.0)
    delta = nonzero_tangent_delta(positions, index)
    return direction_quaternion(delta)


def nonzero_tangent_delta(
    positions: list[tuple[float, float, float]],
    index: int,
) -> tuple[float, float, float]:
    for radius in range(1, len(positions)):
        left = max(0, index - radius)
        right = min(len(positions) - 1, index + radius)
        delta = sub(positions[right], positions[left])
        if vector_norm(delta) > 1e-9:
            return delta
    return (0.0, 0.0, 0.0)


def direction_quaternion(delta: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_axis = normalize_vector(delta)
    if x_axis is None:
        return (0.0, 0.0, 0.0, 1.0)
    up = (0.0, 0.0, 1.0)
    if abs(dot(x_axis, up)) > 0.95:
        up = (0.0, 1.0, 0.0)
    y_axis = normalize_vector(cross(up, x_axis))
    if y_axis is None:
        return (0.0, 0.0, 0.0, 1.0)
    z_axis = cross(x_axis, y_axis)
    rotation = (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )
    return matrix_to_quaternion(rotation)


def make_direction_markers(
    path: RosPath,
    namespace: str,
    color: tuple[float, float, float, float],
    stamp,
    stride: int,
    arrow_length: float,
) -> MarkerArray:
    markers = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)
    marker_id = 0
    for index, pose in enumerate(path.poses):
        if index % stride != 0 and index != len(path.poses) - 1:
            continue
        rotation = quaternion_to_matrix(
            (
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            )
        )
        forward = (rotation[0][0], rotation[1][0], rotation[2][0])
        start = (
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            float(pose.pose.position.z),
        )
        end = add(start, scale(forward, arrow_length))
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = path.header.frame_id
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [make_point(start), make_point(end)]
        marker.scale.x = 0.035
        marker.scale.y = 0.09
        marker.scale.z = 0.16
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.pose.orientation.w = 1.0
        markers.markers.append(marker)
        marker_id += 1
    return markers


def make_point(point: tuple[float, float, float]) -> Point:
    msg = Point()
    msg.x = float(point[0])
    msg.y = float(point[1])
    msg.z = float(point[2])
    return msg


def quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], ...]:
    qx, qy, qz, qw = normalize_quaternion(quaternion)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def matrix_to_quaternion(rotation: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float, float]:
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
    return normalize_quaternion((qx, qy, qz, qw))


def normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(value / norm for value in quaternion)


def normalize_vector(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = vector_norm(vector)
    if norm <= 1e-12:
        return None
    return tuple(value / norm for value in vector)


def vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def sub(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def add(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def scale(vector: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def matmul(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))


def transpose(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[row][col] for row in range(3)) for col in range(3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish VINS and DVL CSV paths together for RViz overlay.")
    parser.add_argument("--vins-csv", type=Path, default=Path("outputs/live/latest_odometry.csv"))
    parser.add_argument("--dvl-csv", type=Path, default=Path("outputs/live/latest_odometry_dvl_oracle.csv"))
    parser.add_argument("--control-csv", type=Path)
    parser.add_argument("--vins-topic", default="/vins_path_overlay")
    parser.add_argument("--dvl-topic", default="/dvl_reference_path")
    parser.add_argument("--control-topic", default="/control_best_path")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--vins-coordinate-frame", choices=("ros", "opencv", "raw", "flip-y"), default="ros")
    parser.add_argument("--dvl-coordinate-frame", choices=("ros", "opencv", "raw", "flip-y"), default="ros")
    parser.add_argument("--control-coordinate-frame", choices=("ros", "opencv", "raw", "flip-y"), default="ros")
    parser.add_argument("--vins-orientation-source", choices=("auto", "csv", "tangent"), default="auto")
    parser.add_argument("--dvl-orientation-source", choices=("auto", "csv", "tangent"), default="auto")
    parser.add_argument("--control-orientation-source", choices=("auto", "csv", "tangent"), default="auto")
    parser.add_argument(
        "--vins-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Rotate only the displayed VINS path around the RViz/map Z axis after zero-start alignment.",
    )
    parser.add_argument(
        "--dvl-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Rotate only the displayed DVL reference path. Leave at 0 when DVL is the fixed scoring reference.",
    )
    parser.add_argument(
        "--control-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Rotate only the optional control path around the RViz/map Z axis.",
    )
    parser.add_argument("--vins-direction-topic", default="/vins_path_direction_markers")
    parser.add_argument("--dvl-direction-topic", default="/dvl_reference_direction_markers")
    parser.add_argument("--control-direction-topic", default="/control_path_direction_markers")
    parser.add_argument("--direction-arrow-stride", type=int, default=10)
    parser.add_argument("--direction-arrow-length", type=float, default=0.35)
    parser.add_argument(
        "--success-only-vins",
        action="store_true",
        help="Draw only successful VINS poses, leaving the fixed DVL reference unchanged.",
    )
    parser.add_argument(
        "--crop-reference-to-vins-time",
        action="store_true",
        help="Crop DVL/control reference paths to the VINS timestamp window for fair RViz overlay diagnostics.",
    )
    parser.add_argument(
        "--sample-reference-at-vins-times",
        action="store_true",
        help="Resample DVL/control references at successful VINS timestamps for one-to-one RViz overlay diagnostics.",
    )
    parser.add_argument("--no-zero-start", dest="zero_start", action="store_false")
    parser.set_defaults(zero_start=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = PathOverlayPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
