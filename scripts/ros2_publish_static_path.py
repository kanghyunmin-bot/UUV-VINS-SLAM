#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import math
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class StaticPathPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("static_path_reference_publisher")
        self.args = args
        self.frame_id = args.frame_id
        self.path_msg, self.marker_msg, self.direction_markers, self.points, self.relative_times = self._load_messages(args)
        self._sent_direction_marker_clear = False
        self.wall_start = time.monotonic()

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.path_pub = self.create_publisher(RosPath, args.path_topic, qos)
        self.marker_pub = self.create_publisher(Marker, args.marker_topic, qos)
        self.direction_pub = self.create_publisher(MarkerArray, args.direction_marker_topic, qos)
        self.timer = self.create_timer(1.0 / max(args.rate, 0.1), self.publish)
        self.get_logger().info(
            f"Publishing {'progressive' if args.progressive else 'fixed'} path reference: {len(self.path_msg.poses)} points "
            f"to {args.path_topic}, {args.marker_topic}, and {args.direction_marker_topic}"
        )
        if args.publish_empty_on_start:
            self.publish_empty()

    def publish_empty(self) -> None:
        stamp = self.get_clock().now().to_msg()
        path = RosPath()
        path.header.frame_id = self.frame_id
        path.header.stamp = stamp
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "sim_odom_reference"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.args.line_width
        marker.color = ColorRGBA(
            r=self.args.red,
            g=self.args.green,
            b=self.args.blue,
            a=self.args.alpha,
        )
        clear_marker = Marker()
        clear_marker.header.frame_id = self.frame_id
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        clear_array = MarkerArray()
        clear_array.markers.append(clear_marker)
        self.path_pub.publish(path)
        self.marker_pub.publish(marker)
        self.direction_pub.publish(clear_array)

    def publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        path_msg = self.path_msg
        marker_msg = self.marker_msg
        direction_markers = self.direction_markers
        if self.args.progressive:
            path_msg, marker_msg, direction_markers = self._progressive_messages(stamp)
        else:
            path_msg.header.stamp = stamp
            for pose in path_msg.poses:
                pose.header.stamp = stamp
            marker_msg.header.stamp = stamp
            for marker in direction_markers.markers:
                marker.header.stamp = stamp
        self.path_pub.publish(path_msg)
        self.marker_pub.publish(marker_msg)
        if self.args.progressive or not self._sent_direction_marker_clear:
            clear_marker = Marker()
            clear_marker.header.frame_id = self.frame_id
            clear_marker.header.stamp = stamp
            clear_marker.action = Marker.DELETEALL
            clear_array = MarkerArray()
            clear_array.markers.append(clear_marker)
            self.direction_pub.publish(clear_array)
            self._sent_direction_marker_clear = True
        self.direction_pub.publish(direction_markers)

    def _progressive_messages(self, stamp) -> tuple[RosPath, Marker, MarkerArray]:
        elapsed = (time.monotonic() - self.wall_start) * max(self.args.playback_rate, 1e-6)
        if self.args.max_duration_sec > 0.0:
            elapsed = min(elapsed, self.args.max_duration_sec)
        count = bisect.bisect_right(self.relative_times, elapsed)
        if count <= 0 and self.args.publish_empty_on_start:
            points: list[Point] = []
        else:
            count = max(1, min(count, len(self.points)))
            points = self.points[:count]

        path = RosPath()
        path.header.frame_id = self.frame_id
        path.header.stamp = stamp
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "sim_odom_reference"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.args.line_width
        marker.color = ColorRGBA(
            r=self.args.red,
            g=self.args.green,
            b=self.args.blue,
            a=self.args.alpha,
        )

        for point in points:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = stamp
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = point.z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

            marker_point = Point()
            marker_point.x = point.x
            marker_point.y = point.y
            marker_point.z = point.z
            marker.points.append(marker_point)
        return path, marker, make_direction_markers(points, self.args)

    def _load_messages(self, args: argparse.Namespace) -> tuple[RosPath, Marker, MarkerArray, list[Point], list[float]]:
        rows = load_rows(args.csv)
        if not rows:
            raise RuntimeError(f"No rows in CSV: {args.csv}")

        path = RosPath()
        path.header.frame_id = self.frame_id
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.ns = "sim_odom_reference"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = args.line_width
        marker.color = ColorRGBA(r=args.red, g=args.green, b=args.blue, a=args.alpha)

        source_points = [csv_to_ros_position(row, args.coordinate_frame) for row in rows]
        alignment = compute_static_alignment(args, rows, source_points)
        first_time = row_time(rows[0]) if args.reset_time_zero else 0.0

        points: list[Point] = []
        relative_times: list[float] = []
        for x, y, z in source_points:
            x, y, z = apply_static_alignment(x, y, z, alignment)
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

            point = Point()
            point.x = x
            point.y = y
            point.z = z
            marker.points.append(point)
            points.append(point)
        for row in rows:
            relative_times.append(max(0.0, row_time(row) - first_time))
        return path, marker, make_direction_markers(points, args), points, relative_times


def make_direction_markers(points: list[Point], args: argparse.Namespace) -> MarkerArray:
    array = MarkerArray()
    if len(points) < 2:
        return array
    stride = max(1, args.direction_arrow_stride)
    marker_id = 0
    for index in range(0, len(points) - 1, stride):
        start = points[index]
        end_source = points[min(index + stride, len(points) - 1)]
        dx = end_source.x - start.x
        dy = end_source.y - start.y
        dz = end_source.z - start.z
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length <= 1e-9:
            continue
        scale = args.direction_arrow_length / length
        end = Point()
        end.x = start.x + dx * scale
        end.y = start.y + dy * scale
        end.z = start.z + dz * scale

        marker = Marker()
        marker.header.frame_id = args.frame_id
        marker.ns = "path_reference_direction"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.points = [start, end]
        marker.scale.x = args.direction_arrow_shaft_diameter
        marker.scale.y = args.direction_arrow_head_diameter
        marker.scale.z = args.direction_arrow_head_length
        marker.color = ColorRGBA(r=args.red, g=args.green, b=args.blue, a=args.alpha)
        array.markers.append(marker)
        marker_id += 1
    return array


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def csv_to_ros_position(row: dict[str, str], coordinate_frame: str) -> tuple[float, float, float]:
    x_csv = float(row["x"])
    y_csv = float(row["y"])
    z_csv = float(row["z"])
    if coordinate_frame in {"opencv", "raw", "ned"}:
        return (x_csv, y_csv, z_csv)
    if coordinate_frame == "flip-y":
        return (x_csv, -y_csv, z_csv)
    if coordinate_frame != "ros":
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    # CSV storage follows the same OpenCV optical convention as ros2_publish_odometry.py.
    return (z_csv, -x_csv, -y_csv)


def row_time(row: dict[str, str]) -> float:
    return float(row.get("timestamp_sec", "0.0"))


def interpolate_points(rows: list[dict[str, str]], coordinate_frame: str, times: list[float]) -> list[tuple[float, float, float]]:
    if not rows:
        return []
    source = [(row_time(row), csv_to_ros_position(row, coordinate_frame)) for row in rows]
    source.sort(key=lambda item: item[0])
    out: list[tuple[float, float, float]] = []
    j = 0
    for t in times:
        while j + 1 < len(source) and source[j + 1][0] < t:
            j += 1
        if t <= source[0][0]:
            out.append(source[0][1])
        elif t >= source[-1][0]:
            out.append(source[-1][1])
        else:
            t0, p0 = source[j]
            t1, p1 = source[j + 1]
            ratio = 0.0 if abs(t1 - t0) < 1e-9 else (t - t0) / (t1 - t0)
            out.append(tuple(p0[k] * (1.0 - ratio) + p1[k] * ratio for k in range(3)))
    return out


def compute_static_alignment(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    source_points: list[tuple[float, float, float]],
) -> tuple[float, float, float, float, float] | None:
    if args.align_mode == "none" or not args.align_target_csv:
        return None
    target_rows = load_rows(args.align_target_csv)
    if len(rows) < 3 or len(target_rows) < 3:
        return None
    times = [row_time(row) for row in rows]
    target_points = interpolate_points(target_rows, args.align_target_coordinate_frame, times)
    pairs = [
        (src, tgt)
        for src, tgt, t in zip(source_points, target_points, times)
        if row_time(target_rows[0]) <= t <= row_time(target_rows[-1])
    ]
    if len(pairs) < 3:
        return None

    sx = sum(p[0][0] for p in pairs) / len(pairs)
    sy = sum(p[0][1] for p in pairs) / len(pairs)
    sz = sum(p[0][2] for p in pairs) / len(pairs)
    tx = sum(p[1][0] for p in pairs) / len(pairs)
    ty = sum(p[1][1] for p in pairs) / len(pairs)
    tz = sum(p[1][2] for p in pairs) / len(pairs)

    dot = 0.0
    cross = 0.0
    source_energy = 0.0
    for src, tgt in pairs:
        ax = src[0] - sx
        ay = src[1] - sy
        bx = tgt[0] - tx
        by = tgt[1] - ty
        dot += ax * bx + ay * by
        cross += ax * by - ay * bx
        source_energy += ax * ax + ay * ay
    yaw = math.atan2(cross, dot)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    scale = 1.0
    if args.align_mode == "similarity" and source_energy > 1e-9:
        numerator = 0.0
        for src, tgt in pairs:
            ax = src[0] - sx
            ay = src[1] - sy
            bx = tgt[0] - tx
            by = tgt[1] - ty
            rx = cos_yaw * ax - sin_yaw * ay
            ry = sin_yaw * ax + cos_yaw * ay
            numerator += rx * bx + ry * by
        scale = numerator / source_energy
    offset_x = tx - scale * (cos_yaw * sx - sin_yaw * sy)
    offset_y = ty - scale * (sin_yaw * sx + cos_yaw * sy)
    offset_z = tz - scale * sz
    rmse = 0.0
    for src, tgt in pairs:
        out_x = scale * (cos_yaw * src[0] - sin_yaw * src[1]) + offset_x
        out_y = scale * (sin_yaw * src[0] + cos_yaw * src[1]) + offset_y
        rmse += (out_x - tgt[0]) ** 2 + (out_y - tgt[1]) ** 2
    rmse = math.sqrt(rmse / len(pairs))
    print(
        f"static path aligned to {args.align_target_csv}: mode={args.align_mode} "
        f"yaw_deg={math.degrees(yaw):.2f} scale={scale:.4f} rmse_xy={rmse:.3f}m",
        flush=True,
    )
    return yaw, scale, offset_x, offset_y, offset_z


def apply_static_alignment(
    x: float,
    y: float,
    z: float,
    alignment: tuple[float, float, float, float, float] | None,
) -> tuple[float, float, float]:
    if alignment is None:
        return x, y, z
    yaw, scale, offset_x, offset_y, offset_z = alignment
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        scale * (cos_yaw * x - sin_yaw * y) + offset_x,
        scale * (sin_yaw * x + cos_yaw * y) + offset_y,
        scale * z + offset_z,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a fixed CSV path as a red RViz reference line.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--path-topic", default="/sim_odom_reference_path")
    parser.add_argument("--marker-topic", default="/sim_odom_reference_marker")
    parser.add_argument("--direction-marker-topic", default="/dvl_reference_direction_markers")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--progressive", action="store_true")
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float, default=0.0)
    parser.add_argument("--publish-empty-on-start", action="store_true")
    parser.add_argument("--reset-time-zero", action="store_true")
    parser.add_argument("--line-width", type=float, default=0.08)
    parser.add_argument("--direction-arrow-stride", type=int, default=16)
    parser.add_argument("--direction-arrow-length", type=float, default=0.32)
    parser.add_argument("--direction-arrow-shaft-diameter", type=float, default=0.045)
    parser.add_argument("--direction-arrow-head-diameter", type=float, default=0.14)
    parser.add_argument("--direction-arrow-head-length", type=float, default=0.16)
    parser.add_argument(
        "--coordinate-frame",
        choices=("ros", "opencv", "raw", "ned", "flip-y"),
        default="ros",
        help=(
            "ros: convert CSV OpenCV optical coordinates to RViz/map coordinates. "
            "opencv/raw/ned: publish CSV coordinates directly with no axis conversion. "
            "ned means the RViz world numeric axes are DVL NED axes. "
            "flip-y: publish direct coordinates with only Y sign inverted."
        ),
    )
    parser.add_argument("--align-target-csv", type=Path, default=None)
    parser.add_argument(
        "--align-target-coordinate-frame",
        choices=("ros", "opencv", "raw", "ned", "flip-y"),
        default="raw",
    )
    parser.add_argument("--align-mode", choices=("none", "rigid", "similarity"), default="none")
    parser.add_argument("--red", type=float, default=1.0)
    parser.add_argument("--green", type=float, default=0.0)
    parser.add_argument("--blue", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = StaticPathPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
