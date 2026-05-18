#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


class StaticPathPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("static_sim_odom_reference_publisher")
        self.frame_id = args.frame_id
        self.path_msg, self.marker_msg = self._load_messages(args)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.path_pub = self.create_publisher(RosPath, args.path_topic, qos)
        self.marker_pub = self.create_publisher(Marker, args.marker_topic, qos)
        self.timer = self.create_timer(1.0 / max(args.rate, 0.1), self.publish)
        self.get_logger().info(
            f"Publishing fixed red sim odom reference: {len(self.path_msg.poses)} points "
            f"to {args.path_topic} and {args.marker_topic}"
        )

    def publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = stamp
        for pose in self.path_msg.poses:
            pose.header.stamp = stamp
        self.marker_msg.header.stamp = stamp
        self.path_pub.publish(self.path_msg)
        self.marker_pub.publish(self.marker_msg)

    def _load_messages(self, args: argparse.Namespace) -> tuple[RosPath, Marker]:
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

        for row in rows:
            x, y, z = csv_to_ros_position(row, args.coordinate_frame)
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
        return path, marker


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def csv_to_ros_position(row: dict[str, str], coordinate_frame: str) -> tuple[float, float, float]:
    x_csv = float(row["x"])
    y_csv = float(row["y"])
    z_csv = float(row["z"])
    if coordinate_frame in {"opencv", "raw"}:
        return (x_csv, y_csv, z_csv)
    if coordinate_frame == "flip-y":
        return (x_csv, -y_csv, z_csv)
    if coordinate_frame != "ros":
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    # CSV storage follows the same OpenCV optical convention as ros2_publish_odometry.py.
    return (z_csv, -x_csv, -y_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a fixed CSV path as a red RViz reference line.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--path-topic", default="/sim_odom_reference_path")
    parser.add_argument("--marker-topic", default="/sim_odom_reference_marker")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--line-width", type=float, default=0.08)
    parser.add_argument(
        "--coordinate-frame",
        choices=("ros", "opencv", "raw", "flip-y"),
        default="ros",
        help=(
            "ros: convert CSV OpenCV optical coordinates to RViz/map coordinates. "
            "opencv/raw: publish CSV coordinates directly with no axis conversion. "
            "flip-y: publish direct coordinates with only Y sign inverted."
        ),
    )
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
