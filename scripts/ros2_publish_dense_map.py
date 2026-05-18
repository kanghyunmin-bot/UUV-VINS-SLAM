#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker


class DenseMapPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("dense_stereo_map_publisher")
        self.args = args
        self.last_mtime_ns = 0
        self.cloud = PointCloud2()
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.publisher = self.create_publisher(PointCloud2, args.topic, qos)
        self.rectangle_path_pub = self.create_publisher(RosPath, args.rectangle_path_topic, qos)
        self.rectangle_marker_pub = self.create_publisher(Marker, args.rectangle_marker_topic, qos)
        self.rectangle_path = RosPath()
        self.rectangle_marker = Marker()
        self.reload()
        self.timer = self.create_timer(1.0 / max(args.rate, 0.1), self.publish)

    def reload(self) -> None:
        if not self.args.csv.exists():
            self.get_logger().warn(f"Waiting for dense map CSV: {self.args.csv}")
            return
        mtime_ns = self.args.csv.stat().st_mtime_ns
        if not self.args.watch and self.last_mtime_ns == mtime_ns:
            return
        if self.args.watch and self.last_mtime_ns == mtime_ns:
            return
        points = load_points(
            self.args.csv,
            self.args.coordinate_frame,
            self.args.max_points,
            math.radians(self.args.yaw_offset_deg),
        )
        self.cloud = make_cloud(points, self.args.frame_id)
        self.rectangle_path, self.rectangle_marker = load_rectangle(
            self.args.rectangle_csv,
            self.args.rectangle_coordinate_frame,
            self.args.frame_id,
            math.radians(self.args.rectangle_yaw_offset_deg),
        )
        self.last_mtime_ns = mtime_ns
        self.get_logger().info(f"Loaded {len(points)} dense map points from {self.args.csv}")

    def publish(self) -> None:
        if self.args.watch:
            self.reload()
        stamp = self.get_clock().now().to_msg()
        self.cloud.header.stamp = stamp
        self.publisher.publish(self.cloud)
        if self.rectangle_path.poses:
            self.rectangle_path.header.stamp = stamp
            for pose in self.rectangle_path.poses:
                pose.header.stamp = stamp
            self.rectangle_path_pub.publish(self.rectangle_path)
        if self.rectangle_marker.points:
            self.rectangle_marker.header.stamp = stamp
            self.rectangle_marker_pub.publish(self.rectangle_marker)


def load_points(
    path: Path,
    coordinate_frame: str,
    max_points: int,
    yaw_offset_rad: float = 0.0,
) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            point = convert_point(
                (float(row["x"]), float(row["y"]), float(row["z"])),
                coordinate_frame,
            )
            point = apply_yaw(point, yaw_offset_rad)
            points.append(point)
    if max_points > 0 and len(points) > max_points:
        stride = max(1, len(points) // max_points)
        points = points[::stride][:max_points]
    return points


def load_rectangle(
    path: Path | None,
    coordinate_frame: str,
    frame_id: str,
    yaw_offset_rad: float = 0.0,
) -> tuple[RosPath, Marker]:
    path_msg = RosPath()
    path_msg.header.frame_id = frame_id
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = "fitted_pool_rectangle"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.035
    marker.color.r = 1.0
    marker.color.g = 0.2
    marker.color.b = 0.9
    marker.color.a = 0.95
    marker.pose.orientation.w = 1.0
    if path is None or not path.exists():
        return path_msg, marker
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            point = convert_point(
                (float(row["x"]), float(row["y"]), float(row["z"])),
                coordinate_frame,
            )
            point = apply_yaw(point, yaw_offset_rad)
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            marker.points.append(Point(x=float(point[0]), y=float(point[1]), z=float(point[2])))
    return path_msg, marker


def convert_point(point: tuple[float, float, float], coordinate_frame: str) -> tuple[float, float, float]:
    x, y, z = point
    if coordinate_frame in {"opencv", "raw"}:
        return (x, y, z)
    if coordinate_frame == "flip-y":
        return (x, -y, z)
    if coordinate_frame != "ros":
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    return (z, -x, -y)


def apply_yaw(point: tuple[float, float, float], yaw_offset_rad: float) -> tuple[float, float, float]:
    if abs(yaw_offset_rad) <= 1e-12:
        return point
    x, y, z = point
    cos_yaw = math.cos(yaw_offset_rad)
    sin_yaw = math.sin(yaw_offset_rad)
    return (cos_yaw * x - sin_yaw * y, sin_yaw * x + cos_yaw * y, z)


def make_cloud(points: list[tuple[float, float, float]], frame_id: str) -> PointCloud2:
    cloud = PointCloud2()
    cloud.header.frame_id = frame_id
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 12
    cloud.row_step = cloud.point_step * cloud.width
    cloud.is_dense = True
    cloud.data = b"".join(struct.pack("<fff", *point) for point in points)
    return cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a dense stereo map CSV as a static RViz PointCloud2.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--topic", default="/dense_stereo_map")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--coordinate-frame", choices=("ros", "opencv", "raw", "flip-y"), default="ros")
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--rectangle-csv", type=Path)
    parser.add_argument("--rectangle-coordinate-frame", choices=("ros", "opencv", "raw", "flip-y"), default="raw")
    parser.add_argument("--rectangle-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--rectangle-path-topic", default="/pool_rectangle_path")
    parser.add_argument("--rectangle-marker-topic", default="/pool_rectangle_marker")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--watch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = DenseMapPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
