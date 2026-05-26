#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node


class OdomPathPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("odom_path_publisher")
        self.min_step_m = args.min_step_m
        self.max_poses = args.max_poses
        self.frame_id = args.frame_id
        self.path = Path()
        self.path.header.frame_id = self.frame_id
        self.last_xyz: tuple[float, float, float] | None = None
        self.odom_sub = self.create_subscription(Odometry, args.odom_topic, self.odom_callback, 100)
        self.path_pub = self.create_publisher(Path, args.path_topic, 10)
        self.get_logger().info(f"Publishing odometry path {args.odom_topic} -> {args.path_topic}")

    def odom_callback(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        xyz = (float(p.x), float(p.y), float(p.z))
        if self.last_xyz is not None:
            dx = xyz[0] - self.last_xyz[0]
            dy = xyz[1] - self.last_xyz[1]
            dz = xyz[2] - self.last_xyz[2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) < self.min_step_m:
                return
        self.last_xyz = xyz

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = self.frame_id or msg.header.frame_id
        pose.pose = msg.pose.pose

        self.path.header.stamp = msg.header.stamp
        self.path.header.frame_id = pose.header.frame_id
        self.path.poses.append(pose)
        if self.max_poses > 0 and len(self.path.poses) > self.max_poses:
            self.path.poses = self.path.poses[-self.max_poses :]
        self.path_pub.publish(self.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accumulate an Odometry stream into a Path for RViz.")
    parser.add_argument("--odom-topic", default="/odometry/filtered")
    parser.add_argument("--path-topic", default="/filtered_path")
    parser.add_argument("--frame-id", default="world")
    parser.add_argument("--min-step-m", type=float, default=0.02)
    parser.add_argument("--max-poses", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = OdomPathPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
