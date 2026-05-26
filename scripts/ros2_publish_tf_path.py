#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class TfPathPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("tf_path_publisher")
        self.parent_frame = args.parent_frame
        self.child_frame = args.child_frame
        self.min_step_m = args.min_step_m
        self.max_poses = args.max_poses
        self.path = Path()
        self.path.header.frame_id = self.parent_frame
        self.last_xyz: tuple[float, float, float] | None = None

        self.tf_sub = self.create_subscription(TFMessage, args.tf_topic, self.tf_callback, 100)
        self.path_pub = self.create_publisher(Path, args.path_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)
        self.get_logger().info(
            f"Publishing TF-derived path {self.parent_frame}->{self.child_frame} "
            f"to {args.path_topic} and {args.odom_topic}"
        )

    def tf_callback(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if transform.header.frame_id != self.parent_frame:
                continue
            if transform.child_frame_id != self.child_frame:
                continue
            self.publish_transform(transform)

    def publish_transform(self, transform) -> None:
        t = transform.transform.translation
        xyz = (float(t.x), float(t.y), float(t.z))
        if self.last_xyz is not None:
            dx = xyz[0] - self.last_xyz[0]
            dy = xyz[1] - self.last_xyz[1]
            dz = xyz[2] - self.last_xyz[2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) < self.min_step_m:
                return
        self.last_xyz = xyz

        pose = PoseStamped()
        pose.header = transform.header
        pose.header.frame_id = self.parent_frame
        pose.pose.position.x = xyz[0]
        pose.pose.position.y = xyz[1]
        pose.pose.position.z = xyz[2]
        pose.pose.orientation = transform.transform.rotation
        self.path.header.stamp = transform.header.stamp
        self.path.poses.append(pose)
        if self.max_poses > 0 and len(self.path.poses) > self.max_poses:
            self.path.poses = self.path.poses[-self.max_poses :]

        odom = Odometry()
        odom.header = transform.header
        odom.header.frame_id = self.parent_frame
        odom.child_frame_id = self.child_frame
        odom.pose.pose = pose.pose

        self.path_pub.publish(self.path)
        self.odom_pub.publish(odom)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Republish a TF edge as a Path and Odometry stream for RViz comparison.")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--parent-frame", default="world")
    parser.add_argument("--child-frame", default="fcu_link")
    parser.add_argument("--path-topic", default="/tf_localized_path")
    parser.add_argument("--odom-topic", default="/tf_localized_odometry")
    parser.add_argument("--min-step-m", type=float, default=0.02)
    parser.add_argument("--max-poses", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = TfPathPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
