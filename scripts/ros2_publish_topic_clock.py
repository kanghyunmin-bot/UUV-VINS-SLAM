#!/usr/bin/env python3
from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class TopicClockPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("topic_clock_publisher")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        clock_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(Clock, "/clock", clock_qos)
        self.sub = self.create_subscription(args.msg_type, args.stamp_topic, self.callback, qos)
        self.last_nsec = -1
        self.allow_backward_jump = args.allow_backward_jump
        self.get_logger().info(f"Publishing /clock from {args.stamp_topic}")

    def callback(self, msg) -> None:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return
        nsec = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if nsec <= self.last_nsec:
            if not self.allow_backward_jump:
                return
            self.get_logger().warn(
                f"Clock jumped backward: {self.last_nsec} -> {nsec}; publishing for loop replay"
            )
        self.last_nsec = nsec
        clock = Clock()
        clock.clock = stamp
        self.pub.publish(clock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish /clock from another stamped ROS topic.")
    parser.add_argument("--stamp-topic", default="/mavros/imu/data")
    parser.add_argument("--msg-type", default="sensor_msgs.msg.Imu")
    parser.add_argument(
        "--allow-backward-jump",
        action="store_true",
        help="Publish repeated-loop timestamps even when they jump backward.",
    )
    args = parser.parse_args()
    module_name, class_name = args.msg_type.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    args.msg_type = getattr(module, class_name)
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = TopicClockPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
