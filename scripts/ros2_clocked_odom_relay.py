#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock


class ClockedOdomRelay(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("clocked_odom_relay")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=args.queue_size,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.clock_pub = self.create_publisher(Clock, args.clock_topic, qos)
        self.odom_pub = self.create_publisher(Odometry, args.output_topic, qos)
        self.sub = self.create_subscription(Odometry, args.input_topic, self.callback, qos)
        self.allow_backward_jump = args.allow_backward_jump
        self.odom_delay_sec = args.odom_delay_sec
        self.last_nsec = -1
        self.get_logger().info(
            f"Relaying {args.input_topic} -> {args.output_topic}, clock={args.clock_topic}"
        )

    def callback(self, msg: Odometry) -> None:
        nsec = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if nsec <= self.last_nsec:
            if not self.allow_backward_jump:
                return
            self.get_logger().warn(
                f"Clock jumped backward: {self.last_nsec} -> {nsec}; publishing for loop replay"
            )
        self.last_nsec = nsec

        clock = Clock()
        clock.clock = msg.header.stamp
        self.clock_pub.publish(clock)

        if self.odom_delay_sec > 0.0:
            time.sleep(self.odom_delay_sec)
        self.odom_pub.publish(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish /clock before relaying odometry.")
    parser.add_argument("--input-topic", default="/odometry")
    parser.add_argument("--output-topic", default="/vins/odometry_clocked")
    parser.add_argument("--clock-topic", default="/clock")
    parser.add_argument("--queue-size", type=int, default=100)
    parser.add_argument("--odom-delay-sec", type=float, default=0.01)
    parser.add_argument("--allow-backward-jump", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ClockedOdomRelay(args)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
