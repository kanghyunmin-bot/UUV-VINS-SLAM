#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
from collections import deque

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


class RealsenseImuFuseNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("realsense_imu_fuse")
        self.args = args
        qos = QoSProfile(depth=args.queue_size)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.accel_buf: deque[Imu] = deque(maxlen=args.buffer_size)
        self.publisher = self.create_publisher(Imu, args.output_topic, qos)
        self.create_subscription(Imu, args.accel_topic, self.accel_callback, qos)
        self.create_subscription(Imu, args.gyro_topic, self.gyro_callback, qos)
        self.published = 0
        self.dropped = 0
        self.get_logger().info(
            f"fusing gyro={args.gyro_topic} accel={args.accel_topic} -> {args.output_topic}"
        )

    def accel_callback(self, msg: Imu) -> None:
        self.accel_buf.append(msg)

    def gyro_callback(self, gyro_msg: Imu) -> None:
        accel_msg = self.nearest_accel(gyro_msg)
        if accel_msg is None:
            self.dropped += 1
            return

        msg = Imu()
        msg.header = gyro_msg.header
        if self.args.frame_id:
            msg.header.frame_id = self.args.frame_id
        gyro = rotate_vector(
            (
                gyro_msg.angular_velocity.x,
                gyro_msg.angular_velocity.y,
                gyro_msg.angular_velocity.z,
            ),
            self.args.frame_transform,
        )
        accel = rotate_vector(
            (
                accel_msg.linear_acceleration.x,
                accel_msg.linear_acceleration.y,
                accel_msg.linear_acceleration.z,
            ),
            self.args.frame_transform,
        )
        msg.angular_velocity.x = gyro[0]
        msg.angular_velocity.y = gyro[1]
        msg.angular_velocity.z = gyro[2]
        msg.angular_velocity_covariance = gyro_msg.angular_velocity_covariance
        msg.linear_acceleration.x = accel[0]
        msg.linear_acceleration.y = accel[1]
        msg.linear_acceleration.z = accel[2]
        msg.linear_acceleration_covariance = accel_msg.linear_acceleration_covariance
        msg.orientation_covariance[0] = -1.0
        self.publisher.publish(msg)
        self.published += 1

    def nearest_accel(self, gyro_msg: Imu) -> Imu | None:
        if not self.accel_buf:
            return None
        gyro_t = stamp_to_float(gyro_msg)
        accel_times = [stamp_to_float(msg) for msg in self.accel_buf]
        insert = bisect.bisect_left(accel_times, gyro_t)
        candidates = []
        if insert < len(self.accel_buf):
            candidates.append(self.accel_buf[insert])
        if insert > 0:
            candidates.append(self.accel_buf[insert - 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda msg: abs(stamp_to_float(msg) - gyro_t))
        if abs(stamp_to_float(best) - gyro_t) > self.args.max_delta_sec:
            return None
        return best


def stamp_to_float(msg: Imu) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def rotate_vector(vector: tuple[float, float, float], transform: str) -> tuple[float, float, float]:
    x, y, z = vector
    if transform == "identity":
        return (x, y, z)
    if transform == "optical-to-ros":
        return (z, -x, -y)
    if transform == "ros-to-optical":
        return (-y, -z, x)
    raise ValueError(f"Unsupported frame transform: {transform}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse D435i accel/sample and gyro/sample into one Imu topic.")
    parser.add_argument("--gyro-topic", default="/camera/camera/gyro/sample")
    parser.add_argument("--accel-topic", default="/camera/camera/accel/sample")
    parser.add_argument("--output-topic", default="/camera/camera/imu")
    parser.add_argument("--frame-id", default="camera_imu")
    parser.add_argument(
        "--frame-transform",
        choices=("identity", "optical-to-ros", "ros-to-optical"),
        default="identity",
        help=(
            "Rotate accel/gyro vectors before publishing. RealSense D435i bag "
            "samples are stamped in camera_*_optical_frame; VINS body_T_cam "
            "usually expects the ROS camera/body frame."
        ),
    )
    parser.add_argument("--max-delta-sec", type=float, default=0.025)
    parser.add_argument("--buffer-size", type=int, default=400)
    parser.add_argument("--queue-size", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = RealsenseImuFuseNode(args)
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.get_logger().info(f"published={node.published} dropped={node.dropped}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
