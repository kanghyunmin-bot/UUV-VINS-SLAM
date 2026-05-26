#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMU_VECTOR_TRANSFORMS = {
    "identity": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "optical-to-ros": ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "ros-to-optical": ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
}
IMU_FRAME_TRANSFORM_CHOICES = tuple(IMU_VECTOR_TRANSFORMS)


@dataclass(frozen=True)
class StereoStamp:
    left_ns: int
    right_ns: int


@dataclass(frozen=True)
class StampWindow:
    start_index: int
    end_index: int
    reason: str


@dataclass(frozen=True)
class TimedImu:
    stamp_ns: int
    msg: Imu


class StereoBagPublisher(Node):
    def __init__(
        self,
        args: argparse.Namespace,
        stamps: list[StereoStamp],
        imu_events: list[TimedImu] | None = None,
    ) -> None:
        super().__init__("stereo_bag_realtime_publisher")
        self.args = args
        self.stamps = stamps
        self.imu_events = imu_events or []
        self.imu_index = 0
        image_qos = QoSProfile(depth=args.queue_size)
        image_qos.history = HistoryPolicy.KEEP_LAST
        image_qos.reliability = ReliabilityPolicy.RELIABLE
        imu_qos = QoSProfile(depth=args.queue_size)
        imu_qos.history = HistoryPolicy.KEEP_LAST
        imu_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.left_pub = self.create_publisher(Image, args.left_topic, image_qos)
        self.right_pub = self.create_publisher(Image, args.right_topic, image_qos)
        self.imu_pub = (
            self.create_publisher(Imu, args.imu_output_topic, imu_qos)
            if args.publish_imu_from_stamp_bag
            else None
        )

    def wait_for_subscribers(self) -> None:
        deadline = time.monotonic() + self.args.wait_subscribers
        while rclpy.ok() and time.monotonic() < deadline:
            image_ready = self.left_pub.get_subscription_count() > 0 and self.right_pub.get_subscription_count() > 0
            imu_ready = self.imu_pub is None or self.imu_pub.get_subscription_count() > 0
            if image_ready and imu_ready:
                return
            try:
                rclpy.spin_once(self, timeout_sec=0.05)
            except ExternalShutdownException:
                return
        self.get_logger().warn("subscriber wait timeout; publishing stereo bag images anyway")

    def publish_stream(self) -> None:
        left_images, right_images = read_bag_images_for_stamps(self.args, self.stamps)
        self.wait_for_subscribers()

        start_wall = time.monotonic()
        first_imu_ns = self.imu_events[0].stamp_ns if self.imu_events else self.stamps[0].left_ns
        start_stamp_ns = min(self.stamps[0].left_ns, first_imu_ns)
        published = 0
        imu_published = 0
        imu_ahead_ns = int(self.args.imu_ahead_sec * 1e9)
        for stamp in self.stamps:
            if not rclpy.ok():
                break
            imu_published += self.publish_imu_until(stamp.left_ns + imu_ahead_ns, start_wall, start_stamp_ns)
            if not self.sleep_until(stamp.left_ns, start_wall, start_stamp_ns):
                break
            left_msg = deserialize_message(left_images[stamp.left_ns], Image)
            right_msg = deserialize_message(right_images[stamp.right_ns], Image)
            restamp_image(left_msg, stamp.left_ns, self.args.left_frame_id)
            restamp_image(right_msg, stamp.right_ns, self.args.right_frame_id)
            try:
                self.left_pub.publish(left_msg)
                self.right_pub.publish(right_msg)
            except Exception as exc:
                self.get_logger().warn(f"stopping bag image publish after ROS context/publisher error: {exc}")
                break
            published += 1

        self.get_logger().info(f"published stereo bag frames={published} fused_imu={imu_published}")

    def sleep_until(self, stamp_ns: int, start_wall: float, start_stamp_ns: int) -> bool:
        target_wall = start_wall + ((stamp_ns - start_stamp_ns) / 1e9) / max(self.args.rate, 1e-9)
        while rclpy.ok() and time.monotonic() < target_wall:
            remaining = target_wall - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.005, remaining))
        return rclpy.ok()

    def publish_imu_until(self, limit_ns: int, start_wall: float, start_stamp_ns: int) -> int:
        if self.imu_pub is None:
            return 0
        count = 0
        while self.imu_index < len(self.imu_events) and self.imu_events[self.imu_index].stamp_ns <= limit_ns:
            event = self.imu_events[self.imu_index]
            if not self.sleep_until(event.stamp_ns, start_wall, start_stamp_ns):
                return count
            try:
                self.imu_pub.publish(event.msg)
            except Exception as exc:
                self.get_logger().warn(f"stopping IMU publish after ROS context/publisher error: {exc}")
                return count
            self.imu_index += 1
            count += 1
        return count


class StereoMp4Publisher(Node):
    def __init__(
        self,
        args: argparse.Namespace,
        stamps: list[StereoStamp],
        video_start_index: int,
        imu_events: list[TimedImu] | None = None,
    ) -> None:
        super().__init__("stereo_mp4_realtime_publisher")
        self.args = args
        self.stamps = stamps
        self.video_start_index = video_start_index
        self.imu_events = imu_events or []
        self.imu_index = 0
        image_qos = QoSProfile(depth=args.queue_size)
        image_qos.history = HistoryPolicy.KEEP_LAST
        image_qos.reliability = ReliabilityPolicy.RELIABLE
        imu_qos = QoSProfile(depth=args.queue_size)
        imu_qos.history = HistoryPolicy.KEEP_LAST
        imu_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.left_pub = self.create_publisher(Image, args.left_topic, image_qos)
        self.right_pub = self.create_publisher(Image, args.right_topic, image_qos)
        self.imu_pub = (
            self.create_publisher(Imu, args.imu_output_topic, imu_qos)
            if args.publish_imu_from_stamp_bag
            else None
        )

    def wait_for_subscribers(self) -> None:
        deadline = time.monotonic() + self.args.wait_subscribers
        while rclpy.ok() and time.monotonic() < deadline:
            image_ready = self.left_pub.get_subscription_count() > 0 and self.right_pub.get_subscription_count() > 0
            imu_ready = self.imu_pub is None or self.imu_pub.get_subscription_count() > 0
            if image_ready and imu_ready:
                return
            try:
                rclpy.spin_once(self, timeout_sec=0.05)
            except ExternalShutdownException:
                return
        self.get_logger().warn("subscriber wait timeout; publishing stereo MP4 anyway")

    def publish_stream(self) -> None:
        left_cap = cv2.VideoCapture(str(self.args.left_mp4))
        right_cap = cv2.VideoCapture(str(self.args.right_mp4))
        if not left_cap.isOpened():
            raise RuntimeError(f"failed to open left MP4: {self.args.left_mp4}")
        if not right_cap.isOpened():
            raise RuntimeError(f"failed to open right MP4: {self.args.right_mp4}")

        self.wait_for_subscribers()
        for _ in range(self.video_start_index):
            if not left_cap.grab() or not right_cap.grab():
                break

        max_frames = len(self.stamps)
        if self.args.max_frames > 0:
            max_frames = min(max_frames, self.args.max_frames)
        if self.args.max_duration_sec > 0.0:
            first_ns = self.stamps[0].left_ns
            max_ns = first_ns + int(self.args.max_duration_sec * 1e9)
            max_frames = min(max_frames, sum(1 for stamp in self.stamps if stamp.left_ns <= max_ns))

        start_wall = time.monotonic()
        first_imu_ns = self.imu_events[0].stamp_ns if self.imu_events else self.stamps[0].left_ns
        start_stamp_ns = min(self.stamps[0].left_ns, first_imu_ns)
        published = 0
        imu_published = 0
        imu_ahead_ns = int(self.args.imu_ahead_sec * 1e9)
        for index in range(max_frames):
            if not rclpy.ok():
                break
            stamp = self.stamps[index]
            left_ok, left_frame = left_cap.read()
            right_ok, right_frame = right_cap.read()
            if not left_ok or not right_ok:
                break

            imu_published += self.publish_imu_until(stamp.left_ns + imu_ahead_ns, start_wall, start_stamp_ns)
            if not self.sleep_until(stamp.left_ns, start_wall, start_stamp_ns):
                break

            try:
                self.left_pub.publish(make_image(left_frame, stamp.left_ns, self.args.left_frame_id))
                self.right_pub.publish(make_image(right_frame, stamp.right_ns, self.args.right_frame_id))
            except Exception as exc:
                self.get_logger().warn(f"stopping MP4 publish after ROS context/publisher error: {exc}")
                break
            published += 1

        left_cap.release()
        right_cap.release()
        self.get_logger().info(
            f"published stereo MP4 frames={published} fused_imu={imu_published} "
            f"video_start_index={self.video_start_index}"
        )

    def sleep_until(self, stamp_ns: int, start_wall: float, start_stamp_ns: int) -> bool:
        target_wall = start_wall + ((stamp_ns - start_stamp_ns) / 1e9) / max(self.args.rate, 1e-9)
        while rclpy.ok() and time.monotonic() < target_wall:
            remaining = target_wall - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.005, remaining))
        return rclpy.ok()

    def publish_imu_until(self, limit_ns: int, start_wall: float, start_stamp_ns: int) -> int:
        if self.imu_pub is None:
            return 0
        count = 0
        while self.imu_index < len(self.imu_events) and self.imu_events[self.imu_index].stamp_ns <= limit_ns:
            event = self.imu_events[self.imu_index]
            if not self.sleep_until(event.stamp_ns, start_wall, start_stamp_ns):
                return count
            try:
                self.imu_pub.publish(event.msg)
            except Exception as exc:
                self.get_logger().warn(f"stopping IMU publish after ROS context/publisher error: {exc}")
                return count
            self.imu_index += 1
            count += 1
        return count


def make_image(frame, stamp_ns: int, frame_id: str) -> Image:
    if frame.ndim == 3:
        # The D435i IR MP4s are grayscale content decoded as BGR by OpenCV.
        # Taking one channel avoids a full per-pixel color conversion in the
        # live publisher and keeps the RViz/VINS loop closer to real time.
        frame = frame[:, :, 0]
    if frame.dtype.name != "uint8":
        frame = frame.astype("uint8")
    msg = Image()
    msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.height = int(frame.shape[0])
    msg.width = int(frame.shape[1])
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = int(frame.shape[1])
    msg.data = frame.tobytes()
    return msg


def restamp_image(msg: Image, stamp_ns: int, fallback_frame_id: str) -> None:
    msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    if not msg.header.frame_id:
        msg.header.frame_id = fallback_frame_id


def read_bag_images_for_stamps(
    args: argparse.Namespace,
    stamps: list[StereoStamp],
) -> tuple[dict[int, bytes], dict[int, bytes]]:
    if args.stamp_bag is None:
        raise RuntimeError("--image-source bag requires --stamp-bag")
    conn = sqlite3.connect(str(args.stamp_bag))
    try:
        topic_ids = {name: topic_id for topic_id, name in conn.execute("select id, name from topics")}
        left_id = require_topic(topic_ids, args.left_stamp_topic)
        right_id = require_topic(topic_ids, args.right_stamp_topic)
        left_images = read_image_data_for_stamps(
            conn,
            left_id,
            args.time_source,
            {stamp.left_ns for stamp in stamps},
        )
        right_images = read_image_data_for_stamps(
            conn,
            right_id,
            args.time_source,
            {stamp.right_ns for stamp in stamps},
        )
    finally:
        conn.close()

    missing_left = [stamp.left_ns for stamp in stamps if stamp.left_ns not in left_images]
    missing_right = [stamp.right_ns for stamp in stamps if stamp.right_ns not in right_images]
    if missing_left or missing_right:
        raise RuntimeError(
            "missing source bag image messages for selected schedule: "
            f"left={len(missing_left)} right={len(missing_right)}. "
            "Use --time-source header/db that matches the selected schedule."
        )
    return left_images, right_images


def read_stereo_stamps(args: argparse.Namespace) -> list[StereoStamp]:
    if args.stereo_schedule_csv is not None:
        stamps = read_stereo_schedule_csv(args.stereo_schedule_csv)
        validate_stereo_schedule(stamps, args.sync_tolerance_ms, str(args.stereo_schedule_csv))
        return stamps
    if args.stamp_bag is None:
        return stamps_from_video(args)
    conn = sqlite3.connect(str(args.stamp_bag))
    try:
        topic_ids = {name: topic_id for topic_id, name in conn.execute("select id, name from topics")}
        left_id = require_topic(topic_ids, args.left_stamp_topic)
        right_id = require_topic(topic_ids, args.right_stamp_topic)
        left_stamps = read_image_stamps(conn, left_id, args.time_source)
        right_stamps = read_image_stamps(conn, right_id, args.time_source)
        pairs = match_stamps(left_stamps, right_stamps, int(args.sync_tolerance_ms * 1_000_000))
        if not pairs:
            raise RuntimeError("no synchronized image stamps found in stamp bag")
        return pairs
    finally:
        conn.close()


def choose_stamp_window(args: argparse.Namespace, stamps: list[StereoStamp]) -> StampWindow:
    manual_start = first_stamp_index(stamps, args.start_offset_sec)
    start_index = manual_start
    end_index = len(stamps)
    reason = "manual_or_full_stereo_window"

    if args.auto_trim_to_imu:
        if args.stamp_bag is None:
            raise RuntimeError("--auto-trim-to-imu requires --stamp-bag")
        conn = sqlite3.connect(str(args.stamp_bag))
        try:
            topic_ids = {name: topic_id for topic_id, name in conn.execute("select id, name from topics")}
            if args.imu_source_topic:
                imu_id = require_topic(topic_ids, args.imu_source_topic)
                if args.imu_source_type == "nav_msgs/msg/Odometry":
                    imu_stamps = read_odom_stamps(conn, imu_id, args.time_source)
                else:
                    imu_stamps = read_imu_stamps(conn, imu_id, args.time_source)
                if not imu_stamps:
                    raise RuntimeError(f"IMU source topic exists but has no messages: {args.imu_source_topic}")
                usable_start_ns = imu_stamps[0] + int(args.imu_start_margin_sec * 1e9)
                usable_end_ns = imu_stamps[-1] - int(args.imu_end_margin_sec * 1e9)
                source_note = args.imu_source_topic
            else:
                gyro_id = require_topic(topic_ids, args.gyro_stamp_topic)
                accel_id = require_topic(topic_ids, args.accel_stamp_topic)
                gyro_stamps = read_imu_stamps(conn, gyro_id, args.time_source)
                accel_stamps = read_imu_stamps(conn, accel_id, args.time_source)
                if not gyro_stamps or not accel_stamps:
                    raise RuntimeError("gyro/accel stamp topic exists but has no messages")
                usable_start_ns = max(gyro_stamps[0], accel_stamps[0]) + int(args.imu_start_margin_sec * 1e9)
                usable_end_ns = min(gyro_stamps[-1], accel_stamps[-1]) - int(args.imu_end_margin_sec * 1e9)
                source_note = f"{args.gyro_stamp_topic}+{args.accel_stamp_topic}"
            if usable_end_ns <= usable_start_ns:
                raise RuntimeError("invalid IMU overlap window after margins")
            imu_start_index = first_stamp_at_or_after(stamps, usable_start_ns)
            imu_end_index = first_stamp_after(stamps, usable_end_ns)
            start_index = max(start_index, imu_start_index)
            end_index = min(end_index, imu_end_index)
            reason = (
                "stereo_trimmed_to_imu_overlap "
                f"imu_source={source_note} "
                f"imu_window=[{usable_start_ns},{usable_end_ns}] "
                f"manual_start_index={manual_start}"
            )
        finally:
            conn.close()

    if start_index >= end_index:
        raise RuntimeError(
            f"empty stereo publish window: start={start_index} end={end_index} total={len(stamps)} reason={reason}"
        )
    return StampWindow(start_index=start_index, end_index=end_index, reason=reason)


def stamps_from_video(args: argparse.Namespace) -> list[StereoStamp]:
    cap = cv2.VideoCapture(str(args.left_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open left MP4: {args.left_mp4}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or args.fallback_fps
    cap.release()
    origin_ns = int(time.time() * 1e9)
    return [
        StereoStamp(origin_ns + int(index / fps * 1e9), origin_ns + int(index / fps * 1e9))
        for index in range(frames)
    ]


def require_topic(topic_ids: dict[str, int], topic: str) -> int:
    if topic not in topic_ids:
        raise RuntimeError(f"missing topic in stamp bag: {topic}")
    return topic_ids[topic]


def read_image_stamps(conn: sqlite3.Connection, topic_id: int, time_source: str) -> list[int]:
    stamps = []
    rows = conn.execute("select timestamp, data from messages where topic_id = ? order by timestamp", (topic_id,))
    for db_timestamp_ns, data in rows:
        if time_source == "db":
            stamps.append(int(db_timestamp_ns))
        else:
            msg = deserialize_message(data, Image)
            stamps.append(int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec))
    return stamps


def read_image_data_for_stamps(
    conn: sqlite3.Connection,
    topic_id: int,
    time_source: str,
    desired_stamps: set[int],
) -> dict[int, bytes]:
    messages: dict[int, bytes] = {}
    if not desired_stamps:
        return messages
    rows = conn.execute("select timestamp, data from messages where topic_id = ? order by timestamp", (topic_id,))
    for db_timestamp_ns, data in rows:
        if time_source == "db":
            stamp_ns = int(db_timestamp_ns)
        else:
            msg = deserialize_message(data, Image)
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if stamp_ns in desired_stamps and stamp_ns not in messages:
            messages[stamp_ns] = bytes(data)
            if len(messages) == len(desired_stamps):
                break
    return messages


def read_stereo_schedule_csv(path: Path) -> list[StereoStamp]:
    stamps: list[StereoStamp] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            stamps.append(StereoStamp(int(row["left_ns"]), int(row["right_ns"])))
    if not stamps:
        raise RuntimeError(f"stereo schedule CSV is empty: {path}")
    return stamps


def validate_stereo_schedule(stamps: list[StereoStamp], tolerance_ms: float, source: str) -> None:
    if tolerance_ms <= 0.0:
        return
    deltas_ms = [(stamp.right_ns - stamp.left_ns) / 1e6 for stamp in stamps]
    max_abs_delta = max(abs(delta) for delta in deltas_ms)
    if max_abs_delta <= tolerance_ms:
        return
    bad_count = sum(abs(delta) > tolerance_ms for delta in deltas_ms)
    first_bad = next(
        (index for index, delta in enumerate(deltas_ms) if abs(delta) > tolerance_ms),
        -1,
    )
    raise RuntimeError(
        "stereo schedule violates sync tolerance: "
        f"source={source} rows={len(stamps)} bad={bad_count} "
        f"tolerance_ms={tolerance_ms:.3f} max_abs_delta_ms={max_abs_delta:.3f} "
        f"first_bad_index={first_bad}"
    )


def read_imu_stamps(conn: sqlite3.Connection, topic_id: int, time_source: str) -> list[int]:
    stamps = []
    rows = conn.execute("select timestamp, data from messages where topic_id = ? order by timestamp", (topic_id,))
    for db_timestamp_ns, data in rows:
        if time_source == "db":
            stamps.append(int(db_timestamp_ns))
        else:
            msg = deserialize_message(data, Imu)
            stamps.append(int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec))
    return stamps


def read_fused_imu_events(args: argparse.Namespace, start_ns: int, end_ns: int) -> list[TimedImu]:
    if args.stamp_bag is None:
        raise RuntimeError("--publish-imu-from-stamp-bag requires --stamp-bag")
    conn = sqlite3.connect(str(args.stamp_bag))
    try:
        topic_ids = {name: topic_id for topic_id, name in conn.execute("select id, name from topics")}
        if args.imu_source_topic:
            imu_id = require_topic(topic_ids, args.imu_source_topic)
            if args.imu_source_type == "nav_msgs/msg/Odometry":
                source_msgs = read_odom_messages(conn, imu_id, args.time_source)
                events = make_odom_imu_events(args, source_msgs, start_ns, end_ns)
            else:
                source_msgs = read_imu_messages(conn, imu_id, args.time_source)
                events = [
                    TimedImu(stamp, make_direct_imu(args, msg, stamp))
                    for stamp, msg in source_msgs
                    if start_ns <= stamp <= end_ns
                ]
            if not events:
                raise RuntimeError(f"no IMU events found in window from {args.imu_source_topic}")
            if args.imu_resample_hz > 0.0:
                events = resample_imu_events(events, args.imu_resample_hz, args.imu_resample_max_gap_sec)
            return events
        gyro_id = require_topic(topic_ids, args.gyro_stamp_topic)
        accel_id = require_topic(topic_ids, args.accel_stamp_topic)
        gyro_msgs = read_imu_messages(conn, gyro_id, args.time_source)
        accel_msgs = read_imu_messages(conn, accel_id, args.time_source)
    finally:
        conn.close()

    accel_stamps = [stamp for stamp, _ in accel_msgs]
    events: list[TimedImu] = []
    max_delta_ns = int(args.imu_max_delta_sec * 1e9)
    for gyro_stamp, gyro_msg in gyro_msgs:
        if gyro_stamp < start_ns or gyro_stamp > end_ns:
            continue
        insertion = bisect.bisect_left(accel_stamps, gyro_stamp)
        candidates = []
        if insertion < len(accel_msgs):
            candidates.append(accel_msgs[insertion])
        if insertion > 0:
            candidates.append(accel_msgs[insertion - 1])
        if not candidates:
            continue
        accel_stamp, accel_msg = min(candidates, key=lambda item: abs(item[0] - gyro_stamp))
        if abs(accel_stamp - gyro_stamp) > max_delta_ns:
            continue
        events.append(TimedImu(gyro_stamp, make_fused_imu(args, gyro_msg, accel_msg)))
    if not events:
        raise RuntimeError("no fused IMU events could be created from gyro/accel topics")
    if args.imu_resample_hz > 0.0:
        events = resample_imu_events(events, args.imu_resample_hz, args.imu_resample_max_gap_sec)
    return events


def read_imu_messages(conn: sqlite3.Connection, topic_id: int, time_source: str) -> list[tuple[int, Imu]]:
    messages: list[tuple[int, Imu]] = []
    rows = conn.execute("select timestamp, data from messages where topic_id = ? order by timestamp", (topic_id,))
    for db_timestamp_ns, data in rows:
        msg = deserialize_message(data, Imu)
        if time_source == "db":
            stamp_ns = int(db_timestamp_ns)
        else:
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        messages.append((stamp_ns, msg))
    return messages


def read_odom_stamps(conn: sqlite3.Connection, topic_id: int, time_source: str) -> list[int]:
    stamps = []
    rows = conn.execute("select timestamp, data from messages where topic_id = ? order by timestamp", (topic_id,))
    for db_timestamp_ns, data in rows:
        if time_source == "db":
            stamps.append(int(db_timestamp_ns))
        else:
            msg = deserialize_message(data, Odometry)
            stamps.append(int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec))
    return stamps


def read_odom_messages(conn: sqlite3.Connection, topic_id: int, time_source: str) -> list[tuple[int, Odometry]]:
    messages: list[tuple[int, Odometry]] = []
    rows = conn.execute("select timestamp, data from messages where topic_id = ? order by timestamp", (topic_id,))
    for db_timestamp_ns, data in rows:
        msg = deserialize_message(data, Odometry)
        if time_source == "db":
            stamp_ns = int(db_timestamp_ns)
        else:
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        messages.append((stamp_ns, msg))
    return messages


def make_direct_imu(args: argparse.Namespace, source_msg: Imu, stamp_ns: int) -> Imu:
    msg = Imu()
    msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    msg.header.frame_id = args.imu_frame_id if args.imu_frame_id else source_msg.header.frame_id
    gyro = rotate_vector(
        (
            source_msg.angular_velocity.x,
            source_msg.angular_velocity.y,
            source_msg.angular_velocity.z,
        ),
        args.imu_frame_transform,
    )
    accel = rotate_vector(
        (
            source_msg.linear_acceleration.x,
            source_msg.linear_acceleration.y,
            source_msg.linear_acceleration.z,
        ),
        args.imu_frame_transform,
    )
    msg.angular_velocity.x = gyro[0]
    msg.angular_velocity.y = gyro[1]
    msg.angular_velocity.z = gyro[2]
    msg.angular_velocity_covariance = source_msg.angular_velocity_covariance
    msg.linear_acceleration.x = accel[0]
    msg.linear_acceleration.y = accel[1]
    msg.linear_acceleration.z = accel[2]
    msg.linear_acceleration_covariance = source_msg.linear_acceleration_covariance
    msg.orientation = source_msg.orientation
    msg.orientation_covariance = source_msg.orientation_covariance
    return msg


def make_odom_imu_events(
    args: argparse.Namespace,
    source_msgs: list[tuple[int, Odometry]],
    start_ns: int,
    end_ns: int,
) -> list[TimedImu]:
    events: list[TimedImu] = []
    for index, (stamp_ns, odom_msg) in enumerate(source_msgs):
        if stamp_ns < start_ns or stamp_ns > end_ns:
            continue
        prev_item = source_msgs[index - 1] if index > 0 else None
        next_item = source_msgs[index + 1] if index + 1 < len(source_msgs) else None
        events.append(TimedImu(stamp_ns, make_odom_imu(args, stamp_ns, odom_msg, prev_item, next_item)))
    return events


def make_odom_imu(
    args: argparse.Namespace,
    stamp_ns: int,
    odom_msg: Odometry,
    prev_item: tuple[int, Odometry] | None,
    next_item: tuple[int, Odometry] | None,
) -> Imu:
    msg = Imu()
    msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    msg.header.frame_id = args.imu_frame_id if args.imu_frame_id else odom_msg.child_frame_id
    msg.orientation = odom_msg.pose.pose.orientation
    normalize_imu_orientation(msg)

    twist_angular = odom_msg.twist.twist.angular
    twist_norm = math.sqrt(
        twist_angular.x * twist_angular.x +
        twist_angular.y * twist_angular.y +
        twist_angular.z * twist_angular.z
    )
    if math.isfinite(twist_norm) and twist_norm > 1e-7:
        gyro = (twist_angular.x, twist_angular.y, twist_angular.z)
    else:
        gyro = odom_angular_velocity_from_neighbors(stamp_ns, msg, prev_item, next_item)
    gyro = rotate_vector(gyro, args.imu_frame_transform)
    msg.angular_velocity.x = gyro[0]
    msg.angular_velocity.y = gyro[1]
    msg.angular_velocity.z = gyro[2]
    msg.angular_velocity_covariance = [0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05]

    accel = gravity_in_body_from_orientation(msg, args.odom_gravity_norm)
    accel = rotate_vector(accel, args.imu_frame_transform)
    msg.linear_acceleration.x = accel[0]
    msg.linear_acceleration.y = accel[1]
    msg.linear_acceleration.z = accel[2]
    msg.linear_acceleration_covariance = [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0]
    msg.orientation_covariance = [0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02]
    return msg


def make_fused_imu(args: argparse.Namespace, gyro_msg: Imu, accel_msg: Imu) -> Imu:
    msg = Imu()
    msg.header.stamp.sec = gyro_msg.header.stamp.sec
    msg.header.stamp.nanosec = gyro_msg.header.stamp.nanosec
    msg.header.frame_id = args.imu_frame_id if args.imu_frame_id else gyro_msg.header.frame_id
    gyro = rotate_vector(
        (
            gyro_msg.angular_velocity.x,
            gyro_msg.angular_velocity.y,
            gyro_msg.angular_velocity.z,
        ),
        args.imu_frame_transform,
    )
    accel = rotate_vector(
        (
            accel_msg.linear_acceleration.x,
            accel_msg.linear_acceleration.y,
            accel_msg.linear_acceleration.z,
        ),
        args.imu_frame_transform,
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
    return msg


def resample_imu_events(events: list[TimedImu], hz: float, max_gap_sec: float) -> list[TimedImu]:
    if len(events) < 2 or hz <= 0.0:
        return events
    step_ns = max(1, int(1e9 / hz))
    stamps = [event.stamp_ns for event in events]
    max_gap_ns = int(max_gap_sec * 1e9) if max_gap_sec > 0.0 else 0
    out: list[TimedImu] = []
    index = 0
    stamp_ns = stamps[0]
    while stamp_ns <= stamps[-1]:
        while index + 1 < len(events) and stamps[index + 1] < stamp_ns:
            index += 1
        if index + 1 >= len(events):
            break
        left = events[index]
        right = events[index + 1]
        gap_ns = right.stamp_ns - left.stamp_ns
        if gap_ns <= 0:
            stamp_ns += step_ns
            continue
        if max_gap_ns > 0 and gap_ns > max_gap_ns:
            stamp_ns += step_ns
            continue
        alpha = (stamp_ns - left.stamp_ns) / gap_ns
        alpha = max(0.0, min(1.0, alpha))
        out.append(TimedImu(stamp_ns, interpolate_imu(left.msg, right.msg, stamp_ns, alpha)))
        stamp_ns += step_ns
    return out or events


def interpolate_imu(left: Imu, right: Imu, stamp_ns: int, alpha: float) -> Imu:
    msg = Imu()
    msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    msg.header.frame_id = left.header.frame_id
    msg.angular_velocity.x = lerp(left.angular_velocity.x, right.angular_velocity.x, alpha)
    msg.angular_velocity.y = lerp(left.angular_velocity.y, right.angular_velocity.y, alpha)
    msg.angular_velocity.z = lerp(left.angular_velocity.z, right.angular_velocity.z, alpha)
    msg.angular_velocity_covariance = left.angular_velocity_covariance
    msg.linear_acceleration.x = lerp(left.linear_acceleration.x, right.linear_acceleration.x, alpha)
    msg.linear_acceleration.y = lerp(left.linear_acceleration.y, right.linear_acceleration.y, alpha)
    msg.linear_acceleration.z = lerp(left.linear_acceleration.z, right.linear_acceleration.z, alpha)
    msg.linear_acceleration_covariance = left.linear_acceleration_covariance
    msg.orientation.x = lerp(left.orientation.x, right.orientation.x, alpha)
    msg.orientation.y = lerp(left.orientation.y, right.orientation.y, alpha)
    msg.orientation.z = lerp(left.orientation.z, right.orientation.z, alpha)
    msg.orientation.w = lerp(left.orientation.w, right.orientation.w, alpha)
    normalize_imu_orientation(msg)
    msg.orientation_covariance = left.orientation_covariance
    return msg


def lerp(left: float, right: float, alpha: float) -> float:
    return left + (right - left) * alpha


def normalize_imu_orientation(msg: Imu) -> None:
    norm = math.sqrt(
        msg.orientation.x * msg.orientation.x
        + msg.orientation.y * msg.orientation.y
        + msg.orientation.z * msg.orientation.z
        + msg.orientation.w * msg.orientation.w
    )
    if norm <= 1e-9 or not math.isfinite(norm):
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0
        return
    msg.orientation.x /= norm
    msg.orientation.y /= norm
    msg.orientation.z /= norm
    msg.orientation.w /= norm


def quaternion_tuple_from_odom(msg: Odometry) -> tuple[float, float, float, float]:
    q = msg.pose.pose.orientation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm <= 1e-9 or not math.isfinite(norm):
        return (0.0, 0.0, 0.0, 1.0)
    return (q.x / norm, q.y / norm, q.z / norm, q.w / norm)


def quaternion_tuple_from_imu(msg: Imu) -> tuple[float, float, float, float]:
    q = msg.orientation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm <= 1e-9 or not math.isfinite(norm):
        return (0.0, 0.0, 0.0, 1.0)
    return (q.x / norm, q.y / norm, q.z / norm, q.w / norm)


def quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def quat_multiply(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate(
    q: tuple[float, float, float, float],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    p = (v[0], v[1], v[2], 0.0)
    out = quat_multiply(quat_multiply(q, p), quat_conjugate(q))
    return (out[0], out[1], out[2])


def odom_angular_velocity_from_neighbors(
    stamp_ns: int,
    current_msg: Imu,
    prev_item: tuple[int, Odometry] | None,
    next_item: tuple[int, Odometry] | None,
) -> tuple[float, float, float]:
    if prev_item is not None and next_item is not None:
        left_stamp, left_msg = prev_item
        right_stamp, right_msg = next_item
        q_left = quaternion_tuple_from_odom(left_msg)
        q_right = quaternion_tuple_from_odom(right_msg)
    elif prev_item is not None:
        left_stamp, left_msg = prev_item
        right_stamp = stamp_ns
        q_left = quaternion_tuple_from_odom(left_msg)
        q_right = quaternion_tuple_from_imu(current_msg)
    elif next_item is not None:
        left_stamp = stamp_ns
        right_stamp, right_msg = next_item
        q_left = quaternion_tuple_from_imu(current_msg)
        q_right = quaternion_tuple_from_odom(right_msg)
    else:
        return (0.0, 0.0, 0.0)

    dt = (right_stamp - left_stamp) * 1e-9
    if dt <= 1e-6 or not math.isfinite(dt):
        return (0.0, 0.0, 0.0)
    q_delta = quat_multiply(quat_conjugate(q_left), q_right)
    if q_delta[3] < 0.0:
        q_delta = (-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3])
    vec_norm = math.sqrt(q_delta[0] * q_delta[0] + q_delta[1] * q_delta[1] + q_delta[2] * q_delta[2])
    if vec_norm <= 1e-9 or not math.isfinite(vec_norm):
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vec_norm, max(-1.0, min(1.0, q_delta[3])))
    scale = angle / (vec_norm * dt)
    return (q_delta[0] * scale, q_delta[1] * scale, q_delta[2] * scale)


def gravity_in_body_from_orientation(msg: Imu, gravity_norm: float) -> tuple[float, float, float]:
    q_body_to_world = quaternion_tuple_from_imu(msg)
    return quat_rotate(quat_conjugate(q_body_to_world), (0.0, 0.0, gravity_norm))


def rotate_vector(vector: tuple[float, float, float], transform: str) -> tuple[float, float, float]:
    x, y, z = vector
    try:
        matrix = IMU_VECTOR_TRANSFORMS[transform]
    except KeyError as exc:
        raise ValueError(f"Unsupported IMU frame transform: {transform}") from exc
    return tuple(row[0] * x + row[1] * y + row[2] * z for row in matrix)


def match_stamps(left_stamps: list[int], right_stamps: list[int], tolerance_ns: int) -> list[StereoStamp]:
    pairs = []
    used_right: set[int] = set()
    for left_ns in left_stamps:
        insertion = bisect.bisect_left(right_stamps, left_ns)
        candidates = []
        if insertion < len(right_stamps):
            candidates.append(insertion)
        if insertion > 0:
            candidates.append(insertion - 1)
        candidates = [idx for idx in candidates if idx not in used_right]
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: abs(right_stamps[idx] - left_ns))
        if abs(right_stamps[best] - left_ns) <= tolerance_ns:
            pairs.append(StereoStamp(left_ns, right_stamps[best]))
            used_right.add(best)
    return pairs


def first_stamp_index(stamps: list[StereoStamp], start_offset_sec: float) -> int:
    if start_offset_sec <= 0.0:
        return 0
    threshold_ns = stamps[0].left_ns + int(start_offset_sec * 1e9)
    for index, stamp in enumerate(stamps):
        if stamp.left_ns >= threshold_ns:
            return index
    return max(0, len(stamps) - 1)


def first_stamp_at_or_after(stamps: list[StereoStamp], threshold_ns: int) -> int:
    for index, stamp in enumerate(stamps):
        if stamp.left_ns >= threshold_ns:
            return index
    return len(stamps)


def first_stamp_after(stamps: list[StereoStamp], threshold_ns: int) -> int:
    for index, stamp in enumerate(stamps):
        if stamp.left_ns > threshold_ns:
            return index
    return len(stamps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish stereo MP4 frames as live ROS2 image topics using rosbag timestamps.")
    parser.add_argument(
        "--image-source",
        choices=("mp4", "bag"),
        default="mp4",
        help="Publish decoded MP4 frames, or replay original sensor_msgs/Image messages from --stamp-bag.",
    )
    parser.add_argument("--left-mp4", type=Path, default=PROJECT_ROOT / "video_src/camera_camera_infra1_image_rect_raw.mp4")
    parser.add_argument("--right-mp4", type=Path, default=PROJECT_ROOT / "video_src/camera_camera_infra2_image_rect_raw.mp4")
    parser.add_argument("--left-topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--right-topic", default="/camera/camera/infra2/image_rect_raw")
    parser.add_argument("--left-frame-id", default="camera_infra1_optical_frame")
    parser.add_argument("--right-frame-id", default="camera_infra2_optical_frame")
    parser.add_argument("--stamp-bag", type=Path)
    parser.add_argument(
        "--stereo-schedule-csv",
        type=Path,
        help="Use an already-cropped left_ns/right_ns schedule instead of rematching image stamps from --stamp-bag.",
    )
    parser.add_argument("--left-stamp-topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--right-stamp-topic", default="/camera/camera/infra2/image_rect_raw")
    parser.add_argument("--time-source", choices=("header", "db"), default="header")
    parser.add_argument("--sync-tolerance-ms", type=float, default=3.0)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument(
        "--start-offset-sec",
        type=float,
        default=0.0,
        help="Optional manual skip before publishing. Prefer --auto-trim-to-imu for live VINS runs.",
    )
    parser.add_argument(
        "--auto-trim-to-imu",
        action="store_true",
        help="Trim stereo MP4/stamps to the overlapping gyro+accel timestamp window from --stamp-bag.",
    )
    parser.add_argument("--gyro-stamp-topic", default="/camera/camera/gyro/sample")
    parser.add_argument("--accel-stamp-topic", default="/camera/camera/accel/sample")
    parser.add_argument(
        "--imu-source-topic",
        default="",
        help="Use an existing IMU-like topic from --stamp-bag instead of fusing gyro+accel topics.",
    )
    parser.add_argument(
        "--imu-source-type",
        choices=("sensor_msgs/msg/Imu", "nav_msgs/msg/Odometry"),
        default="sensor_msgs/msg/Imu",
        help="Message type for --imu-source-topic.",
    )
    parser.add_argument(
        "--odom-gravity-norm",
        type=float,
        default=9.805,
        help="Synthetic gravity magnitude used when converting nav_msgs/Odometry orientation to Imu acceleration.",
    )
    parser.add_argument("--imu-start-margin-sec", type=float, default=0.03)
    parser.add_argument("--imu-end-margin-sec", type=float, default=0.03)
    parser.add_argument(
        "--publish-imu-from-stamp-bag",
        action="store_true",
        help="Publish a fused Imu stream from gyro/accel topics in --stamp-bag, interleaved with MP4 images.",
    )
    parser.add_argument("--imu-output-topic", default="/camera/camera/imu")
    parser.add_argument("--imu-frame-id", default="camera_imu")
    parser.add_argument(
        "--imu-frame-transform",
        choices=IMU_FRAME_TRANSFORM_CHOICES,
        default="identity",
        help=(
            "Optional IMU vector frame conversion for standard frame contracts. "
            "Use identity for the active D435i/MAVROS replay unless a measured static TF proves otherwise."
        ),
    )
    parser.add_argument("--imu-max-delta-sec", type=float, default=0.025)
    parser.add_argument("--imu-resample-hz", type=float, default=0.0)
    parser.add_argument("--imu-resample-max-gap-sec", type=float, default=1.0)
    parser.add_argument(
        "--imu-ahead-sec",
        type=float,
        default=0.02,
        help="Publish IMU up to this much after each image stamp before the image, so VINS has a complete bracket.",
    )
    parser.add_argument("--max-duration-sec", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--video-start-index-override",
        type=int,
        default=-1,
        help=(
            "Override how many frames are skipped in the MP4 before publishing. "
            "Use 0 when the MP4 was already cropped but stamps still come from the original bag."
        ),
    )
    parser.add_argument("--wait-subscribers", type=float, default=10.0)
    parser.add_argument("--queue-size", type=int, default=200)
    parser.add_argument(
        "--export-schedule-dir",
        type=Path,
        help="Write stereo/IMU timestamp CSVs for synchronized replay/debugging.",
    )
    parser.add_argument(
        "--export-schedule-only",
        action="store_true",
        help="Exit after writing --export-schedule-dir without publishing through rclpy.",
    )
    parser.add_argument(
        "--uniform-stereo-timestamps",
        action="store_true",
        help=(
            "Redistribute the selected stereo frames uniformly between the first and "
            "last selected timestamps. Use this for MP4 files whose frame index, not "
            "per-frame ROS timestamp jitter, is the actual playback contract."
        ),
    )
    return parser.parse_args()


def apply_publish_limits(args: argparse.Namespace, stamps: list[StereoStamp]) -> list[StereoStamp]:
    max_frames = len(stamps)
    if args.max_frames > 0:
        max_frames = min(max_frames, args.max_frames)
    if args.max_duration_sec > 0.0:
        first_ns = stamps[0].left_ns
        max_ns = first_ns + int(args.max_duration_sec * 1e9)
        max_frames = min(max_frames, sum(1 for stamp in stamps if stamp.left_ns <= max_ns))
    return stamps[:max_frames]


def make_uniform_stereo_stamps(stamps: list[StereoStamp]) -> list[StereoStamp]:
    if len(stamps) <= 1:
        return stamps
    first_ns = stamps[0].left_ns
    last_ns = stamps[-1].left_ns
    if last_ns <= first_ns:
        return stamps
    span_ns = last_ns - first_ns
    count = len(stamps)
    uniform: list[StereoStamp] = []
    for index in range(count):
        left_ns = first_ns + int(round(span_ns * index / max(1, count - 1)))
        uniform.append(StereoStamp(left_ns, left_ns))
    return uniform


def write_schedule(
    args: argparse.Namespace,
    stamps: list[StereoStamp],
    video_start_index: int,
    imu_events: list[TimedImu] | None,
) -> None:
    if args.export_schedule_dir is None:
        return
    args.export_schedule_dir.mkdir(parents=True, exist_ok=True)
    stereo_csv = args.export_schedule_dir / "stereo_schedule.csv"
    imu_csv = args.export_schedule_dir / "imu_schedule.csv"
    metadata = args.export_schedule_dir / "metadata.env"
    with stereo_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["left_ns", "right_ns"])
        for stamp in stamps:
            writer.writerow([stamp.left_ns, stamp.right_ns])
    with imu_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "stamp_ns",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "accel_x",
                "accel_y",
                "accel_z",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
            ]
        )
        for event in imu_events or []:
            msg = event.msg
            writer.writerow(
                [
                    event.stamp_ns,
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ]
            )
    metadata.write_text(
        "\n".join(
            [
                f"VIDEO_START_INDEX={video_start_index}",
                f"STEREO_SCHEDULE_CSV='{stereo_csv}'",
                f"IMU_SCHEDULE_CSV='{imu_csv}'",
                f"STEREO_SCHEDULE_FRAMES={len(stamps)}",
                f"IMU_SCHEDULE_EVENTS={len(imu_events or [])}",
                "",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    if args.image_source == "bag" and args.uniform_stereo_timestamps:
        raise SystemExit("--image-source bag cannot use --uniform-stereo-timestamps because bag images need exact source stamps")
    stamps = read_stereo_stamps(args)
    if not stamps:
        raise SystemExit("no stereo timestamps available")
    window = choose_stamp_window(args, stamps)
    trimmed_stamps = apply_publish_limits(args, stamps[window.start_index:window.end_index])
    if args.uniform_stereo_timestamps:
        trimmed_stamps = make_uniform_stereo_stamps(trimmed_stamps)
    imu_events: list[TimedImu] | None = None
    if args.publish_imu_from_stamp_bag:
        imu_start_ns = trimmed_stamps[0].left_ns - int(0.10 * 1e9)
        imu_end_ns = trimmed_stamps[-1].left_ns + int(max(args.imu_ahead_sec, 0.05) * 1e9)
        imu_events = read_fused_imu_events(args, imu_start_ns, imu_end_ns)
    print(
        "stereo publish window: "
        f"start_index={window.start_index} end_index={window.end_index} "
        f"frames={len(trimmed_stamps)} imu_events={len(imu_events or [])} reason={window.reason}",
        flush=True,
    )
    video_start_index = window.start_index
    if args.video_start_index_override >= 0:
        video_start_index = args.video_start_index_override
    write_schedule(args, trimmed_stamps, video_start_index, imu_events)
    if args.export_schedule_only:
        return
    rclpy.init()
    if args.image_source == "bag":
        node = StereoBagPublisher(args, trimmed_stamps, imu_events)
    else:
        node = StereoMp4Publisher(args, trimmed_stamps, video_start_index, imu_events)
    try:
        node.publish_stream()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
