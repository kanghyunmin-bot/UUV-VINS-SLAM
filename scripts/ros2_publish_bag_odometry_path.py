#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import sqlite3
import time
import math
from pathlib import Path as FilePath

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.node import Node
from rclpy.serialization import deserialize_message
import numpy as np


class BagOdometryPathPublisher(Node):
    def __init__(self, args: argparse.Namespace, rows: list[tuple[int, Odometry]]) -> None:
        super().__init__("bag_odometry_path_publisher")
        self.args = args
        self.rows = rows
        self.alignment = compute_alignment(args, rows)
        self.odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)
        self.path_pub = self.create_publisher(RosPath, args.path_topic, 10)
        self.get_logger().info(
            f"Publishing bag odometry {args.source_topic} -> {args.odom_topic}, {args.path_topic}; "
            f"messages={len(rows)} rate={args.rate} loop={args.loop}"
        )

    def play_once(self) -> None:
        path = RosPath()
        path.header.frame_id = self.args.frame_id
        if self.args.publish_empty_on_start:
            path.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(path)
        first_stamp = self.rows[0][0]
        wall_start = time.monotonic()
        last_publish_ns: int | None = None
        for stamp_ns, source in self.rows:
            if not rclpy.ok():
                return
            if self.args.rate > 0.0:
                target = (stamp_ns - first_stamp) / 1e9 / self.args.rate
                delay = target - (time.monotonic() - wall_start)
                if delay > 0.0:
                    time.sleep(delay)

            msg = Odometry()
            msg.header = copy.deepcopy(source.header)
            if self.args.restamp_now:
                msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.args.frame_id or source.header.frame_id
            msg.child_frame_id = self.args.child_frame_id or source.child_frame_id
            msg.pose = source.pose
            msg.twist = source.twist
            apply_alignment(msg, self.alignment)
            self.odom_pub.publish(msg)

            if last_publish_ns is not None and self.args.path_stride_ns > 0:
                if stamp_ns - last_publish_ns < self.args.path_stride_ns:
                    continue
            last_publish_ns = stamp_ns

            pose = PoseStamped()
            pose.header = msg.header
            pose.pose = msg.pose.pose
            path.header.stamp = msg.header.stamp
            path.poses.append(pose)
            if self.args.max_poses > 0 and len(path.poses) > self.args.max_poses:
                path.poses = path.poses[-self.args.max_poses :]
            self.path_pub.publish(path)

    def run(self) -> None:
        while rclpy.ok():
            self.play_once()
            if not self.args.loop:
                return
            time.sleep(self.args.loop_pause_sec)


def require_topic(conn: sqlite3.Connection, topic: str) -> int:
    rows = list(conn.execute("select id from topics where name = ?", (topic,)))
    if not rows:
        raise RuntimeError(f"missing odometry topic in bag: {topic}")
    return int(rows[0][0])


def stamp_ns(msg: Odometry, db_timestamp_ns: int, time_source: str) -> int:
    if time_source == "db":
        return int(db_timestamp_ns)
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def read_rows(args: argparse.Namespace) -> list[tuple[int, Odometry]]:
    conn = sqlite3.connect(str(args.bag_db))
    try:
        topic_id = require_topic(conn, args.source_topic)
        rows: list[tuple[int, Odometry]] = []
        query = "select timestamp, data from messages where topic_id = ? order by timestamp"
        for db_timestamp_ns, data in conn.execute(query, (topic_id,)):
            msg = deserialize_message(data, Odometry)
            stamp = stamp_ns(msg, int(db_timestamp_ns), args.time_source)
            if args.start_offset_sec > 0.0 and rows:
                pass
            rows.append((stamp, msg))
    finally:
        conn.close()
    if not rows:
        raise RuntimeError(f"no odometry messages in {args.source_topic}")

    start_ns = int(args.start_stamp_ns) if args.start_stamp_ns > 0 else rows[0][0] + int(args.start_offset_sec * 1e9)
    end_ns = start_ns + int(args.max_duration_sec * 1e9) if args.max_duration_sec > 0.0 else None
    rows = [(stamp, msg) for stamp, msg in rows if stamp >= start_ns and (end_ns is None or stamp <= end_ns)]
    if not rows:
        raise RuntimeError("odometry time window is empty")
    return rows


def csv_to_xy(row: dict[str, str], coordinate_frame: str) -> tuple[float, float]:
    x = float(row["x"])
    y = float(row["y"])
    z = float(row.get("z", "0.0"))
    if coordinate_frame in {"raw", "opencv", "ned"}:
        return x, y
    if coordinate_frame == "flip-y":
        return x, -y
    if coordinate_frame == "ros":
        return z, -x
    raise ValueError(f"unsupported coordinate frame: {coordinate_frame}")


def compute_alignment(args: argparse.Namespace, rows: list[tuple[int, Odometry]]) -> dict[str, object] | None:
    if args.align_mode == "none" or not args.align_csv:
        return None
    target_rows = []
    with FilePath(args.align_csv).open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            x, y = csv_to_xy(row, args.align_csv_coordinate_frame)
            target_rows.append((float(row["timestamp_sec"]), x, y, float(row.get("z", "0.0"))))
    if len(target_rows) < 3 or len(rows) < 3:
        return None

    source_t0 = rows[0][0]
    source_times = np.array([(stamp - source_t0) / 1e9 for stamp, _ in rows], dtype=float)
    source_x = np.array([msg.pose.pose.position.x for _, msg in rows], dtype=float)
    source_y = np.array([msg.pose.pose.position.y for _, msg in rows], dtype=float)

    if args.align_mode == "start_yaw":
        yaw = math.radians(args.align_yaw_deg)
        c = math.cos(yaw)
        s = math.sin(yaw)
        r = np.array([[c, -s], [s, c]], dtype=float)
        source_start = np.array([source_x[0], source_y[0]], dtype=float)
        target_start = np.array([target_rows[0][1], target_rows[0][2]], dtype=float)
        translation = target_start - (r @ source_start)
        translation_z = float(target_rows[0][3]) - float(rows[0][1].pose.pose.position.z)
        print(
            f"localization display alignment mode=start_yaw "
            f"yaw_deg={args.align_yaw_deg:.2f} "
            f"start=({target_start[0]:.3f},{target_start[1]:.3f},{target_rows[0][3]:.3f})",
            flush=True,
        )
        return {"rotation": r, "translation": translation, "translation_z": translation_z, "scale": 1.0, "yaw": yaw}

    target_times = np.array([item[0] for item in target_rows], dtype=float)
    target_xy = np.array([[item[1], item[2]] for item in target_rows], dtype=float)
    keep = (target_times >= source_times[0]) & (target_times <= source_times[-1])
    target_times = target_times[keep]
    target_xy = target_xy[keep]
    if len(target_times) < 3:
        return None
    source_xy = np.column_stack(
        [
            np.interp(target_times, source_times, source_x),
            np.interp(target_times, source_times, source_y),
        ]
    )
    if args.align_mode == "origin":
        r = np.eye(2)
        scale = 1.0
        translation = target_xy[0] - source_xy[0]
        aligned = source_xy + translation
        rmse = float(np.sqrt(np.mean(np.sum((aligned - target_xy) ** 2, axis=1))))
        print(
            f"localization display alignment mode=origin "
            f"translation=({translation[0]:.3f},{translation[1]:.3f}) rmse_xy={rmse:.3f}m",
            flush=True,
        )
        return {"rotation": r, "translation": translation, "scale": scale, "yaw": 0.0}

    source_center = source_xy.mean(axis=0)
    target_center = target_xy.mean(axis=0)
    source_zero = source_xy - source_center
    target_zero = target_xy - target_center
    h = source_zero.T @ target_zero
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    scale = 1.0
    if args.align_mode == "similarity":
        denom = float(np.sum(source_zero * source_zero))
        if denom > 1e-9:
            scale = float(np.sum(s) / denom)
    translation = target_center - scale * (r @ source_center)
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    aligned = (scale * (source_xy @ r.T)) + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target_xy) ** 2, axis=1))))
    print(
        f"localization display alignment mode={args.align_mode} "
        f"yaw_deg={math.degrees(yaw):.2f} scale={scale:.4f} rmse_xy={rmse:.3f}m",
        flush=True,
    )
    return {"rotation": r, "translation": translation, "scale": scale, "yaw": yaw}


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def multiply_quaternion(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def apply_alignment(msg: Odometry, alignment: dict[str, object] | None) -> None:
    if alignment is None:
        return
    p = msg.pose.pose.position
    xy = np.array([float(p.x), float(p.y)], dtype=float)
    r = alignment["rotation"]
    t = alignment["translation"]
    scale = float(alignment["scale"])
    out = scale * (r @ xy) + t
    p.x = float(out[0])
    p.y = float(out[1])
    p.z = float(p.z) + float(alignment.get("translation_z", 0.0))

    q = msg.pose.pose.orientation
    qx, qy, qz, qw = multiply_quaternion(
        yaw_quaternion(float(alignment["yaw"])),
        (q.x, q.y, q.z, q.w),
    )
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm > 1e-9:
        q.x = qx / norm
        q.y = qy / norm
        q.z = qz / norm
        q.w = qw / norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an odometry topic from a rosbag2 sqlite DB as odometry+path.")
    parser.add_argument("--bag-db", required=True)
    parser.add_argument("--source-topic", default="/odometry/filtered")
    parser.add_argument("--odom-topic", default="/localization/odometry")
    parser.add_argument("--path-topic", default="/localization/path")
    parser.add_argument("--frame-id", default="world")
    parser.add_argument("--child-frame-id", default="fcu_link")
    parser.add_argument("--time-source", choices=["header", "db"], default="header")
    parser.add_argument("--start-offset-sec", type=float, default=0.0)
    parser.add_argument("--start-stamp-ns", type=int, default=0)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--loop-pause-sec", type=float, default=1.0)
    parser.add_argument("--path-stride-sec", type=float, default=0.05)
    parser.add_argument("--max-poses", type=int, default=5000)
    parser.add_argument("--publish-empty-on-start", action="store_true")
    parser.add_argument("--restamp-now", action="store_true")
    parser.add_argument("--align-csv", default="")
    parser.add_argument("--align-csv-coordinate-frame", choices=("ros", "opencv", "raw", "ned", "flip-y"), default="flip-y")
    parser.add_argument("--align-mode", choices=("none", "origin", "rigid", "similarity", "start_yaw"), default="none")
    parser.add_argument("--align-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()
    args.path_stride_ns = int(args.path_stride_sec * 1e9)
    return args


def main() -> None:
    args = parse_args()
    rows = read_rows(args)
    rclpy.init()
    node = BagOdometryPathPublisher(args, rows)
    try:
        node.run()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
