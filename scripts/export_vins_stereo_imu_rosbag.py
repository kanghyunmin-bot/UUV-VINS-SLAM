#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image, Imu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BAG = (
    Path.home()
    / "Desktop"
    / "uuv_sim"
    / "real_robot_ros_bag"
    / "extracted_2026_04_01"
    / "bag_2026-04-01_20-20-30"
    / "bag_2026-04-01_20-20-30_0.db3"
)


@dataclass(frozen=True)
class MessageRef:
    rowid: int
    db_timestamp_ns: int
    stamp_ns: int


@dataclass(frozen=True)
class StereoPair:
    left: MessageRef
    right: MessageRef
    delta_ns: int


@dataclass(frozen=True)
class ImuSample:
    stamp_ns: int
    x: float
    y: float
    z: float


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output directory already exists: {args.output_dir}. Use --force to overwrite.")
        shutil.rmtree(args.output_dir)
    image0_dir = args.output_dir / "image_0"
    image1_dir = args.output_dir / "image_1"
    image0_dir.mkdir(parents=True, exist_ok=True)
    image1_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(args.bag))
    try:
        topic_ids = read_topic_ids(conn)
        left_id = require_topic(topic_ids, args.left_topic)
        right_id = require_topic(topic_ids, args.right_topic)
        left_info_id = require_topic(topic_ids, args.left_info_topic)
        right_info_id = require_topic(topic_ids, args.right_info_topic)

        left_info = read_first_message(conn, left_info_id, CameraInfo)
        right_info = read_first_message(conn, right_info_id, CameraInfo)

        left_refs = read_image_refs(conn, left_id, args.time_source)
        right_refs = read_image_refs(conn, right_id, args.time_source)
        pairs = match_pairs(left_refs, right_refs, int(args.sync_tolerance_ms * 1_000_000))

        if args.imu_source == "realsense":
            gyro_id = require_topic(topic_ids, args.gyro_topic)
            accel_id = require_topic(topic_ids, args.accel_topic)
            gyro = read_imu_axis(conn, gyro_id, "gyro", args.time_source)
            accel = read_imu_axis(conn, accel_id, "accel", args.time_source)
            imu_rows = fuse_realsense_imu(gyro, accel)
            imu_topics = {"gyro": args.gyro_topic, "accel": args.accel_topic}
        else:
            imu_id = require_topic(topic_ids, args.mavros_imu_topic)
            imu_rows = read_combined_imu(conn, imu_id, args.time_source)
            imu_topics = {"imu": args.mavros_imu_topic}

        if not imu_rows:
            raise SystemExit("No IMU rows found.")
        raw_imu_samples = len(imu_rows)
        if args.resample_imu_hz > 0.0:
            imu_rows = resample_imu_rows(imu_rows, args.resample_imu_hz)
            if not imu_rows:
                raise SystemExit("IMU resampling produced no rows.")

        first_imu_ns = imu_rows[0][0]
        last_imu_ns = imu_rows[-1][0]
        required_imu_lead_ns = int(args.required_imu_lead_ms * 1_000_000)
        pairs = [
            pair
            for pair in pairs
            if first_imu_ns <= pair.left.stamp_ns <= last_imu_ns - required_imu_lead_ns
        ]
        if not pairs:
            raise SystemExit("No synchronized stereo pairs overlap usable IMU coverage.")
        if args.start_frame:
            pairs = pairs[args.start_frame :]
        if args.stride > 1:
            pairs = pairs[:: args.stride]
        if args.max_frames > 0:
            pairs = pairs[: args.max_frames]
        if not pairs:
            raise SystemExit("No stereo pairs remain after IMU/start/max-frame filtering.")

        origin_ns = first_imu_ns if args.origin_source == "first_imu" else pairs[0].left.stamp_ns
        last_image_ns = pairs[-1].left.stamp_ns
        imu_end_ns = last_image_ns + int(args.imu_tail_ms * 1_000_000)
        imu_rows = [row for row in imu_rows if origin_ns <= row[0] <= imu_end_ns]
        if not imu_rows:
            raise SystemExit("No IMU rows overlap the selected image window.")

        times = []
        for index, pair in enumerate(pairs):
            left_msg = read_message_by_rowid(conn, pair.left.rowid, Image)
            right_msg = read_message_by_rowid(conn, pair.right.rowid, Image)
            left_image = image_to_gray(left_msg)
            right_image = image_to_gray(right_msg)
            cv2.imwrite(str(image0_dir / f"{index:06d}.png"), left_image)
            cv2.imwrite(str(image1_dir / f"{index:06d}.png"), right_image)
            times.append((pair.left.stamp_ns - origin_ns) / 1e9)

        write_times(args.output_dir / "times.txt", times)
        write_imu_csv(args.output_dir / "imu.csv", imu_rows, origin_ns)

        baseline = baseline_from_camera_info(right_info)
        fps = estimate_fps(times)
        metadata = {
            "format": "vins_fusion_rosbag_stereo_imu",
            "bag": str(args.bag),
            "output_dir": str(args.output_dir),
            "image0_dir": str(image0_dir),
            "image1_dir": str(image1_dir),
            "times_txt": str(args.output_dir / "times.txt"),
            "imu_csv": str(args.output_dir / "imu.csv"),
            "left_topic": args.left_topic,
            "right_topic": args.right_topic,
            "left_info_topic": args.left_info_topic,
            "right_info_topic": args.right_info_topic,
            "imu_source": args.imu_source,
            "imu_topics": imu_topics,
            "time_source": args.time_source,
            "raw_imu_samples": raw_imu_samples,
            "resample_imu_hz": args.resample_imu_hz,
            "frames": len(times),
            "imu_samples": len(imu_rows),
            "fps": fps,
            "duration_sec": times[-1] - times[0] if len(times) > 1 else 0.0,
            "width": int(left_info.width),
            "height": int(left_info.height),
            "start_frame": args.start_frame,
            "stride": args.stride,
            "sync_tolerance_ms": args.sync_tolerance_ms,
            "max_stereo_delta_ms": max(abs(pair.delta_ns) for pair in pairs) / 1e6,
            "origin_stamp_ns": origin_ns,
            "first_image_stamp_ns": pairs[0].left.stamp_ns,
            "last_image_stamp_ns": last_image_ns,
            "last_imu_stamp_ns": imu_rows[-1][0],
            "imu_tail_ms": args.imu_tail_ms,
            "baseline_m": baseline,
            "camera0": camera_info_to_dict(left_info),
            "camera1": camera_info_to_dict(right_info),
        }
        metadata_path = args.output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export synchronized stereo images and fused IMU samples from a ROS2 bag for VINS-Fusion."
    )
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "underwater_stereo_imu_rosbag",
    )
    parser.add_argument("--left-topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--right-topic", default="/camera/camera/infra2/image_rect_raw")
    parser.add_argument("--left-info-topic", default="/camera/camera/infra1/camera_info")
    parser.add_argument("--right-info-topic", default="/camera/camera/infra2/camera_info")
    parser.add_argument("--gyro-topic", default="/camera/camera/gyro/sample")
    parser.add_argument("--accel-topic", default="/camera/camera/accel/sample")
    parser.add_argument("--mavros-imu-topic", default="/mavros/imu/data")
    parser.add_argument("--imu-source", choices=["realsense", "mavros"], default="realsense")
    parser.add_argument(
        "--time-source",
        choices=["header", "db"],
        default="header",
        help="Use ROS message header stamps or rosbag DB timestamps for exported image/IMU timing.",
    )
    parser.add_argument(
        "--resample-imu-hz",
        type=float,
        default=0.0,
        help="0 keeps original IMU stamps. Positive values interpolate IMU onto a fixed-rate grid.",
    )
    parser.add_argument(
        "--origin-source",
        choices=("first_imu", "first_image"),
        default="first_imu",
        help="Timestamp origin for exported image/IMU CSVs. first_image keeps camera time near zero for DVL overlay scoring.",
    )
    parser.add_argument("--sync-tolerance-ms", type=float, default=3.0)
    parser.add_argument("--imu-tail-ms", type=float, default=80.0)
    parser.add_argument("--required-imu-lead-ms", type=float, default=35.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=120, help="0 exports every available frame.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_topic_ids(conn: sqlite3.Connection) -> dict[str, int]:
    return {name: topic_id for topic_id, name in conn.execute("select id, name from topics")}


def require_topic(topic_ids: dict[str, int], topic: str) -> int:
    if topic not in topic_ids:
        raise SystemExit(f"Missing topic in rosbag: {topic}")
    return topic_ids[topic]


def read_first_message(conn: sqlite3.Connection, topic_id: int, message_type):
    row = conn.execute(
        "select data from messages where topic_id = ? order by timestamp limit 1",
        (topic_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"No messages for topic_id={topic_id}")
    return deserialize_message(row[0], message_type)


def read_message_by_rowid(conn: sqlite3.Connection, rowid: int, message_type):
    row = conn.execute("select data from messages where rowid = ?", (rowid,)).fetchone()
    if row is None:
        raise SystemExit(f"Missing message rowid={rowid}")
    return deserialize_message(row[0], message_type)


def read_image_refs(conn: sqlite3.Connection, topic_id: int, time_source: str) -> list[MessageRef]:
    refs = []
    rows = conn.execute(
        "select rowid, timestamp, data from messages where topic_id = ? order by timestamp",
        (topic_id,),
    )
    for rowid, db_timestamp_ns, data in rows:
        msg = deserialize_message(data, Image)
        stamp_ns = db_timestamp_ns if time_source == "db" else stamp_to_ns(msg.header.stamp)
        refs.append(MessageRef(rowid, db_timestamp_ns, stamp_ns))
    return refs


def match_pairs(left_refs: list[MessageRef], right_refs: list[MessageRef], tolerance_ns: int) -> list[StereoPair]:
    right_stamps = [ref.stamp_ns for ref in right_refs]
    pairs: list[StereoPair] = []
    used_right: set[int] = set()
    for left in left_refs:
        insertion = bisect.bisect_left(right_stamps, left.stamp_ns)
        candidates = []
        if insertion < len(right_refs):
            candidates.append(insertion)
        if insertion > 0:
            candidates.append(insertion - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: abs(right_refs[idx].stamp_ns - left.stamp_ns))
        delta_ns = right_refs[best].stamp_ns - left.stamp_ns
        if abs(delta_ns) <= tolerance_ns and best not in used_right:
            used_right.add(best)
            pairs.append(StereoPair(left, right_refs[best], delta_ns))
    return pairs


def read_imu_axis(conn: sqlite3.Connection, topic_id: int, axis: str, time_source: str) -> list[ImuSample]:
    samples = []
    rows = conn.execute(
        "select timestamp, data from messages where topic_id = ? order by timestamp",
        (topic_id,),
    )
    for db_timestamp_ns, data in rows:
        msg = deserialize_message(data, Imu)
        stamp_ns = db_timestamp_ns if time_source == "db" else stamp_to_ns(msg.header.stamp)
        if axis == "gyro":
            value = msg.angular_velocity
        else:
            value = msg.linear_acceleration
        samples.append(ImuSample(stamp_ns, float(value.x), float(value.y), float(value.z)))
    return samples


def fuse_realsense_imu(gyro: list[ImuSample], accel: list[ImuSample]) -> list[tuple[int, float, float, float, float, float, float]]:
    if not gyro or not accel:
        return []
    accel_times = [sample.stamp_ns for sample in accel]
    rows = []
    for gyr in gyro:
        acc = interpolate_accel(gyr.stamp_ns, accel, accel_times)
        if acc is None:
            continue
        rows.append((gyr.stamp_ns, acc.x, acc.y, acc.z, gyr.x, gyr.y, gyr.z))
    return rows


def resample_imu_rows(
    rows: list[tuple[float, ...]],
    hz: float,
) -> list[tuple[float, ...]]:
    if len(rows) < 2:
        return rows
    rows = sorted(rows, key=lambda row: row[0])
    stamps = np.array([row[0] for row in rows], dtype=np.float64)
    values = np.array([row[1:] for row in rows], dtype=np.float64)
    has_orientation = values.shape[1] >= 10
    if has_orientation:
        # Keep quaternion interpolation on the same hemisphere before lerp+normalize.
        for index in range(1, len(values)):
            if float(np.dot(values[index - 1, 6:10], values[index, 6:10])) < 0.0:
                values[index, 6:10] *= -1.0
    step_ns = int(round(1_000_000_000.0 / hz))
    if step_ns <= 0:
        raise SystemExit("--resample-imu-hz must be positive when enabled.")
    target_stamps = np.arange(rows[0][0], rows[-1][0] + 1, step_ns, dtype=np.int64)
    if len(target_stamps) < 2:
        return rows
    out_values = np.empty((len(target_stamps), values.shape[1]), dtype=np.float64)
    for axis in range(values.shape[1]):
        out_values[:, axis] = np.interp(target_stamps.astype(np.float64), stamps, values[:, axis])
    if has_orientation:
        norms = np.linalg.norm(out_values[:, 6:10], axis=1)
        valid = norms > 1e-12
        out_values[valid, 6:10] /= norms[valid, None]
        out_values[~valid, 6:10] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return [tuple([int(stamp), *[float(value) for value in out_values[index]]]) for index, stamp in enumerate(target_stamps)]


def interpolate_accel(stamp_ns: int, accel: list[ImuSample], accel_times: list[int]) -> ImuSample | None:
    index = bisect.bisect_left(accel_times, stamp_ns)
    if index == 0:
        return accel[0] if abs(accel[0].stamp_ns - stamp_ns) <= 25_000_000 else None
    if index >= len(accel):
        return accel[-1] if abs(accel[-1].stamp_ns - stamp_ns) <= 25_000_000 else None
    before = accel[index - 1]
    after = accel[index]
    span = after.stamp_ns - before.stamp_ns
    if span <= 0 or min(abs(stamp_ns - before.stamp_ns), abs(after.stamp_ns - stamp_ns)) > 25_000_000:
        return None
    ratio = (stamp_ns - before.stamp_ns) / span
    return ImuSample(
        stamp_ns,
        before.x + (after.x - before.x) * ratio,
        before.y + (after.y - before.y) * ratio,
        before.z + (after.z - before.z) * ratio,
    )


def read_combined_imu(
    conn: sqlite3.Connection,
    topic_id: int,
    time_source: str,
) -> list[tuple[float, ...]]:
    rows = []
    for db_timestamp_ns, data in conn.execute(
        "select timestamp, data from messages where topic_id = ? order by timestamp",
        (topic_id,),
    ):
        msg = deserialize_message(data, Imu)
        stamp_ns = db_timestamp_ns if time_source == "db" else stamp_to_ns(msg.header.stamp)
        rows.append(
            (
                stamp_ns,
                float(msg.linear_acceleration.x),
                float(msg.linear_acceleration.y),
                float(msg.linear_acceleration.z),
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
                float(msg.orientation.x),
                float(msg.orientation.y),
                float(msg.orientation.z),
                float(msg.orientation.w),
            )
        )
    return rows


def image_to_gray(msg: Image) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    encoding = msg.encoding.lower()
    if encoding in {"mono8", "8uc1"}:
        array = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, int(msg.step))
        return array[:, :width].copy()
    if encoding in {"bgr8", "rgb8"}:
        array = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
        code = cv2.COLOR_BGR2GRAY if encoding == "bgr8" else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(array, code)
    if encoding in {"bgra8", "rgba8"}:
        array = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 4)
        code = cv2.COLOR_BGRA2GRAY if encoding == "bgra8" else cv2.COLOR_RGBA2GRAY
        return cv2.cvtColor(array, code)
    if encoding in {"mono16", "16uc1"}:
        array = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, int(msg.step) // 2)
        cropped = array[:, :width]
        return cv2.convertScaleAbs(cropped, alpha=255.0 / max(float(cropped.max()), 1.0))
    raise SystemExit(f"Unsupported image encoding: {msg.encoding}")


def write_times(path: Path, times: list[float]) -> None:
    path.write_text("".join(f"{value:.9f}\n" for value in times), encoding="utf-8")


def write_imu_csv(path: Path, rows: list[tuple[float, ...]], origin_ns: int) -> None:
    has_orientation = any(len(row) >= 11 for row in rows)
    with path.open("w", encoding="utf-8") as file:
        if has_orientation:
            file.write("timestamp_sec,ax,ay,az,gx,gy,gz,qx,qy,qz,qw\n")
        else:
            file.write("timestamp_sec,ax,ay,az,gx,gy,gz\n")
        for row in rows:
            stamp_ns, ax, ay, az, gx, gy, gz = row[:7]
            t = (stamp_ns - origin_ns) / 1e9
            if has_orientation and len(row) >= 11:
                qx, qy, qz, qw = row[7:11]
                file.write(
                    f"{t:.9f},{ax:.9f},{ay:.9f},{az:.9f},{gx:.9f},{gy:.9f},{gz:.9f},"
                    f"{qx:.9f},{qy:.9f},{qz:.9f},{qw:.9f}\n"
                )
            else:
                file.write(f"{t:.9f},{ax:.9f},{ay:.9f},{az:.9f},{gx:.9f},{gy:.9f},{gz:.9f}\n")


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def baseline_from_camera_info(info: CameraInfo) -> float:
    fx = float(info.p[0]) if info.p[0] else float(info.k[0])
    tx = float(info.p[3])
    if fx == 0:
        return 0.05
    return abs(tx / fx)


def camera_info_to_dict(info: CameraInfo) -> dict[str, object]:
    return {
        "width": int(info.width),
        "height": int(info.height),
        "k": [float(value) for value in info.k],
        "d": [float(value) for value in info.d],
        "r": [float(value) for value in info.r],
        "p": [float(value) for value in info.p],
    }


def estimate_fps(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    duration = times[-1] - times[0]
    if duration <= 0:
        return 0.0
    return (len(times) - 1) / duration


if __name__ == "__main__":
    main()
