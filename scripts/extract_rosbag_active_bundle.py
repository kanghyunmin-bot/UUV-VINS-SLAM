#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import shutil
import sqlite3
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, Imu

try:
    from dvl_msgs.msg import DVLDR
except ImportError as exc:  # pragma: no cover - depends on sourced ROS overlay.
    DVLDR = None
    DVLDR_IMPORT_ERROR = exc
else:
    DVLDR_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = Path.home() / "Downloads" / "bag_2026-04-02_21-46-20.zip"
DEFAULT_IMPORT_DIR = PROJECT_ROOT / "data" / "rosbag_imports" / "bag_2026-04-02_21-46-20"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "rosbag_active" / "localization bag"


@dataclass(frozen=True)
class MessageRef:
    rowid: int
    db_stamp_ns: int
    stamp_ns: int


@dataclass(frozen=True)
class StereoPair:
    left: MessageRef
    right: MessageRef
    delta_ns: int


def main() -> None:
    args = parse_args()
    imported = ensure_imported_bag(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(imported["db3"]))
    try:
        topic_ids = read_topic_ids(conn)
        left_id = require_topic(topic_ids, args.left_topic)
        right_id = require_topic(topic_ids, args.right_topic)
        dvl_id = require_topic(topic_ids, args.dvl_topic)
        odom_id = require_topic(topic_ids, args.localization_odom_topic)
        path_id = topic_ids.get(args.localization_path_topic)
        imu_id = topic_ids.get(args.imu_topic)

        left_refs = read_image_refs(conn, left_id, args.time_source)
        right_refs = read_image_refs(conn, right_id, args.time_source)
        pairs = match_pairs(left_refs, right_refs, int(args.sync_tolerance_ms * 1_000_000))
        if not pairs:
            raise SystemExit("No synchronized stereo pairs found.")

        origin_ns = pairs[0].left.stamp_ns + int(args.start_offset_sec * 1_000_000_000)
        end_ns = origin_ns + int(args.duration_sec * 1_000_000_000)
        pairs = [pair for pair in pairs if origin_ns <= pair.left.stamp_ns <= end_ns]
        if len(pairs) < 2:
            raise SystemExit("Too few synchronized stereo pairs inside selected window.")

        output_tag = args.output_tag or duration_tag(args.start_offset_sec, args.duration_sec)
        stereo_schedule_name = f"stereo_schedule_{output_tag}.csv"
        stereo_schedule_swapped_name = f"stereo_schedule_swapped_{output_tag}.csv"
        dvl_csv_name = f"dvl_reference_{output_tag}.csv"
        localization_odom_csv_name = f"localization_odometry_{output_tag}.csv"
        localization_path_csv_name = f"localization_path_{output_tag}.csv"
        imu_csv_name = f"imu_{output_tag}.csv"

        write_stereo_schedule(output_dir / stereo_schedule_name, pairs, swapped=False)
        write_stereo_schedule(output_dir / stereo_schedule_swapped_name, pairs, swapped=True)
        video_meta = export_stereo_mp4(conn, pairs, output_dir, args)
        dvl_meta = export_dvl_csv(conn, dvl_id, output_dir / dvl_csv_name, origin_ns, end_ns, args)
        odom_meta = export_odom_csv(
            conn,
            odom_id,
            output_dir / localization_odom_csv_name,
            origin_ns,
            end_ns,
            args,
        )
        path_meta = (
            export_path_csv(
                conn,
                path_id,
                output_dir / localization_path_csv_name,
                origin_ns,
                end_ns,
                args,
            )
            if path_id is not None
            else {"rows": 0, "topic": args.localization_path_topic, "available": False}
        )
        imu_meta = (
            export_imu_csv(conn, imu_id, output_dir / imu_csv_name, origin_ns, end_ns, args)
            if imu_id is not None
            else {"rows": 0, "topic": args.imu_topic, "available": False}
        )

        link_active_bag_files(imported, output_dir)
        manifest = {
            "format": "underwater_active_rosbag_bundle",
            "source_zip": str(args.zip),
            "source_bag_dir": str(imported["bag_dir"]),
            "source_db3": str(imported["db3"]),
            "source_metadata": str(imported["metadata"]),
            "output_dir": str(output_dir),
            "time_source": args.time_source,
            "origin_stamp_ns": origin_ns,
            "duration_sec": args.duration_sec,
            "start_offset_sec": args.start_offset_sec,
            "topics": {
                "left": args.left_topic,
                "right": args.right_topic,
                "dvl_reference": args.dvl_topic,
                "localization_odom": args.localization_odom_topic,
                "localization_path": args.localization_path_topic,
                "imu": args.imu_topic,
            },
            "outputs": {
                "stamp_bag": "stamp_bag.db3",
                "metadata": "metadata.yaml",
                "left_mp4": "infra1_image_rect_raw.mp4",
                "right_mp4": "infra2_image_rect_raw.mp4",
                "stereo_schedule_csv": stereo_schedule_name,
                "stereo_schedule_swapped_csv": stereo_schedule_swapped_name,
                "dvl_reference_csv": dvl_csv_name,
                "localization_odometry_csv": localization_odom_csv_name,
                "localization_path_csv": localization_path_csv_name if path_meta["rows"] else None,
                "imu_csv": imu_csv_name if imu_meta["rows"] else None,
            },
            "video": video_meta,
            "dvl_reference": dvl_meta,
            "localization_odometry": odom_meta,
            "localization_path": path_meta,
            "imu": imu_meta,
            "notes": [
                "DVL CSV is exported as a fixed reference only.",
                "Stereo MP4 files are exported from the original ROS2 Image topics and stay synchronized by the source bag stamps.",
                "The original bag database is symlinked, not copied, so the bundle is lightweight.",
            ],
        }
        (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "MANIFEST.txt").write_text(render_text_manifest(manifest), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract active VINS/DVL/localization files from a ROS2 bag zip.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--import-dir", type=Path, default=DEFAULT_IMPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--left-topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--right-topic", default="/camera/camera/infra2/image_rect_raw")
    parser.add_argument("--dvl-topic", default="/dvl/position")
    parser.add_argument("--localization-odom-topic", default="/odometry/filtered")
    parser.add_argument("--localization-path-topic", default="/localization/path")
    parser.add_argument("--imu-topic", default="/mavros/imu/data")
    parser.add_argument("--time-source", choices=("db", "header"), default="header")
    parser.add_argument("--duration-sec", type=float, default=51.0)
    parser.add_argument("--start-offset-sec", type=float, default=34.0)
    parser.add_argument("--output-tag", default="34_85s")
    parser.add_argument("--sync-tolerance-ms", type=float, default=3.0)
    parser.add_argument("--left-output-name", default="infra1_image_rect_raw.mp4")
    parser.add_argument("--right-output-name", default="infra2_image_rect_raw.mp4")
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--force-output", action="store_true")
    return parser.parse_args()


def ensure_imported_bag(args: argparse.Namespace) -> dict[str, Path]:
    zip_path = args.zip.expanduser().resolve()
    import_dir = args.import_dir.expanduser().resolve()
    import_dir.parent.mkdir(parents=True, exist_ok=True)

    metadata = import_dir / "metadata.yaml"
    zstd_path = import_dir / f"{import_dir.name}_0.db3.zstd"
    db3_path = import_dir / f"{import_dir.name}_0.db3"

    if metadata.exists() and db3_path.exists() and not args.force_import:
        return {"bag_dir": import_dir, "metadata": metadata, "db3": db3_path}

    if not zip_path.exists():
        raise SystemExit(f"Missing bag zip and no usable imported DB: {zip_path}")

    if args.force_import and import_dir.exists():
        shutil.rmtree(import_dir)

    if not metadata.exists() or not zstd_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.startswith(f"{import_dir.name}/") and not name.endswith("/")
            ]
            if not members:
                raise SystemExit(f"Zip does not contain {import_dir.name}/")
            archive.extractall(import_dir.parent, members)

    if not db3_path.exists() or args.force_import:
        subprocess.run(["zstd", "-d", "-f", str(zstd_path), "-o", str(db3_path)], check=True)

    if not metadata.exists() or not db3_path.exists():
        raise SystemExit(f"Bag import failed under {import_dir}")
    return {"bag_dir": import_dir, "metadata": metadata, "db3": db3_path}


def read_topic_ids(conn: sqlite3.Connection) -> dict[str, int]:
    return {name: int(topic_id) for topic_id, name in conn.execute("select id, name from topics")}


def require_topic(topic_ids: dict[str, int], topic: str) -> int:
    if topic not in topic_ids:
        raise SystemExit(f"Missing topic in rosbag: {topic}")
    return topic_ids[topic]


def read_image_refs(conn: sqlite3.Connection, topic_id: int, time_source: str) -> list[MessageRef]:
    refs: list[MessageRef] = []
    for rowid, db_stamp_ns, data in conn.execute(
        "select rowid, timestamp, data from messages where topic_id = ? order by timestamp",
        (topic_id,),
    ):
        msg = deserialize_message(data, Image)
        stamp_ns = int(db_stamp_ns) if time_source == "db" else stamp_to_ns(msg.header.stamp)
        refs.append(MessageRef(int(rowid), int(db_stamp_ns), stamp_ns))
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
            pairs.append(StereoPair(left, right_refs[best], int(delta_ns)))
    return pairs


def export_stereo_mp4(
    conn: sqlite3.Connection,
    pairs: list[StereoPair],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.force_output:
        for name in (args.left_output_name, args.right_output_name):
            path = output_dir / name
            if path.exists() or path.is_symlink():
                path.unlink()

    first_left = read_message_by_rowid(conn, pairs[0].left.rowid, Image)
    first_right = read_message_by_rowid(conn, pairs[0].right.rowid, Image)
    left_first_image = image_to_gray(first_left)
    right_first_image = image_to_gray(first_right)
    fps = estimate_fps([(pair.left.stamp_ns - pairs[0].left.stamp_ns) / 1e9 for pair in pairs])
    fps = max(1.0, min(60.0, fps))
    fourcc = cv2.VideoWriter_fourcc(*args.video_codec[:4])
    left_path = output_dir / args.left_output_name
    right_path = output_dir / args.right_output_name

    left_writer = cv2.VideoWriter(str(left_path), fourcc, fps, (left_first_image.shape[1], left_first_image.shape[0]), True)
    right_writer = cv2.VideoWriter(str(right_path), fourcc, fps, (right_first_image.shape[1], right_first_image.shape[0]), True)
    if not left_writer.isOpened() or not right_writer.isOpened():
        raise SystemExit("Failed to open MP4 writers.")

    try:
        for index, pair in enumerate(pairs):
            if index == 0:
                left_image = left_first_image
                right_image = right_first_image
            else:
                left_image = image_to_gray(read_message_by_rowid(conn, pair.left.rowid, Image))
                right_image = image_to_gray(read_message_by_rowid(conn, pair.right.rowid, Image))
            left_writer.write(cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR))
            right_writer.write(cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR))
    finally:
        left_writer.release()
        right_writer.release()

    deltas_ms = [abs(pair.delta_ns) / 1e6 for pair in pairs]
    return {
        "frames": len(pairs),
        "fps": fps,
        "left_mp4": str(left_path),
        "right_mp4": str(right_path),
        "width": int(left_first_image.shape[1]),
        "height": int(left_first_image.shape[0]),
        "max_stereo_delta_ms": max(deltas_ms),
        "mean_stereo_delta_ms": float(np.mean(deltas_ms)),
    }


def write_stereo_schedule(path: Path, pairs: list[StereoPair], swapped: bool) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["left_ns", "right_ns"])
        for pair in pairs:
            if swapped:
                writer.writerow([pair.right.stamp_ns, pair.left.stamp_ns])
            else:
                writer.writerow([pair.left.stamp_ns, pair.right.stamp_ns])


def export_dvl_csv(
    conn: sqlite3.Connection,
    topic_id: int,
    output_csv: Path,
    origin_ns: int,
    end_ns: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if DVLDR is None:
        raise SystemExit(
            "dvl_msgs is required. Run with `source ros2_overlay_ws/install/setup.zsh` first."
        ) from DVLDR_IMPORT_ERROR

    rows = []
    origin_position: np.ndarray | None = None
    for db_stamp_ns, data in conn.execute(
        "select timestamp, data from messages where topic_id = ? order by timestamp",
        (topic_id,),
    ):
        msg = deserialize_message(data, DVLDR)
        stamp_ns = int(db_stamp_ns) if args.time_source == "db" else stamp_to_ns(msg.header.stamp)
        if stamp_ns < origin_ns or stamp_ns > end_ns:
            continue
        position = np.array([float(msg.position.x), float(msg.position.y), float(msg.position.z)], dtype=np.float64)
        if origin_position is None:
            origin_position = position.copy()
        rel = position - origin_position
        rows.append(
            {
                "timestamp_sec": f"{(stamp_ns - origin_ns) / 1e9:.9f}",
                "x": f"{rel[0]:.9f}",
                "y": f"{rel[1]:.9f}",
                "z": f"{rel[2]:.9f}",
                "qx": "0.000000000",
                "qy": "0.000000000",
                "qz": "0.000000000",
                "qw": "1.000000000",
                "roll": f"{float(msg.roll):.9f}",
                "pitch": f"{float(msg.pitch):.9f}",
                "yaw": f"{float(msg.yaw):.9f}",
            }
        )
    write_csv(output_csv, rows)
    return path_stats(rows, args.dvl_topic, output_csv)


def export_odom_csv(
    conn: sqlite3.Connection,
    topic_id: int,
    output_csv: Path,
    origin_ns: int,
    end_ns: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = []
    origin_position: np.ndarray | None = None
    for db_stamp_ns, data in conn.execute(
        "select timestamp, data from messages where topic_id = ? order by timestamp",
        (topic_id,),
    ):
        msg = deserialize_message(data, Odometry)
        stamp_ns = int(db_stamp_ns) if args.time_source == "db" else stamp_to_ns(msg.header.stamp)
        if stamp_ns < origin_ns or stamp_ns > end_ns:
            continue
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        position = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
        if origin_position is None:
            origin_position = position.copy()
        rel = position - origin_position
        rows.append(
            {
                "timestamp_sec": f"{(stamp_ns - origin_ns) / 1e9:.9f}",
                "x": f"{rel[0]:.9f}",
                "y": f"{rel[1]:.9f}",
                "z": f"{rel[2]:.9f}",
                "qx": f"{float(q.x):.9f}",
                "qy": f"{float(q.y):.9f}",
                "qz": f"{float(q.z):.9f}",
                "qw": f"{float(q.w):.9f}",
                "frame_id": msg.header.frame_id,
                "child_frame_id": msg.child_frame_id,
            }
        )
    write_csv(output_csv, rows)
    return path_stats(rows, args.localization_odom_topic, output_csv)


def export_path_csv(
    conn: sqlite3.Connection,
    topic_id: int | None,
    output_csv: Path,
    origin_ns: int,
    end_ns: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if topic_id is None:
        return {"rows": 0, "available": False, "topic": args.localization_path_topic}
    rows = []
    origin_position: np.ndarray | None = None
    for db_stamp_ns, data in conn.execute(
        "select timestamp, data from messages where topic_id = ? order by timestamp",
        (topic_id,),
    ):
        msg = deserialize_message(data, NavPath)
        stamp_ns = int(db_stamp_ns) if args.time_source == "db" else stamp_to_ns(msg.header.stamp)
        if stamp_ns < origin_ns or stamp_ns > end_ns or not msg.poses:
            continue
        pose = msg.poses[-1].pose
        p = pose.position
        q = pose.orientation
        position = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
        if origin_position is None:
            origin_position = position.copy()
        rel = position - origin_position
        rows.append(
            {
                "timestamp_sec": f"{(stamp_ns - origin_ns) / 1e9:.9f}",
                "x": f"{rel[0]:.9f}",
                "y": f"{rel[1]:.9f}",
                "z": f"{rel[2]:.9f}",
                "qx": f"{float(q.x):.9f}",
                "qy": f"{float(q.y):.9f}",
                "qz": f"{float(q.z):.9f}",
                "qw": f"{float(q.w):.9f}",
                "frame_id": msg.header.frame_id,
            }
        )
    write_csv(output_csv, rows)
    return path_stats(rows, args.localization_path_topic, output_csv)


def export_imu_csv(
    conn: sqlite3.Connection,
    topic_id: int,
    output_csv: Path,
    origin_ns: int,
    end_ns: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = []
    for db_stamp_ns, data in conn.execute(
        "select timestamp, data from messages where topic_id = ? order by timestamp",
        (topic_id,),
    ):
        msg = deserialize_message(data, Imu)
        stamp_ns = int(db_stamp_ns) if args.time_source == "db" else stamp_to_ns(msg.header.stamp)
        if stamp_ns < origin_ns or stamp_ns > end_ns:
            continue
        a = msg.linear_acceleration
        g = msg.angular_velocity
        q = msg.orientation
        rows.append(
            {
                "timestamp_sec": f"{(stamp_ns - origin_ns) / 1e9:.9f}",
                "ax": f"{float(a.x):.9f}",
                "ay": f"{float(a.y):.9f}",
                "az": f"{float(a.z):.9f}",
                "gx": f"{float(g.x):.9f}",
                "gy": f"{float(g.y):.9f}",
                "gz": f"{float(g.z):.9f}",
                "qx": f"{float(q.x):.9f}",
                "qy": f"{float(q.y):.9f}",
                "qz": f"{float(q.z):.9f}",
                "qw": f"{float(q.w):.9f}",
            }
        )
    write_csv(output_csv, rows)
    return {
        "rows": len(rows),
        "topic": args.imu_topic,
        "csv": str(output_csv),
        "available": bool(rows),
    }


def read_message_by_rowid(conn: sqlite3.Connection, rowid: int, message_type: Any) -> Any:
    row = conn.execute("select data from messages where rowid = ?", (int(rowid),)).fetchone()
    if row is None:
        raise SystemExit(f"Missing message rowid={rowid}")
    return deserialize_message(row[0], message_type)


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
        max_value = max(float(cropped.max()), 1.0)
        return cv2.convertScaleAbs(cropped, alpha=255.0 / max_value)
    raise SystemExit(f"Unsupported image encoding: {msg.encoding}")


def link_active_bag_files(imported: dict[str, Path], output_dir: Path) -> None:
    link_relative(imported["metadata"], output_dir / "metadata.yaml")
    link_relative(imported["db3"], output_dir / imported["db3"].name)
    link_relative(imported["db3"], output_dir / "stamp_bag.db3")


def link_relative(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    relative = os.path.relpath(source.resolve(), target.parent.resolve())
    target.symlink_to(relative)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def path_stats(rows: list[dict[str, str]], topic: str, csv_path: Path) -> dict[str, Any]:
    if len(rows) < 2:
        return {"rows": len(rows), "topic": topic, "csv": str(csv_path), "path_length_m": 0.0, "available": bool(rows)}
    positions = np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows], dtype=np.float64)
    length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    span = positions.max(axis=0) - positions.min(axis=0)
    return {
        "rows": len(rows),
        "topic": topic,
        "csv": str(csv_path),
        "path_length_m": length,
        "span_x_m": float(span[0]),
        "span_y_m": float(span[1]),
        "span_z_m": float(span[2]),
        "available": True,
    }


def render_text_manifest(manifest: dict[str, Any]) -> str:
    lines = [
        "Underwater active ROS bag bundle",
        f"source_zip: {manifest['source_zip']}",
        f"source_db3: {manifest['source_db3']}",
        f"origin_stamp_ns: {manifest['origin_stamp_ns']}",
        f"duration_sec: {manifest['duration_sec']}",
        "",
        "Outputs:",
    ]
    for key, value in manifest["outputs"].items():
        if value:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "Stats:"])
    lines.append(f"- stereo_frames: {manifest['video']['frames']}")
    lines.append(f"- stereo_fps: {manifest['video']['fps']:.6f}")
    lines.append(f"- dvl_rows: {manifest['dvl_reference']['rows']}")
    lines.append(f"- dvl_path_length_m: {manifest['dvl_reference']['path_length_m']:.6f}")
    lines.append(f"- localization_odom_rows: {manifest['localization_odometry']['rows']}")
    lines.append(f"- localization_odom_path_length_m: {manifest['localization_odometry']['path_length_m']:.6f}")
    lines.append("")
    lines.extend(manifest["notes"])
    return "\n".join(lines) + "\n"


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def estimate_fps(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    duration = times[-1] - times[0]
    if duration <= 0.0:
        return 0.0
    return float((len(times) - 1) / duration)


def duration_tag(start_offset_sec: float, duration_sec: float) -> str:
    start = compact_seconds(start_offset_sec)
    end = compact_seconds(start_offset_sec + duration_sec)
    return f"{start}_{end}s"


def compact_seconds(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


if __name__ == "__main__":
    main()
