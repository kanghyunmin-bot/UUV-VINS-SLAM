#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    image0_dir = output_dir / "image_0"
    image1_dir = output_dir / "image_1"
    times_path = output_dir / "times.txt"

    if output_dir.exists() and not args.force:
        existing = list(output_dir.glob("*"))
        if existing:
            raise SystemExit(f"{output_dir} already exists. Pass --force to overwrite it.")
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    image0_dir.mkdir(parents=True, exist_ok=True)
    image1_dir.mkdir(parents=True, exist_ok=True)

    left = cv2.VideoCapture(str(args.left_video))
    right = cv2.VideoCapture(str(args.right_video))
    if not left.isOpened():
        raise SystemExit(f"Could not open left video: {args.left_video}")
    if not right.isOpened():
        raise SystemExit(f"Could not open right video: {args.right_video}")

    left_fps = valid_fps(left.get(cv2.CAP_PROP_FPS), args.fallback_fps)
    right_fps = valid_fps(right.get(cv2.CAP_PROP_FPS), args.fallback_fps)
    export_fps = args.fps if args.fps > 0 else min(left_fps, right_fps)

    if args.start_frame > 0:
        left.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        right.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    frame_idx = 0
    exported = 0
    width = 0
    height = 0
    times: list[float] = []
    try:
        while True:
            ok_left, left_frame = left.read()
            ok_right, right_frame = right.read()
            if not ok_left or not ok_right:
                break
            if args.stride > 1 and frame_idx % args.stride != 0:
                frame_idx += 1
                continue
            if args.max_frames > 0 and exported >= args.max_frames:
                break

            left_gray = to_gray(left_frame)
            right_gray = to_gray(right_frame)
            left_gray, right_gray = resize_pair(left_gray, right_gray, args.resize_width)
            height, width = left_gray.shape[:2]

            name = f"{exported:06d}.png"
            cv2.imwrite(str(image0_dir / name), left_gray)
            cv2.imwrite(str(image1_dir / name), right_gray)
            times.append(exported / export_fps)
            exported += 1
            frame_idx += 1
    finally:
        left.release()
        right.release()

    if exported == 0:
        raise SystemExit("No stereo frames were exported.")

    times_path.write_text("".join(f"{stamp:.9f}\n" for stamp in times), encoding="utf-8")
    metadata = {
        "format": "vins_fusion_kitti_stereo",
        "left_video": str(args.left_video),
        "right_video": str(args.right_video),
        "output_dir": str(output_dir),
        "image0_dir": str(image0_dir),
        "image1_dir": str(image1_dir),
        "times_txt": str(times_path),
        "frames": exported,
        "fps": export_fps,
        "duration_sec": times[-1] if len(times) == 1 else times[-1] + (1.0 / export_fps),
        "width": width,
        "height": height,
        "start_frame": args.start_frame,
        "stride": args.stride,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export left/right underwater MP4s as a VINS-Fusion KITTI stereo sequence."
    )
    parser.add_argument(
        "--left-video",
        type=Path,
        default=PROJECT_ROOT / "video_src" / "camera_camera_infra1_image_rect_raw.mp4",
    )
    parser.add_argument(
        "--right-video",
        type=Path,
        default=PROJECT_ROOT / "video_src" / "camera_camera_infra2_image_rect_raw.mp4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "underwater_stereo_kitti",
    )
    parser.add_argument("--max-frames", type=int, default=300, help="0 means export until video end.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resize-width", type=int, default=848, help="0 keeps the original width.")
    parser.add_argument("--fps", type=float, default=0.0, help="Override timestamps written to times.txt.")
    parser.add_argument("--fallback-fps", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def valid_fps(value: float, fallback: float) -> float:
    return value if value and value > 0 else fallback


def to_gray(frame):
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def resize_pair(left, right, resize_width: int):
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    if resize_width <= 0 or left.shape[1] == resize_width:
        return left, right
    scale = resize_width / left.shape[1]
    size = (resize_width, max(1, int(round(left.shape[0] * scale))))
    return (
        cv2.resize(left, size, interpolation=cv2.INTER_AREA),
        cv2.resize(right, size, interpolation=cv2.INTER_AREA),
    )


if __name__ == "__main__":
    main()
