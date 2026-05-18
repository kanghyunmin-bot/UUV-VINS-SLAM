#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENCV_OPTICAL_TO_ROS = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
QUAD_RIGHT_CLOSURE_THRESHOLD_PX = 5.0
QUAD_MIN_CANDIDATES = 12
QUAD_MIN_TRACKS_AFTER_REJECTION = 40


@dataclass
class Track:
    track_id: int
    left_point: np.ndarray
    point_3d: np.ndarray
    right_point: np.ndarray | None = None
    world_point: np.ndarray | None = None
    image_velocity: np.ndarray | None = None
    age: int = 1
    missing_depth_age: int = 0


@dataclass
class PoseRow:
    frame_index: int
    timestamp_sec: float
    success: bool
    position: np.ndarray
    rotation: np.ndarray
    pose_inliers: int
    pose_inlier_ratio: float
    active_tracks: int
    stereo_tracks: int
    note: str


@dataclass
class ImuRow:
    timestamp_sec: float
    gyro: np.ndarray
    orientation: np.ndarray | None = None


@dataclass
class UnderwaterPnpQuality:
    reprojection_rmse_px: float = float("inf")
    imu_rotation_error_rad: float = float("inf")
    stereo_confirmed: int = 0
    stereo_inlier_ratio: float = 0.0
    depth_median_abs_residual_m: float = float("inf")
    depth_p90_abs_residual_m: float = float("inf")
    xyz_median_residual_m: float = float("inf")
    score: float = float("inf")


@dataclass
class UnderwaterPnpContext:
    curr_left: np.ndarray
    curr_right: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float


@dataclass
class FrontendHealth:
    score: float
    state: str
    track_ratio: float
    inlier_ratio: float
    stereo_ratio: float
    quad_reject_ratio: float


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ROS2-native stereo+IMU VIO frontend. It uses only synchronized stereo images "
            "and IMU samples exported from a ROS2 bag; DVL is not consumed."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing image_0, image_1, times.txt, imu.csv, metadata.json.",
    )
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs" / "live" / "latest_ros2_stereo_imu_vio.csv")
    parser.add_argument("--point-cloud-csv", type=Path, default=PROJECT_ROOT / "outputs" / "live" / "latest_point_cloud.csv")
    parser.add_argument(
        "--raw-point-cloud-csv",
        type=Path,
        default=None,
        help="Optional CSV for raw active tracks before stable landmark filtering.",
    )
    parser.add_argument("--summary-json", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "ros2_stereo_imu_vio_summary.json")
    parser.add_argument("--debug-csv", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "ros2_stereo_imu_vio_debug.csv")
    parser.add_argument("--max-frames", type=int, default=0, help="0 uses every exported frame.")
    parser.add_argument(
        "--swap-stereo",
        action="store_true",
        help="Diagnostic mode: treat image_1 as left and image_0 as right.",
    )
    parser.add_argument("--max-features", type=int, default=320)
    parser.add_argument("--min-features", type=int, default=144)
    parser.add_argument("--quality", type=float, default=0.01)
    parser.add_argument("--min-distance", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply CLAHE to stereo images before feature tracking and stereo matching.",
    )
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid", type=int, default=8)
    parser.add_argument("--mask-bottom-ratio", type=float, default=0.0, help="Ignore this bottom image ratio for feature tracking.")
    parser.add_argument("--mask-bottom-px", type=int, default=0, help="Ignore this many bottom image pixels for feature tracking.")
    parser.add_argument("--mask-bright-threshold", type=int, default=0, help="0 disables saturated-region masking.")
    parser.add_argument("--mask-bright-dilate", type=int, default=9)
    parser.add_argument(
        "--feature-detector",
        choices=["gftt", "harris", "fast", "orb"],
        default="gftt",
        help="Feature detector used to seed KLT tracks. gftt is Shi-Tomasi.",
    )
    parser.add_argument("--fast-threshold", type=int, default=20)
    parser.add_argument("--orb-fast-threshold", type=int, default=20)
    parser.add_argument(
        "--reject-repeated-patches",
        action="store_true",
        help="Reject features whose local image patch is too similar to nearby patches.",
    )
    parser.add_argument("--patch-size", type=int, default=13)
    parser.add_argument("--patch-search-radius", type=int, default=80)
    parser.add_argument("--patch-search-stride", type=int, default=8)
    parser.add_argument("--patch-ncc-threshold", type=float, default=0.92)
    parser.add_argument("--patch-min-std", type=float, default=6.0)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=6)
    parser.add_argument("--cell-max-features", type=int, default=18)
    parser.add_argument(
        "--feature-refill-interval",
        type=int,
        default=0,
        help=(
            "0 keeps the legacy behavior: add new features only when active tracks drop below min_features. "
            "A positive value refills every N frames while there is room below max_features, which helps forward motion."
        ),
    )
    parser.add_argument("--klt-window", type=int, default=21)
    parser.add_argument("--klt-levels", type=int, default=3)
    parser.add_argument("--klt-iterations", type=int, default=30)
    parser.add_argument("--klt-eps", type=float, default=0.01)
    parser.add_argument("--fb-threshold", type=float, default=0.8)
    parser.add_argument("--klt-max-error", type=float, default=0.0, help="0 disables LK patch-error filtering for temporal tracking.")
    parser.add_argument("--klt-max-flow-px", type=float, default=0.0, help="0 disables max optical-flow length filtering.")
    parser.add_argument("--klt-min-eigenvalue", type=float, default=0.0, help="0 disables current-frame texture filtering.")
    parser.add_argument("--klt-patch-size", type=int, default=15)
    parser.add_argument("--klt-patch-ncc-threshold", type=float, default=0.0, help="0 disables patch NCC filtering.")
    parser.add_argument("--klt-local-flow-check", action="store_true")
    parser.add_argument("--klt-local-flow-mad-factor", type=float, default=3.5)
    parser.add_argument("--klt-local-flow-min-abs", type=float, default=2.5)
    parser.add_argument("--klt-local-flow-min-points", type=int, default=8)
    parser.add_argument("--klt-use-prediction", action="store_true")
    parser.add_argument("--klt-velocity-check", action="store_true")
    parser.add_argument("--klt-max-accel-px", type=float, default=25.0)
    parser.add_argument("--min-track-age-for-pnp", type=int, default=1)
    parser.add_argument(
        "--temporal-ransac",
        choices=["none", "essential", "fundamental"],
        default="none",
        help=(
            "Optional frame-to-frame geometry check after KLT and before PnP. "
            "This rejects repeated-pattern optical-flow matches without using reference odometry."
        ),
    )
    parser.add_argument("--temporal-ransac-threshold", type=float, default=1.0)
    parser.add_argument("--temporal-ransac-confidence", type=float, default=0.999)
    parser.add_argument("--min-temporal-inliers", type=int, default=24)
    parser.add_argument("--stereo-fb-threshold", type=float, default=1.2)
    parser.add_argument("--stereo-max-error", type=float, default=0.0, help="0 disables LK patch-error filtering for left-right stereo matching.")
    parser.add_argument("--epipolar-threshold", type=float, default=2.0)
    parser.add_argument(
        "--stereo-clahe-fallback-min-klt",
        type=int,
        default=0,
        help="If KLT stereo returns fewer valid depths than this, retry missing points on CLAHE-enhanced stereo images; 0 disables.",
    )
    parser.add_argument("--stereo-clahe-fallback-clip-limit", type=float, default=2.0)
    parser.add_argument("--stereo-clahe-fallback-tile-grid", type=int, default=8)
    parser.add_argument(
        "--stereo-ncc-fallback-min-klt",
        type=int,
        default=0,
        help="If KLT stereo still returns fewer valid depths than this, fill missing depths by sparse row NCC; 0 disables.",
    )
    parser.add_argument("--stereo-ncc-patch-size", type=int, default=11)
    parser.add_argument("--stereo-ncc-min-score", type=float, default=0.86)
    parser.add_argument("--stereo-ncc-min-margin", type=float, default=0.06)
    parser.add_argument("--stereo-ncc-y-radius", type=int, default=1)
    parser.add_argument("--stereo-ncc-min-std", type=float, default=4.0)
    parser.add_argument("--stereo-disparity-consistency", action="store_true")
    parser.add_argument("--stereo-disparity-mad-factor", type=float, default=3.5)
    parser.add_argument("--stereo-disparity-min-abs", type=float, default=2.0)
    parser.add_argument("--stereo-disparity-min-points", type=int, default=8)
    parser.add_argument("--min-disparity", type=float, default=1.5)
    parser.add_argument("--max-disparity", type=float, default=160.0)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=18.0)
    parser.add_argument(
        "--stable-point-cloud-min-age",
        type=int,
        default=3,
        help="Only publish sparse landmarks that survived at least this many frames.",
    )
    parser.add_argument(
        "--stable-point-cloud-fresh-depth-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only publish sparse landmarks with a fresh current-frame stereo depth update.",
    )
    parser.add_argument(
        "--track-depth-max-abs-change",
        type=float,
        default=0.0,
        help="Reject an existing track's new stereo depth if depth changes by more than this many meters; 0 disables.",
    )
    parser.add_argument(
        "--track-depth-max-rel-change",
        type=float,
        default=0.0,
        help="Reject an existing track's new stereo depth if relative depth change exceeds this ratio; 0 disables.",
    )
    parser.add_argument(
        "--focal-scale",
        type=float,
        default=1.0,
        help="Common multiplier for fx/fy. Use for effective underwater focal-length correction.",
    )
    parser.add_argument("--fx-scale", type=float, default=1.0, help="Additional multiplier for fx after --focal-scale.")
    parser.add_argument("--fy-scale", type=float, default=1.0, help="Additional multiplier for fy after --focal-scale.")
    parser.add_argument("--cx-offset-px", type=float, default=0.0, help="Principal point x correction in pixels.")
    parser.add_argument("--cy-offset-px", type=float, default=0.0, help="Principal point y correction in pixels.")
    parser.add_argument("--baseline-scale", type=float, default=1.0, help="Multiplier for stereo baseline.")
    parser.add_argument(
        "--baseline-m",
        type=float,
        default=0.0,
        help="Override stereo baseline in meters before --baseline-scale; 0 uses metadata/CameraInfo.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to stereo-triangulated 3D points before PnP. "
            "Use this to test effective underwater stereo calibration without feeding reference odometry into VIO."
        ),
    )
    parser.add_argument(
        "--stereo-depth-mode",
        choices=["klt", "sgbm", "hybrid"],
        default="klt",
        help="hybrid uses KLT stereo first and fills missing depths from dense SGBM disparity.",
    )
    parser.add_argument("--sgbm-num-disparities", type=int, default=160)
    parser.add_argument("--sgbm-block-size", type=int, default=5)
    parser.add_argument("--sgbm-uniqueness", type=int, default=8)
    parser.add_argument("--sgbm-sample-radius", type=int, default=2)
    parser.add_argument(
        "--sgbm-min-patch-valid-ratio",
        type=float,
        default=0.0,
        help="Reject SGBM feature depths unless this fraction of the local disparity patch is valid; 0 disables.",
    )
    parser.add_argument(
        "--sgbm-patch-max-mad",
        type=float,
        default=0.0,
        help="Reject SGBM feature depths when local disparity MAD is above this many pixels; 0 disables.",
    )
    parser.add_argument(
        "--sgbm-require-center-valid",
        action="store_true",
        help="Reject SGBM feature depths unless the disparity at the exact feature pixel is valid.",
    )
    parser.add_argument(
        "--sgbm-center-max-diff",
        type=float,
        default=0.0,
        help="Reject SGBM feature depths when center disparity differs from patch median by this many pixels; 0 disables.",
    )
    parser.add_argument(
        "--sgbm-fallback-min-klt",
        type=int,
        default=24,
        help="In hybrid mode, use SGBM only when KLT stereo returns fewer valid depths than this.",
    )
    parser.add_argument("--pnp-threshold", type=float, default=2.5)
    parser.add_argument("--pnp-confidence", type=float, default=0.999)
    parser.add_argument("--pnp-iterations", type=int, default=120)
    parser.add_argument("--pnp-ransac-attempts", type=int, default=1)
    parser.add_argument("--pnp-inlier-slack", type=int, default=2)
    parser.add_argument(
        "--pnp-use-imu-prior",
        action="store_true",
        help="When multiple PnP RANSAC candidates have similar inlier counts, prefer the one closer to the IMU rotation delta.",
    )
    parser.add_argument(
        "--opencv-rng-seed",
        type=int,
        default=-1,
        help="Set >=0 for deterministic OpenCV/PnP RANSAC sweeps. Negative keeps OpenCV's default RNG state.",
    )
    parser.add_argument(
        "--underwater-pnp",
        action="store_true",
        help=(
            "Enable underwater-specific PnP validation. DVL remains excluded; the gate uses only "
            "reprojection error, IMU rotation consistency, and current-frame stereo depth consistency."
        ),
    )
    parser.add_argument(
        "--underwater-pnp-rank-candidates",
        action="store_true",
        help="When multiple PnP RANSAC candidates are close in inlier count, rank them by underwater consistency.",
    )
    parser.add_argument(
        "--underwater-pnp-action",
        choices=["clamp", "reject"],
        default="clamp",
        help="Action when underwater PnP consistency fails after the ordinary gates pass.",
    )
    parser.add_argument("--underwater-max-reprojection-rmse-px", type=float, default=3.0)
    parser.add_argument("--underwater-max-imu-rotation-error-rad", type=float, default=0.35)
    parser.add_argument("--underwater-min-stereo-inlier-ratio", type=float, default=0.12)
    parser.add_argument("--underwater-max-depth-residual-m", type=float, default=0.50)
    parser.add_argument(
        "--underwater-max-xyz-residual-m",
        type=float,
        default=0.0,
        help="Optional full 3D residual gate between PnP-predicted current points and fresh stereo depths; 0 disables.",
    )
    parser.add_argument("--min-pnp-inliers", type=int, default=18)
    parser.add_argument(
        "--min-current-stereo-confirmed",
        type=int,
        default=0,
        help="For PnP poses, require this many current-frame stereo-confirmed inlier features; 0 disables the gate.",
    )
    parser.add_argument(
        "--low-current-stereo-action",
        choices=["accept", "clamp", "reject"],
        default="accept",
        help="Action when current-frame stereo confirmation is below --min-current-stereo-confirmed.",
    )
    parser.add_argument(
        "--carry-tracks-without-stereo",
        action="store_true",
        help="For pose-success inlier tracks, keep a short predicted 3D track when current stereo depth is missing.",
    )
    parser.add_argument(
        "--carry-track-max-depth-age",
        type=int,
        default=2,
        help="Maximum consecutive frames a track can be carried without a fresh stereo depth update.",
    )
    parser.add_argument("--pose-solver", choices=["pnp", "stereo3d", "hybrid"], default="pnp")
    parser.add_argument("--stereo3d-threshold", type=float, default=0.12)
    parser.add_argument("--stereo3d-iterations", type=int, default=120)
    parser.add_argument("--min-stereo3d-inliers", type=int, default=18)
    parser.add_argument(
        "--pose-mode",
        choices=["relative", "map"],
        default="relative",
        help="relative uses frame-to-frame 3D points; map uses persistent stereo landmarks in the initial world frame.",
    )
    parser.add_argument("--max-step-m", type=float, default=0.20)
    parser.add_argument("--max-speed-mps", type=float, default=1.0)
    parser.add_argument(
        "--motion-gate-mode",
        choices=["any", "all"],
        default="any",
        help="any accepts a visual step when either speed or absolute step is within bounds; all requires both.",
    )
    parser.add_argument("--max-rotation-rad", type=float, default=0.45)
    parser.add_argument("--imu-rotation-gate-rad", type=float, default=0.75)
    parser.add_argument("--translation-smoothing", type=float, default=0.15)
    parser.add_argument(
        "--coast-on-failure",
        action="store_true",
        help="When visual PnP fails, propagate a short constant-velocity step instead of freezing the pose.",
    )
    parser.add_argument(
        "--coast-rotation-source",
        choices=("visual-then-imu", "visual", "flow-yaw-then-imu", "flow-yaw-then-identity", "flow-yaw", "imu"),
        default="visual-then-imu",
        help="Rotation source used while coasting through temporary frontend loss.",
    )
    parser.add_argument("--visual-coast-min-tracks", type=int, default=8)
    parser.add_argument("--visual-coast-ransac-threshold", type=float, default=1.5)
    parser.add_argument("--visual-coast-max-rotation-rad", type=float, default=0.75)
    parser.add_argument("--flow-yaw-coast-scale", type=float, default=1.0)
    parser.add_argument("--flow-yaw-coast-max-rad", type=float, default=0.35)
    parser.add_argument("--coast-decay", type=float, default=0.85)
    parser.add_argument("--max-coast-frames", type=int, default=8)
    parser.add_argument(
        "--orientation-source",
        choices=[
            "visual",
            "imu",
            "imu-yaw",
            "imu-step",
            "imu-yaw-step",
            "imu-absolute",
            "imu-absolute-step",
        ],
        default="visual",
        help=(
            "visual keeps the original PnP rotation. imu propagates full rotation from gyro. "
            "imu-yaw propagates one gyro axis only and re-solves visual translation with that rotation fixed. "
            "imu-step/imu-yaw-step keep visual translation but use gyro attitude propagation. "
            "imu-absolute uses quaternion attitude from imu.csv when available."
        ),
    )
    parser.add_argument(
        "--imu-attitude-extrinsic",
        choices=["identity", "opencv-to-ros", "ros-to-opencv"],
        default="identity",
        help="Frame bridge applied to absolute IMU quaternion deltas before using them as camera-frame rotation.",
    )
    parser.add_argument(
        "--imu-attitude-delta",
        choices=["prev-to-curr", "curr-to-prev"],
        default="prev-to-curr",
        help="Quaternion delta convention for absolute IMU attitude.",
    )
    parser.add_argument(
        "--imu-yaw-axis",
        choices=["x", "y", "z"],
        default="y",
        help="Gyro axis used by --orientation-source imu-yaw. Realsense optical yaw for a forward camera is often y.",
    )
    parser.add_argument(
        "--gyro-sign",
        type=float,
        default=1.0,
        help="Sign applied to gyro angular velocity before SO(3) integration; useful for frame convention checks.",
    )
    parser.add_argument(
        "--underwater-planar-motion-prior",
        action="store_true",
        help=(
            "For pool-like underwater runs, replace the noisy lateral/vertical PnP step with a "
            "camera-forward planar step. PnP still supplies scale/inlier validation; DVL is not used."
        ),
    )
    parser.add_argument("--planar-forward-gain", type=float, default=1.0)
    parser.add_argument("--planar-lateral-gain", type=float, default=0.0)
    parser.add_argument("--planar-vertical-gain", type=float, default=0.15)
    parser.add_argument("--planar-max-step-m", type=float, default=0.0, help="0 keeps the normal motion gate limit.")
    parser.add_argument(
        "--planar-rotation-slowdown-rad",
        type=float,
        default=0.0,
        help="0 disables. Positive values reduce forward step during large gyro/visual rotation updates.",
    )
    parser.add_argument("--planar-min-turn-speed-scale", type=float, default=0.25)
    parser.add_argument("--disable-imu-rotation-gate", action="store_true")
    parser.add_argument("--write-debug-images", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.opencv_rng_seed) >= 0:
        cv2.setRNGSeed(int(args.opencv_rng_seed))
    dataset_dir = args.dataset_dir.expanduser().resolve()
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    times = read_times(dataset_dir / "times.txt")
    if args.max_frames > 0:
        times = times[: args.max_frames]
    if len(times) < 2:
        raise SystemExit("Need at least two stereo frames.")

    image0_dir = dataset_dir / "image_0"
    image1_dir = dataset_dir / "image_1"
    if args.swap_stereo:
        image0_dir, image1_dir = image1_dir, image0_dir
    left_paths = [image0_dir / f"{index:06d}.png" for index in range(len(times))]
    right_paths = [image1_dir / f"{index:06d}.png" for index in range(len(times))]
    missing = [str(path) for path in left_paths + right_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing exported stereo images. First missing: {missing[0]}")

    camera_matrix, fx, fy, cx, cy, baseline = camera_from_metadata(metadata, args)
    imu_rows = read_imu_csv(dataset_dir / "imu.csv")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.point_cloud_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.raw_point_cloud_csv is not None:
        args.raw_point_cloud_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.debug_csv.parent.mkdir(parents=True, exist_ok=True)

    prev_left = cv2.imread(str(left_paths[0]), cv2.IMREAD_GRAYSCALE)
    prev_right = cv2.imread(str(right_paths[0]), cv2.IMREAD_GRAYSCALE)
    if prev_left is None or prev_right is None:
        raise SystemExit("Failed to read first stereo frame.")
    prev_left = preprocess_gray(prev_left, args)
    prev_right = preprocess_gray(prev_right, args)

    next_track_id = 0
    initial_points = detect_grid_features(prev_left, [], args)
    points_3d, depth_mask, _ = estimate_stereo_depths(
        prev_left,
        prev_right,
        initial_points,
        fx,
        fy,
        cx,
        cy,
        baseline,
        args,
    )
    initial_left = initial_points[depth_mask]
    initial_right_points = klt_stereo_right_points(
        prev_left,
        prev_right,
        initial_points,
        fx,
        fy,
        cx,
        cy,
        baseline,
        args,
    )
    tracks: dict[int, Track] = {}
    for original_index, left_point, point_3d in zip(np.flatnonzero(depth_mask), initial_left, points_3d):
        point_3d = point_3d.astype(np.float64)
        tracks[next_track_id] = Track(
            next_track_id,
            left_point.astype(np.float32),
            point_3d,
            right_point=initial_right_points.get(int(original_index)),
            world_point=point_3d.copy(),
        )
        next_track_id += 1

    position = np.zeros(3, dtype=np.float64)
    rotation_world_camera = np.eye(3, dtype=np.float64)
    previous_step = np.zeros(3, dtype=np.float64)
    previous_camera_step = np.zeros(3, dtype=np.float64)
    rows = [
        PoseRow(
            frame_index=0,
            timestamp_sec=times[0],
            success=True,
            position=position.copy(),
            rotation=rotation_world_camera.copy(),
            pose_inliers=0,
            pose_inlier_ratio=0.0,
            active_tracks=len(tracks),
            stereo_tracks=len(tracks),
            note="initial_pose",
        )
    ]
    point_cloud_rows: list[dict[str, Any]] = []
    raw_point_cloud_rows: list[dict[str, Any]] = []
    append_point_cloud_rows(raw_point_cloud_rows, 0, times[0], tracks, args, stable_only=False)
    append_point_cloud_rows(point_cloud_rows, 0, times[0], tracks, args, stable_only=True)
    debug_rows: list[dict[str, Any]] = []
    accepted_count = 0
    coast_count = 0

    debug_dir = args.debug_csv.parent / "ros2_stereo_imu_debug_frames"
    if args.write_debug_images:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in range(1, len(times)):
        curr_left = cv2.imread(str(left_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
        curr_right = cv2.imread(str(right_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
        if curr_left is None or curr_right is None:
            break
        curr_left = preprocess_gray(curr_left, args)
        curr_right = preprocess_gray(curr_right, args)

        dt = max(times[frame_index] - times[frame_index - 1], 1e-6)
        track_ids = list(tracks.keys())
        prev_points = np.array([tracks[track_id].left_point for track_id in track_ids], dtype=np.float32)
        prev_velocities = np.array(
            [
                tracks[track_id].image_velocity if tracks[track_id].image_velocity is not None else np.zeros(2, dtype=np.float32)
                for track_id in track_ids
            ],
            dtype=np.float32,
        )
        flow = track_klt(prev_left, curr_left, prev_points, args, previous_velocities=prev_velocities)

        kept_ids = [track_id for track_id, keep in zip(track_ids, flow["mask"]) if keep]
        prev_kept_points = prev_points[flow["mask"]]
        curr_points = flow["curr"][flow["mask"]]
        quad_candidates = 0
        quad_kept = 0
        quad_applied = False
        if args.temporal_ransac != "none":
            temporal_mask = filter_temporal_matches(
                prev_kept_points,
                curr_points,
                camera_matrix,
                args,
            )
            kept_ids = [track_id for track_id, keep in zip(kept_ids, temporal_mask) if keep]
            prev_kept_points = prev_kept_points[temporal_mask]
            curr_points = curr_points[temporal_mask]
        if len(kept_ids):
            quad_mask, quad_candidates, quad_kept, quad_applied = filter_quad_stereo_temporal_matches(
                prev_right,
                curr_left,
                curr_right,
                kept_ids,
                curr_points,
                tracks,
                args,
            )
            if quad_applied:
                kept_ids = [track_id for track_id, keep in zip(kept_ids, quad_mask) if keep]
                prev_kept_points = prev_kept_points[quad_mask]
                curr_points = curr_points[quad_mask]
        pnp_ids: list[int] = []
        pnp_image_points: list[np.ndarray] = []
        pnp_object_points: list[np.ndarray] = []
        for track_id, image_point in zip(kept_ids, curr_points):
            track = tracks[track_id]
            if track.age < args.min_track_age_for_pnp:
                continue
            object_point = track.world_point if args.pose_mode == "map" else track.point_3d
            if object_point is None:
                continue
            pnp_ids.append(track_id)
            pnp_image_points.append(image_point)
            pnp_object_points.append(object_point)
        object_points = np.array(pnp_object_points, dtype=np.float32)
        image_points = np.array(pnp_image_points, dtype=np.float32)
        stereo3d_ids: list[int] = []
        stereo3d_prev_points: list[np.ndarray] = []
        stereo3d_curr_points: list[np.ndarray] = []
        if args.pose_solver in {"stereo3d", "hybrid"} and args.pose_mode == "relative" and len(curr_points):
            current_points_3d, current_depth_mask, _ = estimate_stereo_depths(
                curr_left,
                curr_right,
                curr_points,
                fx,
                fy,
                cx,
                cy,
                baseline,
                args,
            )
            current_depth_ids = [track_id for track_id, keep in zip(kept_ids, current_depth_mask) if keep]
            for track_id, current_point_3d in zip(current_depth_ids, current_points_3d):
                track = tracks[track_id]
                if track.age < args.min_track_age_for_pnp:
                    continue
                if not track_depth_update_passes(track.point_3d, current_point_3d, args):
                    continue
                stereo3d_ids.append(track_id)
                stereo3d_prev_points.append(track.point_3d)
                stereo3d_curr_points.append(current_point_3d)
        stereo3d_prev = np.array(stereo3d_prev_points, dtype=np.float64)
        stereo3d_curr = np.array(stereo3d_curr_points, dtype=np.float64)

        pose_success = False
        pnp_inliers = np.empty((0,), dtype=np.int32)
        pose_inlier_track_ids: set[int] = set()
        note = "not_enough_tracked_points"
        planar_prior_applied = False
        proposed_step = np.zeros(3, dtype=np.float64)
        relative_rotation = np.eye(3, dtype=np.float64)
        relative_translation_camera = np.zeros(3, dtype=np.float64)
        proposed_position = position.copy()
        proposed_rotation = rotation_world_camera.copy()
        current_stereo_confirmed = 0
        underwater_quality = UnderwaterPnpQuality()
        underwater_pnp_reason = ""
        imu_rotation = integrate_gyro_rotation(imu_rows, times[frame_index - 1], times[frame_index], args.gyro_sign)
        imu_yaw_rotation = integrate_gyro_rotation(
            imu_rows,
            times[frame_index - 1],
            times[frame_index],
            args.gyro_sign,
            axis=args.imu_yaw_axis,
        )
        imu_absolute_rotation = absolute_imu_relative_rotation(
            imu_rows,
            times[frame_index - 1],
            times[frame_index],
            args.imu_attitude_extrinsic,
            args.imu_attitude_delta,
        )
        imu_gate_rotation = imu_absolute_rotation if args.orientation_source.startswith("imu-absolute") else imu_rotation
        imu_angle = rotation_angle(imu_gate_rotation)
        visual_coast_rotation = estimate_visual_coast_rotation(prev_kept_points, curr_points, camera_matrix, args)
        flow_yaw_coast_rotation = estimate_flow_yaw_coast_rotation(prev_kept_points, curr_points, fx, args)
        visual_angle = 0.0

        if args.pose_solver in {"stereo3d", "hybrid"} and len(stereo3d_prev) >= max(6, args.min_stereo3d_inliers):
            stereo3d_result = solve_stereo3d_motion(stereo3d_prev, stereo3d_curr, args)
            if stereo3d_result is not None:
                relative_rotation, translation, stereo3d_inliers, _ = stereo3d_result
                relative_translation_camera = translation.reshape(3).astype(np.float64)
                pnp_inliers = stereo3d_inliers
                pose_inlier_track_ids = {
                    stereo3d_ids[int(index)]
                    for index in stereo3d_inliers.reshape(-1)
                    if int(index) < len(stereo3d_ids)
                }
                proposed_rotation = rotation_world_camera @ relative_rotation.T
                camera_step = -relative_rotation.T @ translation.reshape(3)
                proposed_step = rotation_world_camera @ camera_step
                proposed_position = position + proposed_step
                visual_angle = rotation_angle(relative_rotation)
                step_norm = float(np.linalg.norm(proposed_step))
                speed = step_norm / dt
                rotation_gate_ok = visual_angle <= args.max_rotation_rad
                imu_gate_ok = True
                if not args.disable_imu_rotation_gate:
                    imu_gate_ok = abs(visual_angle - imu_angle) <= args.imu_rotation_gate_rad
                motion_gate_ok = motion_gate_passes(speed, step_norm, args)

                if len(stereo3d_inliers) < args.min_stereo3d_inliers:
                    note = "stereo3d_low_inliers"
                elif not rotation_gate_ok:
                    note = "stereo3d_rotation_gate"
                elif not imu_gate_ok:
                    note = "stereo3d_imu_rotation_gate"
                elif not motion_gate_ok:
                    if step_norm > 1e-9:
                        limit = min(args.max_step_m, args.max_speed_mps * dt)
                        proposed_step = proposed_step * (limit / step_norm)
                        proposed_position = position + proposed_step
                        note = "stereo3d_motion_clamped"
                        pose_success = True
                    else:
                        note = "stereo3d_zero_step"
                else:
                    note = "stereo3d_accepted"
                    pose_success = True
            else:
                note = "stereo3d_failed"

        if not pose_success and args.pose_solver in {"pnp", "hybrid"} and len(object_points) >= max(6, args.min_pnp_inliers):
            underwater_context = (
                UnderwaterPnpContext(curr_left, curr_right, fx, fy, cx, cy, baseline)
                if args.underwater_pnp and args.underwater_pnp_rank_candidates
                else None
            )
            pnp_result = solve_pnp_motion(
                object_points,
                image_points,
                camera_matrix,
                args,
                imu_gate_rotation,
                underwater_context,
            )
            if pnp_result is not None:
                pnp_rotation, translation, pnp_inliers = pnp_result
                projection_rotation = pnp_rotation.astype(np.float64)
                pose_inlier_track_ids = {
                    pnp_ids[int(index)]
                    for index in pnp_inliers.reshape(-1)
                    if int(index) < len(pnp_ids)
                }
                if args.pose_mode == "map":
                    proposed_rotation = pnp_rotation.T
                    proposed_position = -proposed_rotation @ translation.reshape(3)
                    relative_rotation = proposed_rotation.T @ rotation_world_camera
                    proposed_step = proposed_position - position
                else:
                    relative_rotation = pnp_rotation
                    camera_step_rotation = pnp_rotation
                    fixed_rotation = None
                    if args.orientation_source == "imu":
                        fixed_rotation = imu_rotation
                    elif args.orientation_source == "imu-yaw":
                        fixed_rotation = imu_yaw_rotation
                    elif args.orientation_source == "imu-absolute":
                        fixed_rotation = imu_absolute_rotation
                    elif args.orientation_source == "imu-step":
                        relative_rotation = imu_rotation
                    elif args.orientation_source == "imu-yaw-step":
                        relative_rotation = imu_yaw_rotation
                    elif args.orientation_source == "imu-absolute-step":
                        relative_rotation = imu_absolute_rotation
                    if fixed_rotation is not None:
                        translation_from_imu = solve_translation_fixed_rotation(
                            object_points,
                            image_points,
                            camera_matrix,
                            fixed_rotation,
                            pnp_inliers,
                        )
                        if translation_from_imu is not None:
                            relative_rotation = fixed_rotation
                            camera_step_rotation = fixed_rotation
                            projection_rotation = fixed_rotation
                            translation = translation_from_imu
                    relative_translation_camera = translation.reshape(3).astype(np.float64)
                    proposed_rotation = rotation_world_camera @ relative_rotation.T
                    camera_step = -camera_step_rotation.T @ translation.reshape(3)
                    proposed_step = rotation_world_camera @ camera_step
                    proposed_position = position + proposed_step
                if args.underwater_pnp:
                    underwater_quality = evaluate_underwater_pnp_quality(
                        curr_left,
                        curr_right,
                        object_points,
                        image_points,
                        pnp_inliers,
                        projection_rotation,
                        translation,
                        camera_matrix,
                        imu_gate_rotation,
                        fx,
                        fy,
                        cx,
                        cy,
                        baseline,
                        args,
                        pnp_ids=pnp_ids,
                        tracks=tracks,
                    )
                    current_stereo_confirmed = underwater_quality.stereo_confirmed
                else:
                    current_stereo_confirmed = count_current_stereo_confirmed(
                        curr_left,
                        curr_right,
                        image_points,
                        pnp_inliers,
                        fx,
                        fy,
                        cx,
                        cy,
                        baseline,
                        args,
                    )
                if args.underwater_planar_motion_prior and args.pose_mode == "relative":
                    proposed_step = apply_underwater_planar_motion_prior(
                        proposed_step,
                        proposed_rotation,
                        relative_rotation,
                        dt,
                        args,
                    )
                    proposed_position = position + proposed_step
                    planar_prior_applied = True
                visual_angle = rotation_angle(relative_rotation)
                step_norm = float(np.linalg.norm(proposed_step))
                speed = step_norm / dt
                rotation_gate_ok = visual_angle <= args.max_rotation_rad
                imu_gate_ok = True
                if not args.disable_imu_rotation_gate:
                    imu_gate_ok = abs(visual_angle - imu_angle) <= args.imu_rotation_gate_rad
                motion_gate_ok = motion_gate_passes(speed, step_norm, args)
                underwater_pnp_ok = True
                if args.underwater_pnp:
                    underwater_pnp_ok, underwater_pnp_reason = underwater_pnp_quality_passes(underwater_quality, args)

                if len(pnp_inliers) < args.min_pnp_inliers:
                    note = "pnp_low_inliers"
                elif current_stereo_confirmed < args.min_current_stereo_confirmed:
                    if args.low_current_stereo_action == "clamp" and step_norm > 1e-9:
                        limit = min(args.max_step_m, args.max_speed_mps * dt)
                        proposed_step = proposed_step * (limit / step_norm)
                        proposed_position = position + proposed_step
                        note = "current_stereo_low_clamped"
                        pose_success = True
                    elif args.low_current_stereo_action == "accept":
                        note = "current_stereo_low_accepted"
                        pose_success = True
                    else:
                        note = "current_stereo_low_rejected"
                elif not rotation_gate_ok:
                    note = "visual_rotation_gate"
                elif not imu_gate_ok:
                    note = "imu_rotation_gate"
                elif not underwater_pnp_ok:
                    reason_label = underwater_pnp_reason.split(",")[0] if underwater_pnp_reason else "quality"
                    if args.underwater_pnp_action == "clamp" and step_norm > 1e-9:
                        limit = min(args.max_step_m, args.max_speed_mps * dt)
                        proposed_step = proposed_step * (limit / step_norm)
                        proposed_position = position + proposed_step
                        note = f"underwater_pnp_clamped_{reason_label}"
                        pose_success = True
                    else:
                        note = f"underwater_pnp_rejected_{reason_label}"
                elif not motion_gate_ok:
                    if step_norm > 1e-9:
                        limit = min(args.max_step_m, args.max_speed_mps * dt)
                        proposed_step = proposed_step * (limit / step_norm)
                        proposed_position = position + proposed_step
                        note = "motion_clamped"
                        pose_success = True
                    else:
                        note = "zero_step"
                else:
                    note = "accepted"
                    pose_success = True
            else:
                note = "pnp_failed"

        if pose_success:
            if 0.0 < args.translation_smoothing < 1.0:
                proposed_step = (1.0 - args.translation_smoothing) * proposed_step + args.translation_smoothing * previous_step
                proposed_position = position + proposed_step
            previous_camera_step = rotation_world_camera.T @ proposed_step
            position = proposed_position
            rotation_world_camera = proposed_rotation if args.pose_mode == "map" else rotation_world_camera @ relative_rotation.T
            previous_step = proposed_step
            accepted_count += 1
            coast_count = 0
        elif args.coast_on_failure and coast_count < args.max_coast_frames and float(np.linalg.norm(previous_step)) > 1e-9:
            decay = max(0.0, min(1.0, args.coast_decay))
            coast_rotation: np.ndarray | None = None
            coast_source = ""
            if args.coast_rotation_source in {"flow-yaw", "flow-yaw-then-imu", "flow-yaw-then-identity"} and flow_yaw_coast_rotation is not None:
                coast_rotation = flow_yaw_coast_rotation
                coast_source = "flow_yaw"
            elif args.coast_rotation_source in {"visual", "visual-then-imu"} and visual_coast_rotation is not None:
                coast_rotation = visual_coast_rotation
                coast_source = "visual"
            elif args.coast_rotation_source == "flow-yaw-then-identity":
                coast_rotation = np.eye(3, dtype=np.float64)
                coast_source = "identity"
            elif args.coast_rotation_source in {"visual-then-imu", "flow-yaw-then-imu", "imu"}:
                coast_rotation = imu_rotation
                coast_source = "imu"
            if coast_rotation is not None:
                proposed_rotation = rotation_world_camera @ coast_rotation.T
                coast_camera_step = previous_camera_step * decay
                proposed_step = proposed_rotation @ coast_camera_step
                position = position + proposed_step
                rotation_world_camera = proposed_rotation
                previous_step = proposed_step
                previous_camera_step = coast_camera_step
                coast_count += 1
                note = f"coasted_{coast_source}_after_{note}"
            else:
                note = f"coast_no_visual_rotation_after_{note}"

        if planar_prior_applied and pose_success:
            note = f"{note}_planar"

        inlier_track_ids = set()
        if pose_inlier_track_ids:
            inlier_track_ids = pose_inlier_track_ids
        elif len(pnp_inliers):
            inlier_track_ids = {pnp_ids[int(index)] for index in pnp_inliers.reshape(-1) if int(index) < len(pnp_ids)}
        survivor_ids = [track_id for track_id in kept_ids if track_id in inlier_track_ids] if pose_success else kept_ids
        survivor_points = np.array(
            [point for track_id, point in zip(kept_ids, curr_points) if (track_id in inlier_track_ids if pose_success else True)],
            dtype=np.float32,
        )

        next_tracks: dict[int, Track] = {}
        stereo_count = 0
        if len(survivor_points):
            survivor_3d, survivor_depth_mask, _depth_stats = estimate_stereo_depths(
                curr_left,
                curr_right,
                survivor_points,
                fx,
                fy,
                cx,
                cy,
                baseline,
                args,
            )
            depth_left = survivor_points[survivor_depth_mask]
            depth_ids = [track_id for track_id, keep in zip(survivor_ids, survivor_depth_mask) if keep]
            survivor_right_points = klt_stereo_right_points(
                curr_left,
                curr_right,
                survivor_points,
                fx,
                fy,
                cx,
                cy,
                baseline,
                args,
            )
            stereo_updates: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
            for survivor_index, track_id, left_point, point_3d in zip(np.flatnonzero(survivor_depth_mask), depth_ids, depth_left, survivor_3d):
                previous = tracks[track_id]
                if not track_depth_update_passes(previous.point_3d, point_3d, args):
                    continue
                stereo_updates[track_id] = (left_point, point_3d, survivor_right_points.get(int(survivor_index)))
            stereo_count = len(stereo_updates)
            for track_id, left_point in zip(survivor_ids, survivor_points):
                previous = tracks[track_id]
                image_velocity = left_point.astype(np.float32) - previous.left_point.astype(np.float32)
                right_point = None
                if track_id in stereo_updates:
                    left_point, point_3d, right_point = stereo_updates[track_id]
                    point_3d = point_3d.astype(np.float64)
                    world_point = position + rotation_world_camera @ point_3d
                    missing_depth_age = 0
                elif (
                    args.carry_tracks_without_stereo
                    and pose_success
                    and args.pose_mode == "relative"
                    and track_id in inlier_track_ids
                    and previous.missing_depth_age < args.carry_track_max_depth_age
                ):
                    point_3d = relative_rotation @ previous.point_3d + relative_translation_camera
                    if not track_camera_point_passes(point_3d, args):
                        continue
                    world_point = position + rotation_world_camera @ point_3d
                    missing_depth_age = previous.missing_depth_age + 1
                elif args.pose_mode == "map" and previous.world_point is not None:
                    world_point = previous.world_point.copy()
                    point_3d = rotation_world_camera.T @ (world_point - position)
                    missing_depth_age = previous.missing_depth_age + 1
                else:
                    continue
                next_tracks[track_id] = Track(
                    track_id=track_id,
                    left_point=left_point.astype(np.float32),
                    point_3d=point_3d.astype(np.float64),
                    right_point=right_point.astype(np.float32) if right_point is not None else None,
                    world_point=world_point.astype(np.float64),
                    image_velocity=image_velocity.astype(np.float32),
                    age=tracks[track_id].age + 1 if track_id in tracks else 1,
                    missing_depth_age=missing_depth_age,
                )

        periodic_refill = (
            args.feature_refill_interval > 0
            and frame_index % args.feature_refill_interval == 0
            and len(next_tracks) < args.max_features
        )
        if len(next_tracks) < args.min_features or periodic_refill:
            existing = [track.left_point for track in next_tracks.values()]
            new_left = detect_grid_features(curr_left, existing, args)
            if len(new_left):
                new_points_3d, new_depth_mask, _ = estimate_stereo_depths(
                    curr_left,
                    curr_right,
                    new_left,
                    fx,
                    fy,
                    cx,
                    cy,
                    baseline,
                    args,
                )
                new_right_points = klt_stereo_right_points(
                    curr_left,
                    curr_right,
                    new_left,
                    fx,
                    fy,
                    cx,
                    cy,
                    baseline,
                    args,
                )
                for original_index, left_point, point_3d in zip(np.flatnonzero(new_depth_mask), new_left[new_depth_mask], new_points_3d):
                    if len(next_tracks) >= args.max_features:
                        break
                    point_3d = point_3d.astype(np.float64)
                    next_tracks[next_track_id] = Track(
                        track_id=next_track_id,
                        left_point=left_point.astype(np.float32),
                        point_3d=point_3d,
                        right_point=new_right_points.get(int(original_index)),
                        world_point=(position + rotation_world_camera @ point_3d).astype(np.float64),
                    )
                    next_track_id += 1

        tracks = next_tracks
        append_point_cloud_rows(raw_point_cloud_rows, frame_index, times[frame_index], tracks, args, stable_only=False)
        append_point_cloud_rows(point_cloud_rows, frame_index, times[frame_index], tracks, args, stable_only=True)
        frontend_health = evaluate_frontend_health(
            tracked_count=len(kept_ids),
            pnp_inliers=int(len(pnp_inliers)),
            current_stereo_confirmed=current_stereo_confirmed,
            quad_candidates=quad_candidates,
            quad_kept=quad_kept,
            pose_success=pose_success,
            args=args,
        )
        rows.append(
            PoseRow(
                frame_index=frame_index,
                timestamp_sec=times[frame_index],
                success=pose_success,
                position=position.copy(),
                rotation=rotation_world_camera.copy(),
                pose_inliers=int(len(pnp_inliers)),
                pose_inlier_ratio=float(len(pnp_inliers) / max(1, len(object_points))),
                active_tracks=len(tracks),
                stereo_tracks=stereo_count,
                note=note,
            )
        )
        debug_rows.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": f"{times[frame_index]:.9f}",
                "dt": f"{dt:.9f}",
                "tracked": len(kept_ids),
                "quad_candidates": quad_candidates,
                "quad_kept": quad_kept,
                "quad_applied": int(quad_applied),
                "frontend_health_score": f"{frontend_health.score:.6f}",
                "frontend_health_state": frontend_health.state,
                "frontend_track_ratio": f"{frontend_health.track_ratio:.6f}",
                "frontend_inlier_ratio": f"{frontend_health.inlier_ratio:.6f}",
                "frontend_stereo_ratio": f"{frontend_health.stereo_ratio:.6f}",
                "frontend_quad_reject_ratio": f"{frontend_health.quad_reject_ratio:.6f}",
                "pnp_inliers": int(len(pnp_inliers)),
                "active_tracks": len(tracks),
                "stereo_tracks": stereo_count,
                "step_norm": f"{float(np.linalg.norm(proposed_step)):.9f}",
                "speed_mps": f"{float(np.linalg.norm(proposed_step)) / dt:.9f}",
                "visual_rotation_rad": f"{visual_angle:.9f}",
                "imu_rotation_rad": f"{imu_angle:.9f}",
                "current_stereo_confirmed": current_stereo_confirmed,
                "pnp_reprojection_rmse_px": format_optional_float(underwater_quality.reprojection_rmse_px),
                "pnp_imu_rotation_error_rad": format_optional_float(underwater_quality.imu_rotation_error_rad),
                "pnp_stereo_inlier_ratio": format_optional_float(underwater_quality.stereo_inlier_ratio),
                "pnp_depth_median_abs_residual_m": format_optional_float(underwater_quality.depth_median_abs_residual_m),
                "pnp_depth_p90_abs_residual_m": format_optional_float(underwater_quality.depth_p90_abs_residual_m),
                "pnp_xyz_median_residual_m": format_optional_float(underwater_quality.xyz_median_residual_m),
                "underwater_pnp_score": format_optional_float(underwater_quality.score),
                "underwater_pnp_reason": underwater_pnp_reason,
                "note": note,
            }
        )

        if args.write_debug_images and frame_index % 10 == 0:
            write_debug_image(debug_dir / f"{frame_index:06d}.jpg", curr_left, tracks)

        prev_left = curr_left
        prev_right = curr_right

    write_odometry_csv(args.output_csv, rows)
    write_point_cloud_csv(args.point_cloud_csv, point_cloud_rows)
    if args.raw_point_cloud_csv is not None:
        write_point_cloud_csv(args.raw_point_cloud_csv, raw_point_cloud_rows)
    write_debug_csv(args.debug_csv, debug_rows)
    summary = make_summary(args, metadata, rows, debug_rows, accepted_count)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def read_times(path: Path) -> list[float]:
    return [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_imu_csv(path: Path) -> list[ImuRow]:
    rows: list[ImuRow] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            orientation = None
            if all(key in row and row[key] not in {"", None} for key in ("qx", "qy", "qz", "qw")):
                quaternion = np.array(
                    [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
                    dtype=np.float64,
                )
                norm = float(np.linalg.norm(quaternion))
                if norm > 1e-12:
                    orientation = quaternion / norm
            rows.append(
                ImuRow(
                    timestamp_sec=float(row["timestamp_sec"]),
                    gyro=np.array([float(row["gx"]), float(row["gy"]), float(row["gz"])], dtype=np.float64),
                    orientation=orientation,
                )
            )
    return rows


def camera_from_metadata(metadata: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, float, float, float, float, float]:
    cam0 = metadata["camera0"]
    k = cam0["k"]
    fx = float(k[0]) * args.focal_scale * args.fx_scale
    fy = float(k[4]) * args.focal_scale * args.fy_scale
    cx = float(k[2]) + args.cx_offset_px
    cy = float(k[5]) + args.cy_offset_px
    baseline = float(args.baseline_m) if args.baseline_m > 0.0 else float(metadata.get("baseline_m") or 0.0)
    if baseline <= 0.0:
        p_right = metadata["camera1"]["p"]
        raw_fx = float(k[0])
        baseline = abs(float(p_right[3]) / raw_fx)
    baseline *= args.baseline_scale
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return camera_matrix, fx, fy, cx, cy, baseline


def build_feature_mask(gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    height, width = gray.shape
    mask = np.full((height, width), 255, dtype=np.uint8)

    bottom_px = max(int(args.mask_bottom_px), int(round(height * max(0.0, args.mask_bottom_ratio))))
    if bottom_px > 0:
        mask[max(0, height - bottom_px) :, :] = 0

    if args.mask_bright_threshold > 0:
        bright = gray >= int(args.mask_bright_threshold)
        if np.any(bright):
            dilate = max(1, int(args.mask_bright_dilate))
            if dilate % 2 == 0:
                dilate += 1
            kernel = np.ones((dilate, dilate), dtype=np.uint8)
            bright_mask = cv2.dilate(bright.astype(np.uint8) * 255, kernel, iterations=1) > 0
            mask[bright_mask] = 0

    return mask


def points_in_feature_mask(gray: np.ndarray, points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=bool)
    if args.mask_bottom_px <= 0 and args.mask_bottom_ratio <= 0.0 and args.mask_bright_threshold <= 0:
        return np.ones(len(points), dtype=bool)
    mask = build_feature_mask(gray, args)
    height, width = gray.shape
    keep = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(points):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        keep[index] = 0 <= x < width and 0 <= y < height and mask[y, x] > 0
    return keep


def detect_grid_features(gray: np.ndarray, existing_points: list[np.ndarray], args: argparse.Namespace) -> np.ndarray:
    height, width = gray.shape
    mask = build_feature_mask(gray, args)
    for point in existing_points:
        x, y = point
        cv2.circle(mask, (int(round(x)), int(round(y))), args.min_distance, 0, thickness=-1)

    features: list[tuple[np.ndarray, float]] = []
    rows = max(1, args.grid_rows)
    cols = max(1, args.grid_cols)
    for row in range(rows):
        y0 = int(round(row * height / rows))
        y1 = int(round((row + 1) * height / rows))
        for col in range(cols):
            x0 = int(round(col * width / cols))
            x1 = int(round((col + 1) * width / cols))
            roi = gray[y0:y1, x0:x1]
            roi_mask = mask[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            for point, response in detect_features_in_roi(roi, roi_mask, args):
                absolute = np.array([point[0] + x0, point[1] + y0], dtype=np.float32)
                if args.reject_repeated_patches and patch_is_repeated(gray, absolute, args):
                    continue
                features.append((absolute, response))

    if not features:
        return np.empty((0, 2), dtype=np.float32)
    features = sorted(
        features,
        key=lambda item: (
            int(item[0][1] // max(height / rows, 1)),
            int(item[0][0] // max(width / cols, 1)),
            -item[1],
        ),
    )
    return np.array([point for point, _ in features[: args.max_features]], dtype=np.float32)


def detect_features_in_roi(
    roi: np.ndarray,
    roi_mask: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[np.ndarray, float]]:
    if args.feature_detector in {"gftt", "harris"}:
        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=args.cell_max_features,
            qualityLevel=args.quality,
            minDistance=args.min_distance,
            mask=roi_mask,
            blockSize=args.block_size,
            useHarrisDetector=args.feature_detector == "harris",
        )
        if corners is None:
            return []
        if args.feature_detector == "harris":
            response_map = cv2.cornerHarris(roi, args.block_size, 3, 0.04)
        else:
            response_map = cv2.cornerMinEigenVal(roi, blockSize=args.block_size)
        points: list[tuple[np.ndarray, float]] = []
        height, width = roi.shape
        for corner in corners.reshape(-1, 2):
            x = int(np.clip(round(float(corner[0])), 0, width - 1))
            y = int(np.clip(round(float(corner[1])), 0, height - 1))
            points.append((corner.astype(np.float32), float(response_map[y, x])))
        return sorted(points, key=lambda item: item[1], reverse=True)[: args.cell_max_features]

    if args.feature_detector == "fast":
        detector = cv2.FastFeatureDetector_create(
            threshold=max(1, int(args.fast_threshold)),
            nonmaxSuppression=True,
        )
        keypoints = detector.detect(roi, roi_mask)
    elif args.feature_detector == "orb":
        detector = cv2.ORB_create(
            nfeatures=max(args.cell_max_features * 4, args.cell_max_features),
            fastThreshold=max(1, int(args.orb_fast_threshold)),
            edgeThreshold=max(8, int(args.patch_size)),
        )
        keypoints = detector.detect(roi, roi_mask)
    else:
        keypoints = []

    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    return [
        (np.array(kp.pt, dtype=np.float32), float(kp.response))
        for kp in keypoints[: args.cell_max_features]
    ]


def patch_is_repeated(gray: np.ndarray, point: np.ndarray, args: argparse.Namespace) -> bool:
    patch_size = max(5, int(args.patch_size))
    if patch_size % 2 == 0:
        patch_size += 1
    radius = patch_size // 2
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    height, width = gray.shape
    if x - radius < 0 or x + radius >= width or y - radius < 0 or y + radius >= height:
        return True

    patch = gray[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float32)
    patch -= float(np.mean(patch))
    patch_std = float(np.std(patch))
    if patch_std < args.patch_min_std:
        return True
    patch_norm = float(np.linalg.norm(patch))
    if patch_norm < 1e-9:
        return True

    search_radius = max(args.min_distance, int(args.patch_search_radius))
    stride = max(2, int(args.patch_search_stride))
    y0 = max(radius, y - search_radius)
    y1 = min(height - radius - 1, y + search_radius)
    x0 = max(radius, x - search_radius)
    x1 = min(width - radius - 1, x + search_radius)
    for yy in range(y0, y1 + 1, stride):
        for xx in range(x0, x1 + 1, stride):
            if abs(xx - x) <= args.min_distance and abs(yy - y) <= args.min_distance:
                continue
            other = gray[yy - radius : yy + radius + 1, xx - radius : xx + radius + 1].astype(np.float32)
            other -= float(np.mean(other))
            other_norm = float(np.linalg.norm(other))
            if other_norm < 1e-9:
                continue
            ncc = float(np.sum(patch * other) / (patch_norm * other_norm))
            if ncc >= args.patch_ncc_threshold:
                return True
    return False


def track_klt(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_points: np.ndarray,
    args: argparse.Namespace,
    previous_velocities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if len(prev_points) == 0:
        return {
            "curr": np.empty((0, 2), dtype=np.float32),
            "mask": np.empty((0,), dtype=bool),
        }
    lk_kwargs = {
        "winSize": (args.klt_window, args.klt_window),
        "maxLevel": args.klt_levels,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            args.klt_iterations,
            args.klt_eps,
        ),
    }
    prev_in = prev_points.reshape(-1, 1, 2).astype(np.float32)
    initial_curr = None
    lk_flags = 0
    if args.klt_use_prediction and previous_velocities is not None and len(previous_velocities) == len(prev_points):
        initial_curr = (prev_points.reshape(-1, 2) + previous_velocities.reshape(-1, 2)).reshape(-1, 1, 2).astype(np.float32)
        lk_flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    curr, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_in, initial_curr, flags=lk_flags, **lk_kwargs)
    if curr is None or status is None:
        return {"curr": np.empty((0, 2), dtype=np.float32), "mask": np.zeros(len(prev_points), dtype=bool)}
    back, back_status, back_err = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, curr, None, **lk_kwargs)
    if back is None or back_status is None:
        return {"curr": curr.reshape(-1, 2), "mask": np.zeros(len(prev_points), dtype=bool)}
    curr2 = curr.reshape(-1, 2)
    back2 = back.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)
    back_status = back_status.reshape(-1).astype(bool)
    fb = np.linalg.norm(prev_points.reshape(-1, 2) - back2, axis=1)
    finite = finite_points(curr2, curr_gray.shape)
    mask = status & back_status & finite & (fb <= args.fb_threshold)
    if args.klt_max_error > 0 and err is not None and back_err is not None:
        forward_error = err.reshape(-1)
        backward_error = back_err.reshape(-1)
        mask &= np.isfinite(forward_error) & np.isfinite(backward_error)
        mask &= (forward_error <= args.klt_max_error) & (backward_error <= args.klt_max_error)
    flow = np.linalg.norm(curr2 - prev_points.reshape(-1, 2), axis=1)
    if args.klt_max_flow_px > 0:
        mask &= flow <= args.klt_max_flow_px
    if args.klt_min_eigenvalue > 0:
        mask &= tracked_min_eigenvalue_mask(curr_gray, curr2, args)
    if args.klt_patch_ncc_threshold > 0:
        mask &= patch_ncc_mask(prev_gray, curr_gray, prev_points.reshape(-1, 2), curr2, args)
    if args.klt_local_flow_check:
        mask &= local_flow_consistency_mask(prev_points.reshape(-1, 2), curr2, mask, curr_gray.shape, args)
    if args.klt_velocity_check and previous_velocities is not None and len(previous_velocities) == len(prev_points):
        current_velocity = curr2 - prev_points.reshape(-1, 2)
        acceleration = np.linalg.norm(current_velocity - previous_velocities.reshape(-1, 2), axis=1)
        mask &= acceleration <= args.klt_max_accel_px
    mask &= points_in_feature_mask(curr_gray, curr2, args)
    return {"curr": curr2.astype(np.float32), "mask": mask}


def tracked_min_eigenvalue_mask(gray: np.ndarray, points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    response = cv2.cornerMinEigenVal(gray, blockSize=max(3, int(args.block_size)))
    height, width = gray.shape
    mask = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(points):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height:
            mask[index] = float(response[y, x]) >= args.klt_min_eigenvalue
    return mask


def patch_ncc_mask(prev_gray: np.ndarray, curr_gray: np.ndarray, prev_points: np.ndarray, curr_points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    patch_size = max(5, int(args.klt_patch_size))
    if patch_size % 2 == 0:
        patch_size += 1
    radius = patch_size // 2
    mask = np.zeros(len(curr_points), dtype=bool)
    for index, (prev_point, curr_point) in enumerate(zip(prev_points, curr_points)):
        prev_patch = extract_patch(prev_gray, prev_point, radius)
        curr_patch = extract_patch(curr_gray, curr_point, radius)
        if prev_patch is None or curr_patch is None:
            continue
        prev_patch -= float(np.mean(prev_patch))
        curr_patch -= float(np.mean(curr_patch))
        denom = float(np.linalg.norm(prev_patch) * np.linalg.norm(curr_patch))
        if denom < 1e-9:
            continue
        mask[index] = float(np.sum(prev_patch * curr_patch) / denom) >= args.klt_patch_ncc_threshold
    return mask


def extract_patch(gray: np.ndarray, point: np.ndarray, radius: int) -> np.ndarray | None:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    height, width = gray.shape
    if x - radius < 0 or x + radius >= width or y - radius < 0 or y + radius >= height:
        return None
    return gray[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float32)


def local_flow_consistency_mask(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    base_mask: np.ndarray,
    shape: tuple[int, int],
    args: argparse.Namespace,
) -> np.ndarray:
    height, width = shape
    rows = max(1, int(args.grid_rows))
    cols = max(1, int(args.grid_cols))
    flow = curr_points - prev_points
    keep = base_mask.copy()
    for row in range(rows):
        y0 = row * height / rows
        y1 = (row + 1) * height / rows
        for col in range(cols):
            x0 = col * width / cols
            x1 = (col + 1) * width / cols
            cell = (
                base_mask
                & (prev_points[:, 0] >= x0)
                & (prev_points[:, 0] < x1)
                & (prev_points[:, 1] >= y0)
                & (prev_points[:, 1] < y1)
            )
            indices = np.flatnonzero(cell)
            if len(indices) < args.klt_local_flow_min_points:
                continue
            cell_flow = flow[indices]
            median_flow = np.median(cell_flow, axis=0)
            residual = np.linalg.norm(cell_flow - median_flow, axis=1)
            median_residual = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median_residual)))
            threshold = max(args.klt_local_flow_min_abs, args.klt_local_flow_mad_factor * 1.4826 * mad)
            keep[indices] &= residual <= threshold
    return keep


def filter_temporal_matches(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    camera_matrix: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if len(prev_points) < 8 or len(curr_points) < 8:
        return np.ones(len(curr_points), dtype=bool)
    try:
        if args.temporal_ransac == "essential":
            _, inlier_mask = cv2.findEssentialMat(
                prev_points.astype(np.float32),
                curr_points.astype(np.float32),
                camera_matrix,
                method=cv2.RANSAC,
                prob=args.temporal_ransac_confidence,
                threshold=args.temporal_ransac_threshold,
            )
        elif args.temporal_ransac == "fundamental":
            _, inlier_mask = cv2.findFundamentalMat(
                prev_points.astype(np.float32),
                curr_points.astype(np.float32),
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=args.temporal_ransac_threshold,
                confidence=args.temporal_ransac_confidence,
            )
        else:
            return np.ones(len(curr_points), dtype=bool)
    except cv2.error:
        return np.ones(len(curr_points), dtype=bool)
    if inlier_mask is None:
        return np.ones(len(curr_points), dtype=bool)
    mask = inlier_mask.reshape(-1).astype(bool)
    if len(mask) != len(curr_points) or int(np.count_nonzero(mask)) < args.min_temporal_inliers:
        return np.ones(len(curr_points), dtype=bool)
    return mask


def match_stereo(left_gray: np.ndarray, right_gray: np.ndarray, left_points: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if len(left_points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)
    lk_kwargs = {
        "winSize": (args.klt_window, args.klt_window),
        "maxLevel": args.klt_levels,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            args.klt_iterations,
            args.klt_eps,
        ),
    }
    left_in = left_points.reshape(-1, 1, 2).astype(np.float32)
    right, status, err = cv2.calcOpticalFlowPyrLK(left_gray, right_gray, left_in, None, **lk_kwargs)
    if right is None or status is None:
        return np.empty((0, 2), dtype=np.float32), np.zeros(len(left_points), dtype=bool)
    back, back_status, back_err = cv2.calcOpticalFlowPyrLK(right_gray, left_gray, right, None, **lk_kwargs)
    if back is None or back_status is None:
        return right.reshape(-1, 2), np.zeros(len(left_points), dtype=bool)

    right2 = right.reshape(-1, 2).astype(np.float32)
    back2 = back.reshape(-1, 2).astype(np.float32)
    status = status.reshape(-1).astype(bool)
    back_status = back_status.reshape(-1).astype(bool)
    fb = np.linalg.norm(left_points.reshape(-1, 2) - back2, axis=1)
    disparity = left_points[:, 0] - right2[:, 0]
    epi = np.abs(left_points[:, 1] - right2[:, 1])
    finite = finite_points(right2, right_gray.shape)
    mask = (
        status
        & back_status
        & finite
        & (fb <= args.stereo_fb_threshold)
        & (epi <= args.epipolar_threshold)
        & (disparity >= args.min_disparity)
        & (disparity <= args.max_disparity)
    )
    if args.stereo_max_error > 0 and err is not None and back_err is not None:
        forward_error = err.reshape(-1)
        backward_error = back_err.reshape(-1)
        mask &= np.isfinite(forward_error) & np.isfinite(backward_error)
        mask &= (forward_error <= args.stereo_max_error) & (backward_error <= args.stereo_max_error)
    if args.stereo_disparity_consistency:
        mask &= disparity_consistency_mask(left_points.reshape(-1, 2), disparity, mask, left_gray.shape, args)
    return right2, mask


def klt_stereo_right_points(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    args: argparse.Namespace,
) -> dict[int, np.ndarray]:
    if len(left_points) == 0:
        return {}
    right_points, stereo_mask = match_stereo(left_gray, right_gray, left_points, args)
    stereo_indices = np.flatnonzero(stereo_mask)
    if len(stereo_indices) == 0:
        return {}
    _, depth_mask = triangulate_rectified(
        left_points[stereo_indices],
        right_points[stereo_indices],
        fx,
        fy,
        cx,
        cy,
        baseline,
        args,
    )
    valid_indices = stereo_indices[depth_mask]
    return {int(index): right_points[int(index)].astype(np.float32) for index in valid_indices}


def filter_quad_stereo_temporal_matches(
    prev_right: np.ndarray,
    curr_left: np.ndarray,
    curr_right: np.ndarray,
    kept_ids: list[int],
    curr_left_points: np.ndarray,
    tracks: dict[int, Track],
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int, bool]:
    if len(kept_ids) == 0:
        return np.empty((0,), dtype=bool), 0, 0, False
    available_indices = [index for index, track_id in enumerate(kept_ids) if tracks[track_id].right_point is not None]
    if len(available_indices) < QUAD_MIN_CANDIDATES:
        return np.ones(len(kept_ids), dtype=bool), len(available_indices), len(available_indices), False

    prev_right_points = np.array(
        [tracks[kept_ids[index]].right_point for index in available_indices],
        dtype=np.float32,
    )
    current_left_subset = curr_left_points[available_indices].astype(np.float32)
    right_temporal = track_klt(prev_right, curr_right, prev_right_points, args)
    current_right_from_stereo, current_stereo_mask = match_stereo(curr_left, curr_right, current_left_subset, args)
    closure_error = np.linalg.norm(right_temporal["curr"] - current_right_from_stereo, axis=1)
    contradictory = (
        right_temporal["mask"]
        & current_stereo_mask
        & np.isfinite(closure_error)
        & (closure_error > QUAD_RIGHT_CLOSURE_THRESHOLD_PX)
    )
    local_keep = ~contradictory
    keep = np.zeros(len(kept_ids), dtype=bool)
    for local_index, original_index in enumerate(available_indices):
        keep[original_index] = bool(local_keep[local_index])
    missing_evidence_indices = set(range(len(kept_ids))) - set(available_indices)
    for original_index in missing_evidence_indices:
        keep[original_index] = True
    kept_count = int(np.count_nonzero(keep))
    if kept_count < QUAD_MIN_TRACKS_AFTER_REJECTION:
        return np.ones(len(kept_ids), dtype=bool), len(available_indices), kept_count, False
    return keep, len(available_indices), kept_count, True


def disparity_consistency_mask(
    left_points: np.ndarray,
    disparity: np.ndarray,
    base_mask: np.ndarray,
    shape: tuple[int, int],
    args: argparse.Namespace,
) -> np.ndarray:
    height, width = shape
    rows = max(1, int(args.grid_rows))
    cols = max(1, int(args.grid_cols))
    keep = base_mask.copy()
    for row in range(rows):
        y0 = row * height / rows
        y1 = (row + 1) * height / rows
        for col in range(cols):
            x0 = col * width / cols
            x1 = (col + 1) * width / cols
            cell = (
                base_mask
                & (left_points[:, 0] >= x0)
                & (left_points[:, 0] < x1)
                & (left_points[:, 1] >= y0)
                & (left_points[:, 1] < y1)
            )
            indices = np.flatnonzero(cell)
            if len(indices) < args.stereo_disparity_min_points:
                continue
            values = disparity[indices]
            median = float(np.median(values))
            residual = np.abs(values - median)
            mad = float(np.median(np.abs(residual - np.median(residual))))
            threshold = max(args.stereo_disparity_min_abs, args.stereo_disparity_mad_factor * 1.4826 * mad)
            keep[indices] &= residual <= threshold
    return keep


def match_stereo_sparse_ncc(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if len(left_points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)
    patch_size = max(5, int(args.stereo_ncc_patch_size))
    if patch_size % 2 == 0:
        patch_size += 1
    radius = patch_size // 2
    y_radius = max(0, int(args.stereo_ncc_y_radius))
    min_disparity = max(1, int(math.ceil(args.min_disparity)))
    max_disparity = max(min_disparity, int(math.floor(args.max_disparity)))
    min_score = float(args.stereo_ncc_min_score)
    min_margin = float(args.stereo_ncc_min_margin)
    min_std = float(args.stereo_ncc_min_std)
    height, width = left_gray.shape
    left_float = left_gray.astype(np.float32)
    right_float = right_gray.astype(np.float32)

    right_points = np.zeros((len(left_points), 2), dtype=np.float32)
    mask = np.zeros(len(left_points), dtype=bool)
    for index, point in enumerate(left_points.reshape(-1, 2)):
        x, y = point
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi - radius < 0 or xi + radius >= width or yi - radius < 0 or yi + radius >= height:
            continue
        template = left_float[yi - radius : yi + radius + 1, xi - radius : xi + radius + 1]
        if float(template.std()) < min_std:
            continue
        best_score = -2.0
        second_score = -2.0
        best_x = 0
        best_y = 0
        max_disp_for_point = min(max_disparity, xi - radius)
        if max_disp_for_point < min_disparity:
            continue
        for dy in range(-y_radius, y_radius + 1):
            yr = yi + dy
            if yr - radius < 0 or yr + radius >= height:
                continue
            x_min = max(radius, xi - max_disp_for_point)
            x_max = xi - min_disparity
            if x_max < x_min:
                continue
            strip = right_float[yr - radius : yr + radius + 1, x_min - radius : x_max + radius + 1]
            if strip.shape[0] != patch_size or strip.shape[1] < patch_size:
                continue
            scores = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED).reshape(-1)
            if len(scores) == 0:
                continue
            order = np.argsort(scores)
            local_best_index = int(order[-1])
            local_best_score = float(scores[local_best_index])
            if len(order) > 1:
                local_second_score = float(scores[int(order[-2])])
            else:
                local_second_score = -2.0
            if local_best_score > best_score:
                second_score = max(second_score, best_score, local_second_score)
                best_score = local_best_score
                best_x = x_min + local_best_index
                best_y = yr
            else:
                second_score = max(second_score, local_best_score)
        if best_score < min_score or best_score - second_score < min_margin:
            continue
        disparity = float(xi - best_x)
        if disparity < args.min_disparity or disparity > args.max_disparity:
            continue
        right_points[index] = (float(best_x), float(best_y))
        mask[index] = True
    if args.stereo_disparity_consistency:
        disparity = left_points[:, 0] - right_points[:, 0]
        mask &= disparity_consistency_mask(left_points.reshape(-1, 2), disparity, mask, left_gray.shape, args)
    return right_points, mask


def triangulate_rectified(
    left_points: np.ndarray,
    right_points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if len(left_points) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=bool)
    disparity = left_points[:, 0] - right_points[:, 0]
    z = fx * baseline / np.maximum(disparity, 1e-9)
    x = (left_points[:, 0] - cx) * z / fx
    y = (left_points[:, 1] - cy) * z / fy
    points = np.column_stack([x, y, z]).astype(np.float64)
    points *= args.depth_scale
    mask = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= args.min_depth)
        & (points[:, 2] <= args.max_depth)
    )
    return points[mask], mask


def estimate_stereo_depths(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if len(left_points) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=bool), {
            "valid": 0,
            "klt": 0,
            "clahe_klt": 0,
            "ncc": 0,
            "sgbm": 0,
        }

    points_3d = np.zeros((len(left_points), 3), dtype=np.float64)
    valid = np.zeros(len(left_points), dtype=bool)
    klt_count = 0
    clahe_klt_count = 0
    ncc_count = 0
    sgbm_count = 0

    if args.stereo_depth_mode in {"klt", "hybrid"}:
        right_points, stereo_mask = match_stereo(left_gray, right_gray, left_points, args)
        stereo_indices = np.flatnonzero(stereo_mask)
        if len(stereo_indices):
            klt_points_3d, depth_mask = triangulate_rectified(
                left_points[stereo_indices],
                right_points[stereo_indices],
                fx,
                fy,
                cx,
                cy,
                baseline,
                args,
            )
            valid_indices = stereo_indices[depth_mask]
            points_3d[valid_indices] = klt_points_3d
            valid[valid_indices] = True
            klt_count = len(valid_indices)
        if args.stereo_clahe_fallback_min_klt > 0 and klt_count < args.stereo_clahe_fallback_min_klt:
            missing_indices = np.flatnonzero(~valid)
            if len(missing_indices):
                clahe_left = apply_clahe_gray(
                    left_gray,
                    args.stereo_clahe_fallback_clip_limit,
                    args.stereo_clahe_fallback_tile_grid,
                )
                clahe_right = apply_clahe_gray(
                    right_gray,
                    args.stereo_clahe_fallback_clip_limit,
                    args.stereo_clahe_fallback_tile_grid,
                )
                right_points, stereo_mask = match_stereo(
                    clahe_left,
                    clahe_right,
                    left_points[missing_indices],
                    args,
                )
                stereo_indices = missing_indices[stereo_mask]
                if len(stereo_indices):
                    fallback_points_3d, depth_mask = triangulate_rectified(
                        left_points[stereo_indices],
                        right_points[stereo_mask],
                        fx,
                        fy,
                        cx,
                        cy,
                        baseline,
                        args,
                    )
                    valid_indices = stereo_indices[depth_mask]
                    points_3d[valid_indices] = fallback_points_3d
                    valid[valid_indices] = True
                    clahe_klt_count = len(valid_indices)
        if args.stereo_ncc_fallback_min_klt > 0 and int(np.sum(valid)) < args.stereo_ncc_fallback_min_klt:
            missing_indices = np.flatnonzero(~valid)
            if len(missing_indices):
                right_points, stereo_mask = match_stereo_sparse_ncc(
                    left_gray,
                    right_gray,
                    left_points[missing_indices],
                    args,
                )
                stereo_indices = missing_indices[stereo_mask]
                if len(stereo_indices):
                    ncc_points_3d, depth_mask = triangulate_rectified(
                        left_points[stereo_indices],
                        right_points[stereo_mask],
                        fx,
                        fy,
                        cx,
                        cy,
                        baseline,
                        args,
                    )
                    valid_indices = stereo_indices[depth_mask]
                    points_3d[valid_indices] = ncc_points_3d
                    valid[valid_indices] = True
                    ncc_count = len(valid_indices)

    use_sgbm = args.stereo_depth_mode == "sgbm" or (
        args.stereo_depth_mode == "hybrid" and klt_count < args.sgbm_fallback_min_klt
    )
    if use_sgbm:
        missing_indices = np.flatnonzero(~valid) if args.stereo_depth_mode == "hybrid" else np.arange(len(left_points))
        if len(missing_indices):
            disparity = compute_sgbm_disparity(left_gray, right_gray, args)
            sgbm_points_3d, sgbm_mask = triangulate_from_disparity_map(
                left_points[missing_indices],
                disparity,
                fx,
                fy,
                cx,
                cy,
                baseline,
                args,
            )
            valid_indices = missing_indices[sgbm_mask]
            points_3d[valid_indices] = sgbm_points_3d
            valid[valid_indices] = True
            sgbm_count = len(valid_indices)

    return points_3d[valid], valid, {
        "valid": int(np.sum(valid)),
        "klt": klt_count,
        "clahe_klt": clahe_klt_count,
        "ncc": ncc_count,
        "sgbm": sgbm_count,
    }


def compute_sgbm_disparity(left_gray: np.ndarray, right_gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    num_disparities = max(16, int(math.ceil(args.sgbm_num_disparities / 16.0) * 16))
    block_size = int(args.sgbm_block_size)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=2,
        uniquenessRatio=args.sgbm_uniqueness,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0


def triangulate_from_disparity_map(
    left_points: np.ndarray,
    disparity: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if len(left_points) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=bool)
    points: list[np.ndarray] = []
    valid: list[bool] = []
    radius = max(0, int(args.sgbm_sample_radius))
    height, width = disparity.shape
    for point in left_points:
        x, y = point
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            valid.append(False)
            continue
        x0 = max(0, xi - radius)
        x1 = min(width, xi + radius + 1)
        y0 = max(0, yi - radius)
        y1 = min(height, yi + radius + 1)
        patch = disparity[y0:y1, x0:x1]
        center = float(disparity[yi, xi])
        center_valid = (
            math.isfinite(center)
            and center >= args.min_disparity
            and center <= args.max_disparity
        )
        if args.sgbm_require_center_valid and not center_valid:
            valid.append(False)
            continue
        good = patch[
            np.isfinite(patch)
            & (patch >= args.min_disparity)
            & (patch <= args.max_disparity)
        ]
        if len(good) == 0:
            valid.append(False)
            continue
        if args.sgbm_min_patch_valid_ratio > 0.0:
            valid_ratio = float(len(good)) / max(1, int(patch.size))
            if valid_ratio < args.sgbm_min_patch_valid_ratio:
                valid.append(False)
                continue
        disp = float(np.median(good))
        if args.sgbm_patch_max_mad > 0.0:
            mad = float(np.median(np.abs(good - disp)))
            if mad * 1.4826 > args.sgbm_patch_max_mad:
                valid.append(False)
                continue
        if args.sgbm_center_max_diff > 0.0:
            if not center_valid or abs(center - disp) > args.sgbm_center_max_diff:
                valid.append(False)
                continue
        z = fx * baseline / max(disp, 1e-9)
        if z < args.min_depth or z > args.max_depth:
            valid.append(False)
            continue
        points.append(np.array([(x - cx) * z / fx, (y - cy) * z / fy, z], dtype=np.float64) * args.depth_scale)
        valid.append(True)
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.array(valid, dtype=bool)
    return np.array(points, dtype=np.float64), np.array(valid, dtype=bool)


def track_depth_update_passes(previous_point: np.ndarray, current_point: np.ndarray, args: argparse.Namespace) -> bool:
    if args.track_depth_max_abs_change <= 0.0 and args.track_depth_max_rel_change <= 0.0:
        return True
    if not np.isfinite(previous_point).all() or not np.isfinite(current_point).all():
        return False
    previous_depth = float(previous_point[2])
    current_depth = float(current_point[2])
    if previous_depth <= 1e-9 or current_depth <= 1e-9:
        return False
    delta = abs(current_depth - previous_depth)
    if args.track_depth_max_abs_change > 0.0 and delta > args.track_depth_max_abs_change:
        return False
    if args.track_depth_max_rel_change > 0.0 and delta / max(abs(previous_depth), 1e-9) > args.track_depth_max_rel_change:
        return False
    return True


def track_camera_point_passes(point: np.ndarray, args: argparse.Namespace) -> bool:
    if not np.isfinite(point).all():
        return False
    depth = float(point[2])
    return args.min_depth <= depth <= args.max_depth


def solve_stereo3d_motion(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    if len(previous_points) < max(6, args.min_stereo3d_inliers) or len(current_points) != len(previous_points):
        return None
    valid = np.isfinite(previous_points).all(axis=1) & np.isfinite(current_points).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < max(6, args.min_stereo3d_inliers):
        return None
    prev = previous_points[indices].astype(np.float64)
    curr = current_points[indices].astype(np.float64)
    rng = np.random.default_rng(7)
    best_local_inliers: np.ndarray | None = None
    best_rmse = float("inf")
    sample_size = 3
    iterations = max(1, int(args.stereo3d_iterations))
    threshold = max(1e-6, float(args.stereo3d_threshold))

    for _ in range(iterations):
        if len(prev) < sample_size:
            break
        sample = rng.choice(len(prev), size=sample_size, replace=False)
        candidate = estimate_rigid_transform(prev[sample], curr[sample])
        if candidate is None:
            continue
        rotation, translation = candidate
        residual = np.linalg.norm((rotation @ prev.T).T + translation - curr, axis=1)
        local_inliers = np.flatnonzero(residual <= threshold)
        if len(local_inliers) < args.min_stereo3d_inliers:
            continue
        rmse = float(math.sqrt(np.mean(residual[local_inliers] ** 2)))
        if best_local_inliers is None or len(local_inliers) > len(best_local_inliers) or (
            len(local_inliers) == len(best_local_inliers) and rmse < best_rmse
        ):
            best_local_inliers = local_inliers
            best_rmse = rmse

    if best_local_inliers is None or len(best_local_inliers) < args.min_stereo3d_inliers:
        return None
    refined = estimate_rigid_transform(prev[best_local_inliers], curr[best_local_inliers])
    if refined is None:
        return None
    rotation, translation = refined
    residual = np.linalg.norm((rotation @ prev.T).T + translation - curr, axis=1)
    local_inliers = np.flatnonzero(residual <= threshold)
    if len(local_inliers) < args.min_stereo3d_inliers:
        return None
    rmse = float(math.sqrt(np.mean(residual[local_inliers] ** 2)))
    return rotation.astype(np.float64), translation.astype(np.float64), indices[local_inliers].astype(np.int32), rmse


def estimate_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(source) < 3 or len(target) != len(source):
        return None
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    if np.linalg.matrix_rank(source_centered) < 2:
        return None
    covariance = source_centered.T @ target_centered
    try:
        u, _, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return None
    return rotation, translation


def solve_pnp_motion(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    args: argparse.Namespace,
    expected_rotation: np.ndarray | None = None,
    underwater_context: UnderwaterPnpContext | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    object_points = object_points.astype(np.float32)
    image_points = image_points.astype(np.float32)
    candidates: list[tuple[int, float, float, float, np.ndarray, np.ndarray, np.ndarray]] = []
    attempts = max(1, int(args.pnp_ransac_attempts))
    base_seed = int(args.opencv_rng_seed) if int(args.opencv_rng_seed) >= 0 else 7
    for attempt in range(attempts):
        seed = (base_seed + attempt * 1009 + len(object_points) * 9176) & 0x7FFFFFFF
        if int(args.opencv_rng_seed) >= 0 or attempt > 0:
            cv2.setRNGSeed(seed)
        if attempt == 0:
            permutation = np.arange(len(object_points), dtype=np.int32)
        else:
            permutation = np.random.default_rng(seed).permutation(len(object_points)).astype(np.int32)
        object_attempt = object_points[permutation]
        image_attempt = image_points[permutation]
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_attempt,
                image_attempt,
                camera_matrix,
                None,
                iterationsCount=args.pnp_iterations,
                reprojectionError=args.pnp_threshold,
                confidence=args.pnp_confidence,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            continue
        if not ok or rvec is None or tvec is None or inliers is None:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        inliers = permutation[inliers.reshape(-1).astype(np.int32)]
        rmse = reprojection_rmse(object_points, image_points, camera_matrix, rvec, tvec, inliers)
        angle = rotation_angle(rotation.astype(np.float64))
        imu_error = 0.0
        if args.pnp_use_imu_prior and expected_rotation is not None:
            imu_error = rotation_angle(rotation.astype(np.float64) @ expected_rotation.T)
        candidates.append(
            (
                len(inliers),
                rmse,
                angle,
                imu_error,
                rotation.astype(np.float64),
                tvec.reshape(3).astype(np.float64),
                inliers,
            )
        )
    if not candidates:
        return None
    max_inliers = max(candidate[0] for candidate in candidates)
    inlier_floor = max(0, max_inliers - max(0, int(args.pnp_inlier_slack)))
    close_candidates = [candidate for candidate in candidates if candidate[0] >= inlier_floor]
    if args.underwater_pnp and args.underwater_pnp_rank_candidates and underwater_context is not None:
        best = min(
            close_candidates,
            key=lambda candidate: underwater_pnp_candidate_sort_key(
                object_points,
                image_points,
                camera_matrix,
                candidate,
                expected_rotation,
                underwater_context,
                args,
            ),
        )
    elif args.pnp_use_imu_prior and expected_rotation is not None:
        best = min(close_candidates, key=lambda candidate: (candidate[3], candidate[1], candidate[2], -candidate[0]))
    else:
        best = min(close_candidates, key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
    return best[4], best[5], best[6]


def reprojection_rmse(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    inliers: np.ndarray,
) -> float:
    indices = inliers.reshape(-1).astype(int)
    indices = indices[(indices >= 0) & (indices < len(object_points))]
    if len(indices) == 0:
        return float("inf")
    try:
        projected, _ = cv2.projectPoints(object_points[indices], rvec, tvec, camera_matrix, None)
    except cv2.error:
        return float("inf")
    residual = projected.reshape(-1, 2).astype(np.float64) - image_points[indices].astype(np.float64)
    return float(math.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def valid_inlier_indices(inliers: np.ndarray, length: int) -> np.ndarray:
    if inliers is None or len(inliers) == 0:
        return np.empty((0,), dtype=np.int32)
    indices = inliers.reshape(-1).astype(np.int32)
    return indices[(indices >= 0) & (indices < length)]


def evaluate_underwater_pnp_quality(
    curr_left: np.ndarray,
    curr_right: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    inliers: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    expected_rotation: np.ndarray | None,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    args: argparse.Namespace,
    pnp_ids: list[int] | None = None,
    tracks: dict[int, Track] | None = None,
) -> UnderwaterPnpQuality:
    indices = valid_inlier_indices(inliers, len(object_points))
    if len(indices) == 0:
        return UnderwaterPnpQuality()

    rotation = rotation.astype(np.float64)
    translation = translation.reshape(3).astype(np.float64)
    object_subset = object_points[indices].astype(np.float64)
    image_subset = image_points[indices].astype(np.float64)
    if not (np.isfinite(rotation).all() and np.isfinite(translation).all() and np.isfinite(object_subset).all() and np.isfinite(image_subset).all()):
        return UnderwaterPnpQuality()
    if np.max(np.abs(rotation)) > 1e6 or np.max(np.abs(object_subset)) > 1e6 or np.max(np.abs(translation)) > 1e6:
        return UnderwaterPnpQuality()
    try:
        rvec, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(object_subset, rvec, translation.reshape(3, 1), camera_matrix, None)
        residual_px = projected.reshape(-1, 2).astype(np.float64) - image_subset
        reproj_rmse = float(math.sqrt(np.mean(np.sum(residual_px * residual_px, axis=1))))
    except cv2.error:
        reproj_rmse = float("inf")

    imu_error = 0.0
    if expected_rotation is not None:
        imu_error = rotation_angle(rotation @ expected_rotation.T)

    stereo_points_3d, stereo_mask, _ = estimate_stereo_depths(
        curr_left,
        curr_right,
        image_points[indices].astype(np.float32),
        fx,
        fy,
        cx,
        cy,
        baseline,
        args,
    )
    stereo_confirmed = int(np.sum(stereo_mask))
    stereo_ratio = stereo_confirmed / max(1, len(indices))
    depth_median = float("inf")
    depth_p90 = float("inf")
    xyz_median = float("inf")
    if stereo_confirmed > 0:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            predicted_current = (rotation @ object_subset.T).T + translation
        if not np.isfinite(predicted_current).all():
            predicted_current = np.empty((0, 3), dtype=np.float64)
            stereo_points_3d = np.empty((0, 3), dtype=np.float64)
            stereo_mask = np.zeros(len(stereo_mask), dtype=bool)
            stereo_confirmed = 0
            stereo_ratio = 0.0
        else:
            predicted_valid = predicted_current[stereo_mask]
            depth_residual = np.abs(predicted_valid[:, 2] - stereo_points_3d[:, 2])
            xyz_residual = np.linalg.norm(predicted_valid - stereo_points_3d, axis=1)
            if len(depth_residual):
                depth_median = float(np.median(depth_residual))
                depth_p90 = float(np.percentile(depth_residual, 90))
            if len(xyz_residual):
                xyz_median = float(np.median(xyz_residual))

    track_age_penalty = 0.0
    if pnp_ids is not None and tracks is not None:
        ages = [
            tracks[pnp_ids[int(index)]].age
            for index in indices
            if int(index) < len(pnp_ids) and pnp_ids[int(index)] in tracks
        ]
        if ages:
            median_age = float(np.median(np.array(ages, dtype=np.float64)))
            track_age_penalty = 1.0 / max(1.0, median_age)
        else:
            median_age = 0.0
    else:
        median_age = 0.0

    reproj_score = reproj_rmse / max(float(args.underwater_max_reprojection_rmse_px), 1e-9)
    imu_score = imu_error / max(float(args.underwater_max_imu_rotation_error_rad), 1e-9)
    depth_score = 0.0
    if args.underwater_max_depth_residual_m > 0.0:
        depth_score = depth_median / max(float(args.underwater_max_depth_residual_m), 1e-9)
    xyz_score = 0.0
    if args.underwater_max_xyz_residual_m > 0.0:
        xyz_score = xyz_median / max(float(args.underwater_max_xyz_residual_m), 1e-9)
    stereo_shortfall = max(0.0, float(args.underwater_min_stereo_inlier_ratio) - stereo_ratio)
    stereo_score = stereo_shortfall / max(float(args.underwater_min_stereo_inlier_ratio), 1e-9)
    score = reproj_score + 0.75 * imu_score + depth_score + xyz_score + 1.25 * stereo_score + 0.1 * track_age_penalty
    return UnderwaterPnpQuality(
        reprojection_rmse_px=reproj_rmse,
        imu_rotation_error_rad=imu_error,
        stereo_confirmed=stereo_confirmed,
        stereo_inlier_ratio=float(stereo_ratio),
        depth_median_abs_residual_m=depth_median,
        depth_p90_abs_residual_m=depth_p90,
        xyz_median_residual_m=xyz_median,
        score=float(score),
    )


def underwater_pnp_quality_passes(quality: UnderwaterPnpQuality, args: argparse.Namespace) -> tuple[bool, str]:
    failures: list[str] = []
    if quality.reprojection_rmse_px > args.underwater_max_reprojection_rmse_px:
        failures.append("reprojection")
    if quality.imu_rotation_error_rad > args.underwater_max_imu_rotation_error_rad:
        failures.append("imu")
    if quality.stereo_inlier_ratio < args.underwater_min_stereo_inlier_ratio:
        failures.append("stereo_ratio")
    if args.underwater_max_depth_residual_m > 0.0 and quality.depth_median_abs_residual_m > args.underwater_max_depth_residual_m:
        failures.append("depth_residual")
    if args.underwater_max_xyz_residual_m > 0.0 and quality.xyz_median_residual_m > args.underwater_max_xyz_residual_m:
        failures.append("xyz_residual")
    return len(failures) == 0, ",".join(failures)


def underwater_pnp_candidate_sort_key(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    candidate: tuple[int, float, float, float, np.ndarray, np.ndarray, np.ndarray],
    expected_rotation: np.ndarray | None,
    context: UnderwaterPnpContext,
    args: argparse.Namespace,
) -> tuple[float, float, float, int]:
    quality = evaluate_underwater_pnp_quality(
        context.curr_left,
        context.curr_right,
        object_points,
        image_points,
        candidate[6],
        candidate[4],
        candidate[5],
        camera_matrix,
        expected_rotation,
        context.fx,
        context.fy,
        context.cx,
        context.cy,
        context.baseline,
        args,
    )
    return (quality.score, quality.imu_rotation_error_rad, quality.reprojection_rmse_px, -candidate[0])


def evaluate_frontend_health(
    tracked_count: int,
    pnp_inliers: int,
    current_stereo_confirmed: int,
    quad_candidates: int,
    quad_kept: int,
    pose_success: bool,
    args: argparse.Namespace,
) -> FrontendHealth:
    track_ratio = min(1.0, tracked_count / max(float(args.min_features), 1.0))
    inlier_ratio = min(1.0, pnp_inliers / max(float(args.min_pnp_inliers), 1.0))
    stereo_ratio = 1.0
    if pnp_inliers > 0 and current_stereo_confirmed > 0:
        stereo_ratio = min(1.0, current_stereo_confirmed / max(float(pnp_inliers), 1.0))
    elif pnp_inliers > 0:
        stereo_ratio = 0.0
    quad_reject_ratio = 0.0
    if quad_candidates > 0:
        quad_reject_ratio = max(0.0, min(1.0, (quad_candidates - quad_kept) / max(float(quad_candidates), 1.0)))
    score = 0.35 * track_ratio + 0.35 * inlier_ratio + 0.20 * stereo_ratio + 0.10 * (1.0 - quad_reject_ratio)
    if not pose_success or pnp_inliers == 0:
        state = "lost" if tracked_count < args.min_pnp_inliers else "degraded"
    elif score >= 0.75:
        state = "good"
    elif score >= 0.45:
        state = "degraded"
    else:
        state = "lost"
    return FrontendHealth(
        score=float(score),
        state=state,
        track_ratio=float(track_ratio),
        inlier_ratio=float(inlier_ratio),
        stereo_ratio=float(stereo_ratio),
        quad_reject_ratio=float(quad_reject_ratio),
    )


def format_optional_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.9f}"


def motion_gate_passes(speed: float, step_norm: float, args: argparse.Namespace) -> bool:
    speed_ok = speed <= args.max_speed_mps
    step_ok = step_norm <= args.max_step_m
    if args.motion_gate_mode == "all":
        return speed_ok and step_ok
    return speed_ok or step_ok


def estimate_visual_coast_rotation(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    camera_matrix: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if len(prev_points) < args.visual_coast_min_tracks or len(curr_points) < args.visual_coast_min_tracks:
        return None
    if len(prev_points) != len(curr_points):
        return None
    prev = prev_points.astype(np.float64)
    curr = curr_points.astype(np.float64)
    try:
        essential, inliers = cv2.findEssentialMat(
            prev,
            curr,
            camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=float(args.visual_coast_ransac_threshold),
        )
    except cv2.error:
        return None
    if essential is None or inliers is None:
        return None
    essential = np.asarray(essential, dtype=np.float64)
    best_rotation: np.ndarray | None = None
    best_count = -1
    for start in range(0, essential.shape[0], 3):
        candidate = essential[start : start + 3, :]
        if candidate.shape != (3, 3):
            continue
        try:
            count, rotation, _, _ = cv2.recoverPose(candidate, prev, curr, camera_matrix, mask=inliers.copy())
        except cv2.error:
            continue
        rotation = rotation.astype(np.float64)
        if count > best_count and rotation_angle(rotation) <= args.visual_coast_max_rotation_rad:
            best_count = int(count)
            best_rotation = rotation
    if best_rotation is None or best_count < args.visual_coast_min_tracks:
        return None
    return best_rotation


def estimate_flow_yaw_coast_rotation(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    fx: float,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if len(prev_points) < args.visual_coast_min_tracks or len(curr_points) < args.visual_coast_min_tracks:
        return None
    if len(prev_points) != len(curr_points) or fx <= 1e-9:
        return None
    flow = curr_points.astype(np.float64) - prev_points.astype(np.float64)
    finite = np.isfinite(flow).all(axis=1)
    if int(np.count_nonzero(finite)) < args.visual_coast_min_tracks:
        return None
    yaw = -float(np.median(flow[finite, 0])) / float(fx)
    yaw *= float(args.flow_yaw_coast_scale)
    limit = abs(float(args.flow_yaw_coast_max_rad))
    if not math.isfinite(yaw) or abs(yaw) > limit:
        return None
    return exp_so3(np.array([0.0, yaw, 0.0], dtype=np.float64))


def count_current_stereo_confirmed(
    curr_left: np.ndarray,
    curr_right: np.ndarray,
    image_points: np.ndarray,
    inliers: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    args: argparse.Namespace,
) -> int:
    if len(image_points) == 0 or len(inliers) == 0:
        return 0
    indices = inliers.reshape(-1).astype(int)
    indices = indices[(indices >= 0) & (indices < len(image_points))]
    if len(indices) == 0:
        return 0
    _, _, stats = estimate_stereo_depths(
        curr_left,
        curr_right,
        image_points[indices].astype(np.float32),
        fx,
        fy,
        cx,
        cy,
        baseline,
        args,
    )
    return int(stats["valid"])


def solve_translation_fixed_rotation(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    inliers: np.ndarray,
) -> np.ndarray | None:
    if len(object_points) < 6:
        return None
    indices = inliers.reshape(-1).astype(int) if len(inliers) else np.arange(len(object_points))
    indices = indices[(indices >= 0) & (indices < len(object_points))]
    if len(indices) < 6:
        return None

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    rows = []
    rhs = []
    for index in indices:
        u, v = image_points[index]
        bearing = np.array([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=np.float64)
        skew = np.array(
            [
                [0.0, -bearing[2], bearing[1]],
                [bearing[2], 0.0, -bearing[0]],
                [-bearing[1], bearing[0], 0.0],
            ],
            dtype=np.float64,
        )
        rotated_point = rotation @ object_points[index].astype(np.float64)
        rows.append(skew[:2])
        rhs.append((-skew @ rotated_point)[:2])
    matrix = np.vstack(rows)
    target = np.concatenate(rhs)
    try:
        translation, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(translation).all():
        return None
    return translation.astype(np.float64)


def apply_underwater_planar_motion_prior(
    raw_step_world: np.ndarray,
    proposed_rotation_world_camera: np.ndarray,
    relative_rotation: np.ndarray,
    dt: float,
    args: argparse.Namespace,
) -> np.ndarray:
    raw_step_world = raw_step_world.reshape(3).astype(np.float64)
    if not np.isfinite(raw_step_world).all():
        return np.zeros(3, dtype=np.float64)

    # OpenCV camera/world convention here: x=right, y=down, z=forward.
    # The pool run is mostly planar, so keep horizontal motion in the x-z
    # plane and damp the visually noisy y component.
    forward_world = proposed_rotation_world_camera @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    lateral_world = proposed_rotation_world_camera @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    forward_world[1] = 0.0
    lateral_world[1] = 0.0
    forward_norm = float(np.linalg.norm(forward_world))
    lateral_norm = float(np.linalg.norm(lateral_world))
    if forward_norm <= 1e-9:
        return raw_step_world
    forward_world /= forward_norm
    if lateral_norm > 1e-9:
        lateral_world /= lateral_norm
    else:
        lateral_world = np.zeros(3, dtype=np.float64)

    horizontal_step = raw_step_world.copy()
    horizontal_step[1] = 0.0
    horizontal_norm = float(np.linalg.norm(horizontal_step))
    forward_component = float(np.dot(horizontal_step, forward_world))
    if abs(forward_component) > 1e-9:
        forward_distance = max(0.0, forward_component)
    else:
        forward_distance = horizontal_norm
    forward_distance *= max(0.0, args.planar_forward_gain)

    rotation_angle_rad = rotation_angle(relative_rotation)
    if args.planar_rotation_slowdown_rad > 1e-9:
        slowdown = 1.0 - rotation_angle_rad / max(args.planar_rotation_slowdown_rad, 1e-9)
        slowdown = max(args.planar_min_turn_speed_scale, min(1.0, slowdown))
        forward_distance *= slowdown

    normal_limit = min(args.max_step_m, args.max_speed_mps * max(dt, 1e-6))
    if args.planar_max_step_m > 0.0:
        normal_limit = min(normal_limit, args.planar_max_step_m)
    if normal_limit > 0.0:
        forward_distance = min(forward_distance, normal_limit)

    lateral_distance = float(np.dot(horizontal_step, lateral_world)) * args.planar_lateral_gain
    vertical_distance = float(raw_step_world[1]) * args.planar_vertical_gain
    planar_step = forward_world * forward_distance + lateral_world * lateral_distance
    planar_step[1] = vertical_distance
    return planar_step.astype(np.float64)


def integrate_gyro_rotation(
    imu_rows: list[ImuRow],
    start: float,
    end: float,
    gyro_sign: float = 1.0,
    axis: str | None = None,
) -> np.ndarray:
    if end <= start or not imu_rows:
        return np.eye(3, dtype=np.float64)
    selected = [(row.timestamp_sec, row.gyro) for row in imu_rows if start <= row.timestamp_sec <= end]
    if len(selected) < 2:
        return np.eye(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    for (t0, w0), (t1, w1) in zip(selected, selected[1:]):
        dt = max(t1 - t0, 0.0)
        omega = gyro_sign * 0.5 * (w0 + w1)
        if axis is not None:
            axis_index = {"x": 0, "y": 1, "z": 2}[axis]
            masked = np.zeros(3, dtype=np.float64)
            masked[axis_index] = omega[axis_index]
            omega = masked
        rotation = rotation @ exp_so3(omega * dt)
    return rotation


def absolute_imu_relative_rotation(
    imu_rows: list[ImuRow],
    start: float,
    end: float,
    attitude_extrinsic: str,
    attitude_delta: str,
) -> np.ndarray:
    if end <= start or not imu_rows:
        return np.eye(3, dtype=np.float64)
    before = nearest_imu_orientation(imu_rows, start)
    after = nearest_imu_orientation(imu_rows, end)
    if before is None or after is None:
        return np.eye(3, dtype=np.float64)
    rotation_world_imu_prev = quaternion_to_rotation(before)
    rotation_world_imu_curr = quaternion_to_rotation(after)
    if attitude_delta == "prev-to-curr":
        relative_imu = rotation_world_imu_curr.T @ rotation_world_imu_prev
    elif attitude_delta == "curr-to-prev":
        relative_imu = rotation_world_imu_prev.T @ rotation_world_imu_curr
    else:
        raise ValueError(f"Unsupported IMU attitude delta: {attitude_delta}")
    imu_from_camera = imu_camera_extrinsic(attitude_extrinsic)
    return imu_from_camera.T @ relative_imu @ imu_from_camera


def nearest_imu_orientation(imu_rows: list[ImuRow], timestamp_sec: float) -> np.ndarray | None:
    oriented = [row for row in imu_rows if row.orientation is not None]
    if not oriented:
        return None
    nearest = min(oriented, key=lambda row: abs(row.timestamp_sec - timestamp_sec))
    if abs(nearest.timestamp_sec - timestamp_sec) > 0.10:
        return None
    return nearest.orientation


def imu_camera_extrinsic(name: str) -> np.ndarray:
    if name == "identity":
        return np.eye(3, dtype=np.float64)
    if name == "opencv-to-ros":
        return OPENCV_OPTICAL_TO_ROS.copy()
    if name == "ros-to-opencv":
        return OPENCV_OPTICAL_TO_ROS.T.copy()
    raise ValueError(f"Unsupported IMU attitude extrinsic: {name}")


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quaternion.astype(np.float64)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def exp_so3(vec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vec / theta
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def rotation_angle(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, value)))


def finite_points(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )


def preprocess_gray(gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if not args.clahe:
        return gray
    return apply_clahe_gray(gray, args.clahe_clip_limit, args.clahe_tile_grid)


def apply_clahe_gray(gray: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    tile_size = max(2, int(tile_grid))
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(clip_limit)),
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(gray)


def write_odometry_csv(path: Path, rows: list[PoseRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame_index",
                "timestamp_sec",
                "x",
                "y",
                "z",
                "qx",
                "qy",
                "qz",
                "qw",
                "pose_success",
                "pose_inliers",
                "pose_inlier_ratio",
                "active_tracks",
                "stereo_tracks",
                "scale_mode",
                "note",
            ]
        )
        for row in rows:
            qx, qy, qz, qw = rotation_to_quaternion(row.rotation)
            writer.writerow(
                [
                    row.frame_index,
                    f"{row.timestamp_sec:.9f}",
                    f"{row.position[0]:.9f}",
                    f"{row.position[1]:.9f}",
                    f"{row.position[2]:.9f}",
                    f"{qx:.9f}",
                    f"{qy:.9f}",
                    f"{qz:.9f}",
                    f"{qw:.9f}",
                    "1" if row.success else "0",
                    row.pose_inliers,
                    f"{row.pose_inlier_ratio:.6f}",
                    row.active_tracks,
                    row.stereo_tracks,
                    "metric_stereo_pnp_imu_gated",
                    row.note,
                ]
            )


def append_point_cloud_rows(
    rows: list[dict[str, Any]],
    frame_index: int,
    timestamp_sec: float,
    tracks: dict[int, Track],
    args: argparse.Namespace,
    stable_only: bool,
) -> None:
    for track in tracks.values():
        if stable_only and not stable_point_cloud_track_passes(track, args):
            continue
        if track.world_point is None:
            continue
        point = track.world_point
        if not np.isfinite(point).all():
            continue
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": f"{timestamp_sec:.9f}",
                "track_id": track.track_id,
                "x": f"{float(point[0]):.9f}",
                "y": f"{float(point[1]):.9f}",
                "z": f"{float(point[2]):.9f}",
                "u": f"{float(track.left_point[0]):.3f}",
                "v": f"{float(track.left_point[1]):.3f}",
                "age": track.age,
            }
        )


def stable_point_cloud_track_passes(track: Track, args: argparse.Namespace) -> bool:
    if track.age < max(1, int(args.stable_point_cloud_min_age)):
        return False
    if args.stable_point_cloud_fresh_depth_only and track.missing_depth_age != 0:
        return False
    if track.right_point is None:
        return False
    if not np.isfinite(track.point_3d).all():
        return False
    local_depth = float(track.point_3d[2])
    if local_depth < args.min_depth or local_depth > args.max_depth:
        return False
    return track.world_point is not None and np.isfinite(track.world_point).all()


def write_point_cloud_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["frame_index", "timestamp_sec", "track_id", "x", "y", "z", "u", "v", "age"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_debug_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    rows: list[PoseRow],
    debug_rows: list[dict[str, Any]],
    accepted_count: int,
) -> dict[str, Any]:
    path_length = 0.0
    max_step = 0.0
    max_speed = 0.0
    for prev, cur in zip(rows, rows[1:]):
        dt = max(cur.timestamp_sec - prev.timestamp_sec, 1e-6)
        step = float(np.linalg.norm(cur.position - prev.position))
        path_length += step
        max_step = max(max_step, step)
        max_speed = max(max_speed, step / dt)
    active = [row.active_tracks for row in rows]
    inliers = [int(row.get("pnp_inliers", 0)) for row in debug_rows]
    notes: dict[str, int] = {}
    for row in rows:
        notes[row.note] = notes.get(row.note, 0) + 1
    health_counts: dict[str, int] = {}
    for row in debug_rows:
        state = str(row.get("frontend_health_state", "unknown"))
        health_counts[state] = health_counts.get(state, 0) + 1
    _, fx, fy, cx, cy, baseline = camera_from_metadata(metadata, args)
    return {
        "mode": "ros2_native_stereo_imu_vio",
        "estimator_inputs": ["stereo_left", "stereo_right", "imu"],
        "reference_inputs_used": [],
        "dvl_usage": "not_used_in_estimator_reference_only",
        "dataset_dir": str(args.dataset_dir),
        "output_csv": str(args.output_csv),
        "point_cloud_csv": str(args.point_cloud_csv),
        "raw_point_cloud_csv": str(args.raw_point_cloud_csv) if args.raw_point_cloud_csv is not None else "",
        "summary_json": str(args.summary_json),
        "debug_csv": str(args.debug_csv),
        "metadata": {
            "frames": len(rows),
            "fps": metadata.get("fps"),
            "duration_sec": rows[-1].timestamp_sec - rows[0].timestamp_sec if len(rows) > 1 else 0.0,
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "start_frame": metadata.get("start_frame", 0),
            "stride": metadata.get("stride", 1),
            "baseline_m": metadata.get("baseline_m"),
            "imu_source": metadata.get("imu_source"),
            "imu_samples": metadata.get("imu_samples"),
        },
        "effective_camera": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "baseline_m": baseline,
        },
        "pose_count": len(rows),
        "pose_success_count": accepted_count,
        "pose_success_ratio": accepted_count / max(1, len(rows) - 1),
        "trajectory_stats": {
            "rows": float(len(rows)),
            "duration_sec": rows[-1].timestamp_sec - rows[0].timestamp_sec if len(rows) > 1 else 0.0,
            "path_length_m": path_length,
            "max_step_m": max_step,
            "max_speed_mps": max_speed,
            "span_x_m": float(max(row.position[0] for row in rows) - min(row.position[0] for row in rows)),
            "span_y_m": float(max(row.position[1] for row in rows) - min(row.position[1] for row in rows)),
            "span_z_m": float(max(row.position[2] for row in rows) - min(row.position[2] for row in rows)),
        },
        "tracking_stats": {
            "mean_active_tracks": statistics.fmean(active) if active else 0.0,
            "median_active_tracks": statistics.median(active) if active else 0.0,
            "mean_pnp_inliers": statistics.fmean(inliers) if inliers else 0.0,
        },
        "note_counts": notes,
        "frontend_health_counts": health_counts,
        "parameters": {
            "feature_detector": args.feature_detector,
            "max_features": args.max_features,
            "min_features": args.min_features,
            "swap_stereo": args.swap_stereo,
            "clahe": args.clahe,
            "clahe_clip_limit": args.clahe_clip_limit,
            "clahe_tile_grid": args.clahe_tile_grid,
            "mask_bottom_ratio": args.mask_bottom_ratio,
            "mask_bottom_px": args.mask_bottom_px,
            "mask_bright_threshold": args.mask_bright_threshold,
            "mask_bright_dilate": args.mask_bright_dilate,
            "quality": args.quality,
            "min_distance": args.min_distance,
            "fast_threshold": args.fast_threshold,
            "orb_fast_threshold": args.orb_fast_threshold,
            "reject_repeated_patches": args.reject_repeated_patches,
            "patch_size": args.patch_size,
            "patch_search_radius": args.patch_search_radius,
            "patch_ncc_threshold": args.patch_ncc_threshold,
            "grid": [args.grid_rows, args.grid_cols],
            "cell_max_features": args.cell_max_features,
            "feature_refill_interval": args.feature_refill_interval,
            "fb_threshold": args.fb_threshold,
            "klt_max_error": args.klt_max_error,
            "klt_max_flow_px": args.klt_max_flow_px,
            "klt_min_eigenvalue": args.klt_min_eigenvalue,
            "klt_patch_size": args.klt_patch_size,
            "klt_patch_ncc_threshold": args.klt_patch_ncc_threshold,
            "klt_local_flow_check": args.klt_local_flow_check,
            "klt_local_flow_mad_factor": args.klt_local_flow_mad_factor,
            "klt_use_prediction": args.klt_use_prediction,
            "klt_velocity_check": args.klt_velocity_check,
            "klt_max_accel_px": args.klt_max_accel_px,
            "quad_stereo_temporal": "always_on_reject_contradictions_only",
            "quad_right_closure_threshold_px": QUAD_RIGHT_CLOSURE_THRESHOLD_PX,
            "quad_min_candidates": QUAD_MIN_CANDIDATES,
            "quad_min_tracks_after_rejection": QUAD_MIN_TRACKS_AFTER_REJECTION,
            "min_track_age_for_pnp": args.min_track_age_for_pnp,
            "temporal_ransac": args.temporal_ransac,
            "temporal_ransac_threshold": args.temporal_ransac_threshold,
            "min_temporal_inliers": args.min_temporal_inliers,
            "stereo_fb_threshold": args.stereo_fb_threshold,
            "stereo_max_error": args.stereo_max_error,
            "epipolar_threshold": args.epipolar_threshold,
            "stereo_clahe_fallback_min_klt": args.stereo_clahe_fallback_min_klt,
            "stereo_clahe_fallback_clip_limit": args.stereo_clahe_fallback_clip_limit,
            "stereo_clahe_fallback_tile_grid": args.stereo_clahe_fallback_tile_grid,
            "stereo_ncc_fallback_min_klt": args.stereo_ncc_fallback_min_klt,
            "stereo_ncc_patch_size": args.stereo_ncc_patch_size,
            "stereo_ncc_min_score": args.stereo_ncc_min_score,
            "stereo_ncc_min_margin": args.stereo_ncc_min_margin,
            "stereo_ncc_y_radius": args.stereo_ncc_y_radius,
            "stereo_ncc_min_std": args.stereo_ncc_min_std,
            "stereo_disparity_consistency": args.stereo_disparity_consistency,
            "stereo_disparity_mad_factor": args.stereo_disparity_mad_factor,
            "stereo_disparity_min_abs": args.stereo_disparity_min_abs,
            "track_depth_max_abs_change": args.track_depth_max_abs_change,
            "track_depth_max_rel_change": args.track_depth_max_rel_change,
            "stable_point_cloud_min_age": args.stable_point_cloud_min_age,
            "stable_point_cloud_fresh_depth_only": args.stable_point_cloud_fresh_depth_only,
            "focal_scale": args.focal_scale,
            "fx_scale": args.fx_scale,
            "fy_scale": args.fy_scale,
            "cx_offset_px": args.cx_offset_px,
            "cy_offset_px": args.cy_offset_px,
            "baseline_scale": args.baseline_scale,
            "baseline_m_override": args.baseline_m,
            "pnp_threshold": args.pnp_threshold,
            "pnp_ransac_attempts": args.pnp_ransac_attempts,
            "pnp_inlier_slack": args.pnp_inlier_slack,
            "pnp_use_imu_prior": args.pnp_use_imu_prior,
            "opencv_rng_seed": args.opencv_rng_seed,
            "underwater_pnp": args.underwater_pnp,
            "underwater_pnp_rank_candidates": args.underwater_pnp_rank_candidates,
            "underwater_pnp_action": args.underwater_pnp_action,
            "underwater_max_reprojection_rmse_px": args.underwater_max_reprojection_rmse_px,
            "underwater_max_imu_rotation_error_rad": args.underwater_max_imu_rotation_error_rad,
            "underwater_min_stereo_inlier_ratio": args.underwater_min_stereo_inlier_ratio,
            "underwater_max_depth_residual_m": args.underwater_max_depth_residual_m,
            "underwater_max_xyz_residual_m": args.underwater_max_xyz_residual_m,
            "min_pnp_inliers": args.min_pnp_inliers,
            "min_current_stereo_confirmed": args.min_current_stereo_confirmed,
            "low_current_stereo_action": args.low_current_stereo_action,
            "carry_tracks_without_stereo": args.carry_tracks_without_stereo,
            "carry_track_max_depth_age": args.carry_track_max_depth_age,
            "pose_solver": args.pose_solver,
            "stereo3d_threshold": args.stereo3d_threshold,
            "stereo3d_iterations": args.stereo3d_iterations,
            "min_stereo3d_inliers": args.min_stereo3d_inliers,
            "pose_mode": args.pose_mode,
            "stereo_depth_mode": args.stereo_depth_mode,
            "sgbm_min_patch_valid_ratio": args.sgbm_min_patch_valid_ratio,
            "sgbm_patch_max_mad": args.sgbm_patch_max_mad,
            "sgbm_require_center_valid": args.sgbm_require_center_valid,
            "sgbm_center_max_diff": args.sgbm_center_max_diff,
            "depth_scale": args.depth_scale,
            "max_step_m": args.max_step_m,
            "max_speed_mps": args.max_speed_mps,
            "motion_gate_mode": args.motion_gate_mode,
            "max_rotation_rad": args.max_rotation_rad,
            "imu_rotation_gate_rad": args.imu_rotation_gate_rad,
            "orientation_source": args.orientation_source,
            "imu_attitude_extrinsic": args.imu_attitude_extrinsic,
            "imu_attitude_delta": args.imu_attitude_delta,
            "gyro_sign": args.gyro_sign,
            "imu_yaw_axis": args.imu_yaw_axis,
            "underwater_planar_motion_prior": args.underwater_planar_motion_prior,
            "planar_forward_gain": args.planar_forward_gain,
            "planar_lateral_gain": args.planar_lateral_gain,
            "planar_vertical_gain": args.planar_vertical_gain,
            "planar_max_step_m": args.planar_max_step_m,
            "planar_rotation_slowdown_rad": args.planar_rotation_slowdown_rad,
            "planar_min_turn_speed_scale": args.planar_min_turn_speed_scale,
            "translation_smoothing": args.translation_smoothing,
            "coast_on_failure": args.coast_on_failure,
            "coast_decay": args.coast_decay,
            "max_coast_frames": args.max_coast_frames,
            "imu_rotation_gate_enabled": not args.disable_imu_rotation_gate,
        },
    }


def write_debug_image(path: Path, gray: np.ndarray, tracks: dict[int, Track]) -> None:
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for track in tracks.values():
        x, y = track.left_point
        cv2.circle(image, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)
    cv2.imwrite(str(path), image)


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple((quaternion / norm).tolist())


if __name__ == "__main__":
    main()
