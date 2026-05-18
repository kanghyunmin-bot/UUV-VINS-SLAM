#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vio_frontend import PipelineConfig, VioFrontendPipeline, load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the visual frontend used before a VIO-SLAM backend estimator."
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input MP4 file")
    parser.add_argument("--config", type=Path, help="JSON config file")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-features", type=int)
    parser.add_argument(
        "--feature-method",
        choices=["shi_tomasi_klt", "orb_match", "sift_match", "akaze_match"],
    )
    parser.add_argument("--min-features", type=int)
    parser.add_argument("--quality", type=float)
    parser.add_argument("--min-distance", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--fb-threshold", type=float)
    parser.add_argument("--ransac-threshold", type=float)
    parser.add_argument("--ransac-confidence", type=float)
    parser.add_argument("--geometry-model", choices=["auto", "fundamental", "essential", "none"])
    parser.add_argument("--no-odometry", action="store_true")
    parser.add_argument("--assumed-focal-scale", type=float)
    parser.add_argument("--motion-scale", type=float)
    parser.add_argument("--pose-min-inliers", type=int)
    parser.add_argument("--pose-ransac-threshold", type=float)
    parser.add_argument("--detect-interval", type=int)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--resize-width", type=int)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--draw-rejected", action="store_true")
    return parser.parse_args()


def apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    if args.input is not None:
        config.dataset.input_path = args.input
    if args.output_dir is not None:
        config.output.output_dir = args.output_dir
    if args.max_features is not None:
        config.features.max_features = args.max_features
    if args.feature_method is not None:
        config.features.method = args.feature_method
    if args.min_features is not None:
        config.features.min_features = args.min_features
    if args.quality is not None:
        config.features.quality = args.quality
    if args.min_distance is not None:
        config.features.min_distance = args.min_distance
    if args.block_size is not None:
        config.features.block_size = args.block_size
    if args.fb_threshold is not None:
        config.optical_flow.forward_backward_threshold = args.fb_threshold
    if args.ransac_threshold is not None:
        config.geometry.ransac_threshold = args.ransac_threshold
    if args.ransac_confidence is not None:
        config.geometry.ransac_confidence = args.ransac_confidence
    if args.geometry_model is not None:
        config.geometry.model = args.geometry_model
    if args.no_odometry:
        config.odometry.enabled = False
    if args.assumed_focal_scale is not None:
        config.odometry.assumed_focal_scale = args.assumed_focal_scale
    if args.motion_scale is not None:
        config.odometry.motion_scale = args.motion_scale
    if args.pose_min_inliers is not None:
        config.odometry.min_pose_inliers = args.pose_min_inliers
    if args.pose_ransac_threshold is not None:
        config.odometry.pose_ransac_threshold = args.pose_ransac_threshold
    if args.detect_interval is not None:
        config.features.detect_interval = args.detect_interval
    if args.start_frame is not None:
        config.dataset.start_frame = args.start_frame
    if args.max_frames is not None:
        config.dataset.max_frames = args.max_frames
    if args.resize_width is not None:
        config.preprocess.resize_width = args.resize_width
    if args.clahe is not None:
        config.preprocess.clahe = args.clahe
    if args.no_video:
        config.output.write_video = False
    if args.draw_rejected:
        config.output.draw_rejected = True
    return config


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config) if args.config else PipelineConfig()
    config = apply_overrides(config, args)
    summary = VioFrontendPipeline(config).run()
    print(json.dumps({k: v for k, v in summary.items() if k != "frame_stats"}, indent=2))


if __name__ == "__main__":
    main()
