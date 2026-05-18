#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "vins_fusion" / "templates"


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    width = int(args.image_width or metadata["width"])
    height = int(args.image_height or metadata["height"])
    fx = float(args.fx or max(width, height) * args.focal_scale)
    fy = float(args.fy or fx)
    cx = float(args.cx if args.cx is not None else width * 0.5)
    cy = float(args.cy if args.cy is not None else height * 0.5)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.config_dir
    config_dir.mkdir(parents=True, exist_ok=True)

    cam0_path = config_dir / "underwater_cam0_pinhole.yaml"
    cam1_path = config_dir / "underwater_cam1_pinhole.yaml"
    config_path = config_dir / "underwater_stereo_config.yaml"

    cam_template = (TEMPLATE_DIR / "cam_pinhole.yaml.tpl").read_text(encoding="utf-8")
    cam_values = {
        "__WIDTH__": str(width),
        "__HEIGHT__": str(height),
        "__FX__": f"{fx:.9f}",
        "__FY__": f"{fy:.9f}",
        "__CX__": f"{cx:.9f}",
        "__CY__": f"{cy:.9f}",
    }
    cam0_path.write_text(replace_all(cam_template, {**cam_values, "__CAMERA_NAME__": "underwater_cam0"}), encoding="utf-8")
    cam1_path.write_text(replace_all(cam_template, {**cam_values, "__CAMERA_NAME__": "underwater_cam1"}), encoding="utf-8")

    config_template = (TEMPLATE_DIR / "stereo_config.yaml.tpl").read_text(encoding="utf-8")
    replacements = {
        "__OUTPUT_PATH__": str(output_dir.resolve()) + "/",
        "__CAM0_CALIB__": cam0_path.name,
        "__CAM1_CALIB__": cam1_path.name,
        "__WIDTH__": str(width),
        "__HEIGHT__": str(height),
        "__BASELINE__": f"{args.baseline:.9f}",
        "__MAX_CNT__": str(args.max_cnt),
        "__MIN_DIST__": str(args.min_dist),
        "__FREQ__": str(args.freq),
        "__F_THRESHOLD__": f"{args.f_threshold:.6f}",
        "__MAX_SOLVER_TIME__": f"{args.max_solver_time:.6f}",
        "__MAX_NUM_ITERATIONS__": str(args.max_num_iterations),
        "__KEYFRAME_PARALLAX__": f"{args.keyframe_parallax:.6f}",
    }
    config_path.write_text(replace_all(config_template, replacements), encoding="utf-8")
    print(config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a VINS-Fusion stereo-only config for exported MP4 data.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "underwater_stereo_kitti" / "metadata.json",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "vins_output",
    )
    parser.add_argument("--image-width", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=0)
    parser.add_argument("--focal-scale", type=float, default=0.62)
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--baseline", type=float, default=0.05)
    parser.add_argument("--max-cnt", type=int, default=150)
    parser.add_argument("--min-dist", type=int, default=30)
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--f-threshold", type=float, default=1.0)
    parser.add_argument("--max-solver-time", type=float, default=0.04)
    parser.add_argument("--max-num-iterations", type=int, default=8)
    parser.add_argument("--keyframe-parallax", type=float, default=10.0)
    return parser.parse_args()


def replace_all(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    main()
