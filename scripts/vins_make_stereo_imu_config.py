#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "vins_fusion" / "templates"
BODY_T_CAM0 = (
    (-5.7586305857286746e-03, -4.0463318787729019e-03, 9.9997523237933461e-01, 2.0329267950355900e-02),
    (-9.9998287214160420e-01, -1.0224590553211677e-03, -5.7628118925283633e-03, 7.9325209639615653e-03),
    (1.0457519809151661e-03, -9.9999129084997906e-01, -4.0403746097850135e-03, 2.8559824645148020e-03),
    (0.0, 0.0, 0.0, 1.0),
)


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    width = int(metadata["width"])
    height = int(metadata["height"])

    cam0 = metadata["camera0"]
    cam1 = metadata["camera1"]
    k0 = cam0["k"]
    k1 = cam1["k"]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.config_dir
    config_dir.mkdir(parents=True, exist_ok=True)

    cam0_path = config_dir / "underwater_cam0_pinhole.yaml"
    cam1_path = config_dir / "underwater_cam1_pinhole.yaml"
    config_path = config_dir / "underwater_stereo_imu_config.yaml"

    cam_template = (TEMPLATE_DIR / "cam_pinhole.yaml.tpl").read_text(encoding="utf-8")
    cam0_path.write_text(
        replace_all(
            cam_template,
            {
                "__CAMERA_NAME__": "underwater_cam0",
                "__WIDTH__": str(width),
                "__HEIGHT__": str(height),
                "__FX__": f"{float(k0[0]):.9f}",
                "__FY__": f"{float(k0[4]):.9f}",
                "__CX__": f"{float(k0[2]):.9f}",
                "__CY__": f"{float(k0[5]):.9f}",
            },
        ),
        encoding="utf-8",
    )
    cam1_path.write_text(
        replace_all(
            cam_template,
            {
                "__CAMERA_NAME__": "underwater_cam1",
                "__WIDTH__": str(width),
                "__HEIGHT__": str(height),
                "__FX__": f"{float(k1[0]):.9f}",
                "__FY__": f"{float(k1[4]):.9f}",
                "__CX__": f"{float(k1[2]):.9f}",
                "__CY__": f"{float(k1[5]):.9f}",
            },
        ),
        encoding="utf-8",
    )

    config_template = (TEMPLATE_DIR / "stereo_imu_config.yaml.tpl").read_text(encoding="utf-8")
    body_t_cam0 = BODY_T_CAM0
    if args.extrinsic_mode == "rs_identity_plus":
        body_t_cam0 = identity_camera_transform(args.imu_cam_translation)
        body_t_cam1 = identity_right_camera_transform(args.imu_cam_translation, float(metadata.get("baseline_m") or 0.0500365))
    elif args.extrinsic_mode == "rs_identity_minus":
        body_t_cam0 = identity_camera_transform(args.imu_cam_translation)
        body_t_cam1 = identity_right_camera_transform(args.imu_cam_translation, -float(metadata.get("baseline_m") or 0.0500365))
    body_t_cam1 = (
        rectified_right_camera_transform(float(metadata.get("baseline_m") or 0.0500365))
        if args.extrinsic_mode == "hardcoded"
        else body_t_cam1
    )
    replacements = {
        "__OUTPUT_PATH__": str(output_dir.resolve()) + "/",
        "__CAM0_CALIB__": cam0_path.name,
        "__CAM1_CALIB__": cam1_path.name,
        "__WIDTH__": str(width),
        "__HEIGHT__": str(height),
        "__ESTIMATE_EXTRINSIC__": str(args.estimate_extrinsic),
        "__BODY_T_CAM0_DATA__": matrix_to_opencv_data(body_t_cam0),
        "__BODY_T_CAM1_DATA__": matrix_to_opencv_data(body_t_cam1),
        "__MAX_CNT__": str(args.max_cnt),
        "__MIN_DIST__": str(args.min_dist),
        "__FREQ__": str(args.freq),
        "__F_THRESHOLD__": f"{args.f_threshold:.6f}",
        "__MAX_SOLVER_TIME__": f"{args.max_solver_time:.6f}",
        "__MAX_NUM_ITERATIONS__": str(args.max_num_iterations),
        "__KEYFRAME_PARALLAX__": f"{args.keyframe_parallax:.6f}",
        "__ACC_N__": f"{args.acc_n:.9f}",
        "__GYR_N__": f"{args.gyr_n:.9f}",
        "__ACC_W__": f"{args.acc_w:.9f}",
        "__GYR_W__": f"{args.gyr_w:.9f}",
        "__G_NORM__": f"{args.g_norm:.9f}",
        "__ESTIMATE_TD__": str(args.estimate_td),
        "__TD__": f"{args.td:.9f}",
    }
    config_path.write_text(replace_all(config_template, replacements), encoding="utf-8")
    print(config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a VINS-Fusion stereo+IMU config from exported ROS2 bag metadata.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "underwater_stereo_imu_rosbag" / "metadata.json",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "config_stereo_imu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vins_fusion" / "vins_output_stereo_imu",
    )
    parser.add_argument("--estimate-extrinsic", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument(
        "--extrinsic-mode",
        choices=["hardcoded", "rs_identity_plus", "rs_identity_minus"],
        default="hardcoded",
        help=(
            "hardcoded: legacy D435i-like Kalibr matrix. "
            "rs_identity_plus/minus: use RealSense bag extrinsics where infra/depth/gyro rotations are identity; "
            "right camera is shifted by +/- baseline along rectified camera x."
        ),
    )
    parser.add_argument(
        "--imu-cam-translation",
        default="-0.00552,0.00510,0.01174",
        help="Translation for body_T_cam0 in rs_identity_* modes, parsed as tx,ty,tz meters.",
    )
    parser.add_argument("--estimate-td", type=int, choices=[0, 1], default=0)
    parser.add_argument("--td", type=float, default=0.0)
    parser.add_argument("--max-cnt", type=int, default=150)
    parser.add_argument("--min-dist", type=int, default=30)
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--f-threshold", type=float, default=1.0)
    parser.add_argument("--max-solver-time", type=float, default=0.04)
    parser.add_argument("--max-num-iterations", type=int, default=8)
    parser.add_argument("--keyframe-parallax", type=float, default=10.0)
    parser.add_argument("--acc-n", type=float, default=0.1)
    parser.add_argument("--gyr-n", type=float, default=0.01)
    parser.add_argument("--acc-w", type=float, default=0.001)
    parser.add_argument("--gyr-w", type=float, default=0.0001)
    parser.add_argument("--g-norm", type=float, default=9.805)
    return parser.parse_args()


def rectified_right_camera_transform(baseline_m: float) -> tuple[tuple[float, float, float, float], ...]:
    # The bag uses rectified stereo images. CameraInfo P[3] encodes a virtual
    # right camera shifted along cam0 optical x, so cam1 keeps cam0 rotation.
    rows = [list(row) for row in BODY_T_CAM0]
    translation = [
        BODY_T_CAM0[0][3] + BODY_T_CAM0[0][0] * baseline_m,
        BODY_T_CAM0[1][3] + BODY_T_CAM0[1][0] * baseline_m,
        BODY_T_CAM0[2][3] + BODY_T_CAM0[2][0] * baseline_m,
    ]
    rows[0][3], rows[1][3], rows[2][3] = translation
    return tuple(tuple(row) for row in rows)


def identity_camera_transform(translation_text: str) -> tuple[tuple[float, float, float, float], ...]:
    translation = parse_translation(translation_text)
    return (
        (1.0, 0.0, 0.0, translation[0]),
        (0.0, 1.0, 0.0, translation[1]),
        (0.0, 0.0, 1.0, translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def identity_right_camera_transform(
    translation_text: str,
    baseline_m: float,
) -> tuple[tuple[float, float, float, float], ...]:
    translation = parse_translation(translation_text)
    return (
        (1.0, 0.0, 0.0, translation[0] + baseline_m),
        (0.0, 1.0, 0.0, translation[1]),
        (0.0, 0.0, 1.0, translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def parse_translation(text: str) -> tuple[float, float, float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != 3:
        raise SystemExit("--imu-cam-translation must contain exactly three comma-separated values.")
    return (values[0], values[1], values[2])


def matrix_to_opencv_data(matrix: tuple[tuple[float, float, float, float], ...]) -> str:
    return ", ".join(f"{value:.16g}" for row in matrix for value in row)


def replace_all(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    main()
