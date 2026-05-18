#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation" / "rc_control_model_path_30s.csv"
DEFAULT_REFERENCE = PROJECT_ROOT / "outputs" / "evaluation" / "rc_control_model_dvl_reference_30s.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "evaluation" / "rc_control_model_30s_summary.json"


@dataclass(frozen=True)
class RcSample:
    time_sec: float
    channels: np.ndarray


@dataclass(frozen=True)
class PositionSample:
    time_sec: float
    position_ros: np.ndarray


def main() -> None:
    args = parse_args()
    bag_data = read_bag_segment(args)
    rc_samples = bag_data["rc_samples"]
    ref_samples = bag_data["reference_samples"]
    if len(rc_samples) < 3 or len(ref_samples) < 3:
        raise SystemExit("Need at least three RC and reference samples.")

    reference_times = np.array([sample.time_sec for sample in ref_samples], dtype=np.float64)
    reference_ros = np.array([sample.position_ros for sample in ref_samples], dtype=np.float64)
    reference_ros -= reference_ros[0].copy()

    candidates = build_candidates(args, rc_samples, reference_times, reference_ros)

    if not candidates:
        raise SystemExit("No control model candidates generated.")
    for candidate in candidates:
        candidate["selection_score"] = selection_score(candidate, args.objective)
    candidates.sort(key=candidate_sort_key)
    best = candidates[0]

    write_path_csv(args.output_csv, reference_times, best["_prediction"], "rc_control_model_prediction", best["metrics"])
    write_path_csv(args.reference_csv, reference_times, reference_ros, "dvl_reference_for_control_model_scoring")

    public_candidates = []
    for candidate in candidates[: args.top_k]:
        item = {key: value for key, value in candidate.items() if not key.startswith("_")}
        public_candidates.append(item)

    summary = {
        "mode": "rc_control_model_optimization",
        "bag": str(args.bag),
        "image_topic": args.image_topic,
        "rc_topic": args.rc_topic,
        "reference_topic": args.reference_topic,
        "reference_usage": "offline_control_model_optimization_only",
        "vio_estimator_usage": "not_used_by_vio_estimator",
        "segment": {
            "start_sec": args.start_sec,
            "duration_sec": args.duration_sec,
            "image_origin_db_ns": bag_data["image_origin_db_ns"],
            "segment_start_db_ns": bag_data["segment_start_db_ns"],
            "segment_end_db_ns": bag_data["segment_end_db_ns"],
            "rc_samples": len(rc_samples),
            "reference_samples": len(ref_samples),
        },
        "output_csv": str(args.output_csv),
        "reference_csv": str(args.reference_csv),
        "best": {key: value for key, value in best.items() if not key.startswith("_")},
        "top_candidates": public_candidates,
        "interpretation": summarize_model(best["model"], args.active_channels),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a physically inspectable RC override -> motion model against the MP4-time DVL path. "
            "This does not feed DVL into VIO; it is for control-logic calibration and diagnosis."
        )
    )
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--image-topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--rc-topic", default="/mavros/rc/override")
    parser.add_argument("--reference-topic", default="/dvl/position")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=30.31)
    parser.add_argument("--model-type", choices=("linear", "body-yaw"), default="linear")
    parser.add_argument(
        "--objective",
        choices=("rmse", "xy-rmse", "balanced", "max-corr"),
        default="xy-rmse",
        help="Candidate ranking. balanced adds path-length and final-error penalties.",
    )
    parser.add_argument("--active-channels", default="3,4,5,6")
    parser.add_argument("--delay-grid", default="0,0.1,0.2,0.3,0.5,0.8,1.0,1.3,1.6,2.0")
    parser.add_argument("--deadzone-grid", default="0,0.02,0.04,0.06,0.08,0.10,0.14")
    parser.add_argument("--response-tau-grid", default="0,0.2,0.5,0.8,1.2,1.8,2.5")
    parser.add_argument("--ridge-grid", default="0,0.001,0.01,0.05,0.1")
    parser.add_argument("--include-drift", action="store_true")
    parser.add_argument("--yaw-channel", type=int, default=4)
    parser.add_argument("--vertical-channel", type=int, default=3)
    parser.add_argument(
        "--body-channel-pairs",
        default="5:6,6:5",
        help="forward:lateral RC channel candidates for body-yaw mode.",
    )
    parser.add_argument(
        "--yaw-rate-gain-grid",
        default="-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3",
        help="rad/s per normalized yaw command candidates for body-yaw mode.",
    )
    parser.add_argument(
        "--yaw0-grid-deg",
        default="-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180",
        help="initial heading candidates in degrees for body-yaw mode.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def build_candidates(
    args: argparse.Namespace,
    rc_samples: list[RcSample],
    reference_times: np.ndarray,
    reference_ros: np.ndarray,
) -> list[dict[str, Any]]:
    if args.model_type == "linear":
        return build_linear_candidates(args, rc_samples, reference_times, reference_ros)
    return build_body_yaw_candidates(args, rc_samples, reference_times, reference_ros)


def build_linear_candidates(
    args: argparse.Namespace,
    rc_samples: list[RcSample],
    reference_times: np.ndarray,
    reference_ros: np.ndarray,
) -> list[dict[str, Any]]:
    candidates = []
    for delay_sec in parse_grid(args.delay_grid):
        for deadzone in parse_grid(args.deadzone_grid):
            for response_tau in parse_grid(args.response_tau_grid):
                for ridge_lambda in parse_grid(args.ridge_grid):
                    prediction, model = fit_control_model(
                        rc_samples=rc_samples,
                        reference_times=reference_times,
                        reference_ros=reference_ros,
                        active_channels=args.active_channels,
                        delay_sec=delay_sec,
                        deadzone=deadzone,
                        response_tau=response_tau,
                        ridge_lambda=ridge_lambda,
                        include_drift=args.include_drift,
                    )
                    metrics = trajectory_metrics(prediction, reference_ros)
                    candidates.append(
                        {
                            "model_type": "linear",
                            "delay_sec": delay_sec,
                            "deadzone": deadzone,
                            "response_tau": response_tau,
                            "ridge_lambda": ridge_lambda,
                            "metrics": metrics,
                            "model": model,
                            "_prediction": prediction,
                        }
                    )
    return candidates


def build_body_yaw_candidates(
    args: argparse.Namespace,
    rc_samples: list[RcSample],
    reference_times: np.ndarray,
    reference_ros: np.ndarray,
) -> list[dict[str, Any]]:
    candidates = []
    for delay_sec in parse_grid(args.delay_grid):
        for deadzone in parse_grid(args.deadzone_grid):
            for response_tau in parse_grid(args.response_tau_grid):
                for ridge_lambda in parse_grid(args.ridge_grid):
                    for forward_channel, lateral_channel in parse_body_channel_pairs(args.body_channel_pairs):
                        for yaw_rate_gain in parse_grid(args.yaw_rate_gain_grid):
                            for yaw0_deg in parse_grid(args.yaw0_grid_deg):
                                prediction, model = fit_body_yaw_model(
                                    rc_samples=rc_samples,
                                    reference_times=reference_times,
                                    reference_ros=reference_ros,
                                    delay_sec=delay_sec,
                                    deadzone=deadzone,
                                    response_tau=response_tau,
                                    ridge_lambda=ridge_lambda,
                                    include_drift=args.include_drift,
                                    yaw_channel=args.yaw_channel,
                                    forward_channel=forward_channel,
                                    lateral_channel=lateral_channel,
                                    vertical_channel=args.vertical_channel,
                                    yaw_rate_gain=yaw_rate_gain,
                                    yaw0_rad=math.radians(yaw0_deg),
                                )
                                metrics = trajectory_metrics(prediction, reference_ros)
                                candidates.append(
                                    {
                                        "model_type": "body-yaw",
                                        "delay_sec": delay_sec,
                                        "deadzone": deadzone,
                                        "response_tau": response_tau,
                                        "ridge_lambda": ridge_lambda,
                                        "yaw_rate_gain": yaw_rate_gain,
                                        "yaw0_deg": yaw0_deg,
                                        "forward_channel": forward_channel,
                                        "lateral_channel": lateral_channel,
                                        "metrics": metrics,
                                        "model": model,
                                        "_prediction": prediction,
                                    }
                                )
    return candidates


def read_bag_segment(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from dvl_msgs.msg import DVLDR
        from mavros_msgs.msg import OverrideRCIn
        from rclpy.serialization import deserialize_message
    except ImportError as exc:
        raise SystemExit(
            "Run inside ROS2 env with mavros_msgs and dvl_msgs available. "
            "Example: set +u; source ~/miniconda3/bin/activate ros2_h311; "
            "source ros2_overlay_ws/install/setup.bash"
        ) from exc

    conn = sqlite3.connect(str(args.bag))
    try:
        topics = {name: topic_id for topic_id, name in conn.execute("select id, name from topics")}
        for topic in (args.image_topic, args.rc_topic, args.reference_topic):
            if topic not in topics:
                raise SystemExit(f"Missing topic in bag: {topic}")

        image_origin_db_ns = conn.execute(
            "select min(timestamp) from messages where topic_id = ?",
            (topics[args.image_topic],),
        ).fetchone()[0]
        if image_origin_db_ns is None:
            raise SystemExit(f"No image messages for topic: {args.image_topic}")

        segment_start_db_ns = int(image_origin_db_ns + round(args.start_sec * 1e9))
        segment_end_db_ns = int(segment_start_db_ns + round(args.duration_sec * 1e9))

        rc_samples: list[RcSample] = []
        for timestamp_ns, data in conn.execute(
            "select timestamp, data from messages where topic_id = ? and timestamp between ? and ? order by timestamp",
            (topics[args.rc_topic], segment_start_db_ns, segment_end_db_ns),
        ):
            msg = deserialize_message(data, OverrideRCIn)
            rc_samples.append(
                RcSample(
                    time_sec=(int(timestamp_ns) - segment_start_db_ns) / 1e9,
                    channels=np.array(msg.channels, dtype=np.float64),
                )
            )

        reference_samples: list[PositionSample] = []
        for timestamp_ns, data in conn.execute(
            "select timestamp, data from messages where topic_id = ? and timestamp between ? and ? order by timestamp",
            (topics[args.reference_topic], segment_start_db_ns, segment_end_db_ns),
        ):
            msg = deserialize_message(data, DVLDR)
            p = msg.position
            # DVL local frame is displayed in RViz as x, -y, z in this workspace.
            position_ros = np.array([float(p.x), -float(p.y), float(p.z)], dtype=np.float64)
            reference_samples.append(
                PositionSample(
                    time_sec=(int(timestamp_ns) - segment_start_db_ns) / 1e9,
                    position_ros=position_ros,
                )
            )

        return {
            "image_origin_db_ns": int(image_origin_db_ns),
            "segment_start_db_ns": segment_start_db_ns,
            "segment_end_db_ns": segment_end_db_ns,
            "rc_samples": rc_samples,
            "reference_samples": reference_samples,
        }
    finally:
        conn.close()


def fit_control_model(
    *,
    rc_samples: list[RcSample],
    reference_times: np.ndarray,
    reference_ros: np.ndarray,
    active_channels: str,
    delay_sec: float,
    deadzone: float,
    response_tau: float,
    ridge_lambda: float,
    include_drift: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    channels = parse_channels(active_channels)
    rc_times = np.array([sample.time_sec for sample in rc_samples], dtype=np.float64)
    rc_values = np.array([sample.channels for sample in rc_samples], dtype=np.float64)
    query_times = np.clip(reference_times - delay_sec, rc_times[0], rc_times[-1])
    normalized = []
    for channel in channels:
        pwm = np.interp(query_times, rc_times, rc_values[:, channel - 1])
        normalized.append(apply_deadzone((pwm - 1500.0) / 500.0, deadzone))
    u = np.column_stack(normalized)
    u = low_pass_controls(u, reference_times, response_tau)
    features = integrate_features(reference_times, u)
    if include_drift:
        features = np.column_stack([features, reference_times - reference_times[0]])

    prediction = np.zeros_like(reference_ros)
    coefficients: list[list[float]] = []
    for axis in range(3):
        coeff = ridge_lstsq(features, reference_ros[:, axis], ridge_lambda)
        prediction[:, axis] = features @ coeff
        coefficients.append([float(value) for value in coeff])

    return prediction, {
        "active_channels": channels,
        "delay_sec": float(delay_sec),
        "deadzone": float(deadzone),
        "response_tau": float(response_tau),
        "ridge_lambda": float(ridge_lambda),
        "include_drift": include_drift,
        "coefficient_units": "m_per_sec_per_normalized_pwm_after_time_integration",
        "axes": ["rviz_x_forward", "rviz_y_left", "rviz_z_up"],
        "coefficients_by_axis": {
            "x": coefficients[0],
            "y": coefficients[1],
            "z": coefficients[2],
        },
        "feature_names": [f"integral_ch{channel}" for channel in channels]
        + (["integral_bias_drift"] if include_drift else []),
    }


def fit_body_yaw_model(
    *,
    rc_samples: list[RcSample],
    reference_times: np.ndarray,
    reference_ros: np.ndarray,
    delay_sec: float,
    deadzone: float,
    response_tau: float,
    ridge_lambda: float,
    include_drift: bool,
    yaw_channel: int,
    forward_channel: int,
    lateral_channel: int,
    vertical_channel: int,
    yaw_rate_gain: float,
    yaw0_rad: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    rc_times = np.array([sample.time_sec for sample in rc_samples], dtype=np.float64)
    rc_values = np.array([sample.channels for sample in rc_samples], dtype=np.float64)
    query_times = np.clip(reference_times - delay_sec, rc_times[0], rc_times[-1])

    yaw_u = interpolate_normalized_channel(rc_times, rc_values, query_times, yaw_channel, deadzone)
    forward_u = interpolate_normalized_channel(rc_times, rc_values, query_times, forward_channel, deadzone)
    lateral_u = interpolate_normalized_channel(rc_times, rc_values, query_times, lateral_channel, deadzone)
    vertical_u = interpolate_normalized_channel(rc_times, rc_values, query_times, vertical_channel, deadzone)
    controls = low_pass_controls(
        np.column_stack([yaw_u, forward_u, lateral_u, vertical_u]),
        reference_times,
        response_tau,
    )
    yaw_u, forward_u, lateral_u, vertical_u = controls.T

    yaw_delta = integrate_features(reference_times, yaw_u[:, None])[:, 0] * yaw_rate_gain
    yaw = yaw0_rad + yaw_delta
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    forward_x = integrate_features(reference_times, (cos_yaw * forward_u)[:, None])[:, 0]
    forward_y = integrate_features(reference_times, (sin_yaw * forward_u)[:, None])[:, 0]
    lateral_x = integrate_features(reference_times, (-sin_yaw * lateral_u)[:, None])[:, 0]
    lateral_y = integrate_features(reference_times, (cos_yaw * lateral_u)[:, None])[:, 0]

    xy_feature_count = 4 if include_drift else 2
    xy_features = np.zeros((2 * len(reference_times), xy_feature_count), dtype=np.float64)
    xy_target = np.zeros(2 * len(reference_times), dtype=np.float64)
    xy_features[: len(reference_times), 0] = forward_x
    xy_features[: len(reference_times), 1] = lateral_x
    xy_features[len(reference_times) :, 0] = forward_y
    xy_features[len(reference_times) :, 1] = lateral_y
    elapsed = reference_times - reference_times[0]
    if include_drift:
        xy_features[: len(reference_times), 2] = elapsed
        xy_features[len(reference_times) :, 3] = elapsed
    xy_target[: len(reference_times)] = reference_ros[:, 0]
    xy_target[len(reference_times) :] = reference_ros[:, 1]
    xy_coefficients = ridge_lstsq(xy_features, xy_target, ridge_lambda)
    forward_speed_gain = xy_coefficients[0]
    lateral_speed_gain = xy_coefficients[1]
    drift_x_gain = float(xy_coefficients[2]) if include_drift else 0.0
    drift_y_gain = float(xy_coefficients[3]) if include_drift else 0.0

    z_feature = integrate_features(reference_times, vertical_u[:, None])[:, 0]
    if include_drift:
        z_features = np.column_stack([z_feature, elapsed])
        vertical_coefficients = ridge_lstsq(z_features, reference_ros[:, 2], ridge_lambda)
        vertical_speed_gain = float(vertical_coefficients[0])
        drift_z_gain = float(vertical_coefficients[1])
    elif float(np.std(z_feature)) < 1e-12:
        vertical_speed_gain = 0.0
        drift_z_gain = 0.0
    else:
        vertical_speed_gain = float(ridge_lstsq(z_feature[:, None], reference_ros[:, 2], ridge_lambda)[0])
        drift_z_gain = 0.0

    prediction = np.zeros_like(reference_ros)
    prediction[:, 0] = forward_speed_gain * forward_x + lateral_speed_gain * lateral_x + drift_x_gain * elapsed
    prediction[:, 1] = forward_speed_gain * forward_y + lateral_speed_gain * lateral_y + drift_y_gain * elapsed
    prediction[:, 2] = vertical_speed_gain * z_feature + drift_z_gain * elapsed

    return prediction, {
        "model_type": "body-yaw",
        "delay_sec": float(delay_sec),
        "deadzone": float(deadzone),
        "response_tau": float(response_tau),
        "ridge_lambda": float(ridge_lambda),
        "yaw_channel": yaw_channel,
        "forward_channel": forward_channel,
        "lateral_channel": lateral_channel,
        "vertical_channel": vertical_channel,
        "include_drift": include_drift,
        "yaw_rate_gain_rad_per_sec": float(yaw_rate_gain),
        "yaw0_deg": float(math.degrees(yaw0_rad)),
        "forward_speed_gain_mps": float(forward_speed_gain),
        "lateral_speed_gain_mps": float(lateral_speed_gain),
        "vertical_speed_gain_mps": float(vertical_speed_gain),
        "drift_velocity_mps": {
            "x": drift_x_gain,
            "y": drift_y_gain,
            "z": drift_z_gain,
        },
        "axes": ["rviz_x_forward", "rviz_y_left", "rviz_z_up"],
        "equations": {
            "yaw": "psi(t)=psi0+k_yaw*integral(u_yaw dt)",
            "world_velocity_xy": "[vx,vy]^T=Rz(psi)*[k_forward*u_forward,k_lateral*u_lateral]^T",
            "vertical": "vz=k_vertical*u_vertical",
        },
    }


def interpolate_normalized_channel(
    rc_times: np.ndarray,
    rc_values: np.ndarray,
    query_times: np.ndarray,
    channel: int,
    deadzone: float,
) -> np.ndarray:
    pwm = np.interp(query_times, rc_times, rc_values[:, channel - 1])
    return apply_deadzone((pwm - 1500.0) / 500.0, deadzone)


def apply_deadzone(values: np.ndarray, deadzone: float) -> np.ndarray:
    clipped = np.clip(values, -1.0, 1.0)
    if deadzone <= 0.0:
        return clipped
    magnitude = np.abs(clipped)
    active = magnitude > deadzone
    output = np.zeros_like(clipped)
    output[active] = np.sign(clipped[active]) * ((magnitude[active] - deadzone) / max(1e-9, 1.0 - deadzone))
    return output


def low_pass_controls(u: np.ndarray, times: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 1e-9 or len(u) < 2:
        return u
    filtered = np.zeros_like(u)
    filtered[0] = u[0]
    for index in range(1, len(u)):
        dt = max(0.0, float(times[index] - times[index - 1]))
        alpha = math.exp(-dt / tau)
        filtered[index] = alpha * filtered[index - 1] + (1.0 - alpha) * u[index]
    return filtered


def integrate_features(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    features = np.zeros_like(values)
    if len(times) < 2:
        return features
    dt = np.maximum(0.0, np.diff(times)).reshape(-1, 1)
    increments = 0.5 * (values[:-1] + values[1:]) * dt
    features[1:] = np.cumsum(increments, axis=0)
    return features


def ridge_lstsq(features: np.ndarray, target: np.ndarray, ridge_lambda: float) -> np.ndarray:
    if ridge_lambda <= 0.0:
        coeff, _, _, _ = np.linalg.lstsq(features, target, rcond=None)
        return coeff
    normal = features.T @ features
    penalty = ridge_lambda * np.eye(normal.shape[0], dtype=np.float64)
    rhs = features.T @ target
    return np.linalg.solve(normal + penalty, rhs)


def trajectory_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = prediction - reference
    errors = np.linalg.norm(delta, axis=1)
    xy_errors = np.linalg.norm(delta[:, :2], axis=1)
    corr_x = corrcoef(prediction[:, 0], reference[:, 0])
    corr_y = corrcoef(prediction[:, 1], reference[:, 1])
    corr_z = corrcoef(prediction[:, 2], reference[:, 2])
    return {
        "rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "xy_rmse_m": float(np.sqrt(np.mean(xy_errors * xy_errors))),
        "max_error_m": float(np.max(errors)),
        "mean_corr": float(np.mean([corr_x, corr_y, corr_z])),
        "xy_corr": float(np.mean([corr_x, corr_y])),
        "corr_x": corr_x,
        "corr_y": corr_y,
        "corr_z": corr_z,
        "prediction_path_length_m": path_length(prediction),
        "reference_path_length_m": path_length(reference),
        "final_error_m": float(np.linalg.norm(prediction[-1] - reference[-1])),
    }


def selection_score(candidate: dict[str, Any], objective: str) -> float:
    metrics = candidate["metrics"]
    if objective == "rmse":
        return float(metrics["rmse_m"])
    if objective == "max-corr":
        return -float(metrics["xy_corr"])
    if objective == "balanced":
        reference_length = max(float(metrics["reference_path_length_m"]), 1e-9)
        path_length_error = abs(float(metrics["prediction_path_length_m"]) - reference_length)
        final_error = float(metrics["final_error_m"])
        return float(metrics["xy_rmse_m"]) + 0.25 * path_length_error + 0.15 * final_error
    return float(metrics["xy_rmse_m"])


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = candidate["metrics"]
    return (
        float(candidate.get("selection_score", metrics["xy_rmse_m"])),
        float(metrics["xy_rmse_m"]),
        -float(metrics["xy_corr"]),
        float(metrics["final_error_m"]),
    )


def summarize_model(model: dict[str, Any], active_channels: str) -> dict[str, Any]:
    if model.get("model_type") == "body-yaw":
        return {
            "channel_mapping": {
                "yaw": model["yaw_channel"],
                "forward": model["forward_channel"],
                "lateral": model["lateral_channel"],
                "vertical": model["vertical_channel"],
            },
            "gains": {
                "yaw_rate_gain_rad_per_sec": model["yaw_rate_gain_rad_per_sec"],
                "yaw0_deg": model["yaw0_deg"],
                "forward_speed_gain_mps": model["forward_speed_gain_mps"],
                "lateral_speed_gain_mps": model["lateral_speed_gain_mps"],
                "vertical_speed_gain_mps": model["vertical_speed_gain_mps"],
                "drift_velocity_mps": model.get("drift_velocity_mps", {"x": 0.0, "y": 0.0, "z": 0.0}),
            },
            "physical_assumption": (
                "RC commands produce body-frame forward/lateral/vertical velocities; "
                "yaw command rotates body-frame XY velocity into RViz/world XY."
            ),
        }
    channels = parse_channels(active_channels)
    result: dict[str, Any] = {}
    for axis in ("x", "y", "z"):
        coeffs = model["coefficients_by_axis"][axis]
        pairs = []
        for channel, coeff in zip(channels, coeffs):
            pairs.append({"channel": channel, "gain": coeff})
        if model["include_drift"]:
            pairs.append({"channel": "drift", "gain": coeffs[-1]})
        pairs.sort(key=lambda item: abs(float(item["gain"])), reverse=True)
        result[f"dominant_{axis}"] = pairs[:3]
    return result


def write_path_csv(
    path: Path,
    times: np.ndarray,
    positions_ros: np.ndarray,
    source_note: str,
    metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                "reference_ros_x",
                "reference_ros_y",
                "reference_ros_z",
                "control_rmse_m",
                "control_mean_corr",
                "control_xy_corr",
                "control_rmse_match_ratio",
                "control_rmse_match_percent",
                "control_max_error_m",
            ]
        )
        reference_length = 0.0 if metrics is None else max(float(metrics["reference_path_length_m"]), 1e-9)
        rmse_m = 0.0 if metrics is None else float(metrics["rmse_m"])
        rmse_match_ratio = 1.0 if metrics is None else max(0.0, 1.0 - rmse_m / reference_length)
        for index, (time_sec, position_ros) in enumerate(zip(times, positions_ros)):
            x, y, z = ros_position_to_csv_storage(position_ros)
            writer.writerow(
                [
                    index,
                    f"{time_sec:.9f}",
                    f"{x:.9f}",
                    f"{y:.9f}",
                    f"{z:.9f}",
                    "0.000000000",
                    "0.000000000",
                    "0.000000000",
                    "1.000000000",
                    "1",
                    "0",
                    "0.000000",
                    "0",
                    "0",
                    "rc_control_model",
                    source_note,
                    f"{position_ros[0]:.9f}",
                    f"{position_ros[1]:.9f}",
                    f"{position_ros[2]:.9f}",
                    "" if metrics is None else f"{rmse_m:.9f}",
                    "" if metrics is None else f"{metrics['mean_corr']:.9f}",
                    "" if metrics is None else f"{metrics['xy_corr']:.9f}",
                    "" if metrics is None else f"{rmse_match_ratio:.9f}",
                    "" if metrics is None else f"{rmse_match_ratio * 100.0:.6f}",
                    "" if metrics is None else f"{metrics['max_error_m']:.9f}",
                ]
            )


def ros_position_to_csv_storage(position_ros: np.ndarray) -> tuple[float, float, float]:
    # Store so ros2_publish_odometry.py --coordinate-frame ros reconstructs the same ROS/RViz point.
    return (-float(position_ros[1]), -float(position_ros[2]), float(position_ros[0]))


def parse_channels(text: str) -> list[int]:
    channels = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        channel = int(token)
        if channel < 1 or channel > 18:
            raise SystemExit(f"Invalid RC channel: {channel}")
        channels.append(channel)
    if not channels:
        raise SystemExit("At least one active channel is required.")
    return channels


def parse_body_channel_pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise SystemExit(f"Invalid body channel pair '{token}'. Use forward:lateral, e.g. 5:6.")
        forward_text, lateral_text = token.split(":", 1)
        forward_channel = int(forward_text)
        lateral_channel = int(lateral_text)
        for channel in (forward_channel, lateral_channel):
            if channel < 1 or channel > 18:
                raise SystemExit(f"Invalid RC channel: {channel}")
        pairs.append((forward_channel, lateral_channel))
    if not pairs:
        raise SystemExit("At least one body channel pair is required.")
    return pairs


def parse_grid(text: str) -> list[float]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values or [0.0]


def corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


if __name__ == "__main__":
    main()
