#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "outputs/ros2_stereo_imu_vio/gui_runs/run_20260511_204147_077/stereo_imu_rosbag"
DVL_REFERENCE = PROJECT_ROOT / "outputs/evaluation/dvl_xyz_30s_reference.csv"
LIVE_VINS = PROJECT_ROOT / "outputs/live/latest_odometry_vins_pure_stereo_imu.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/ros2_stereo_imu_vio/sweeps/underwater_background"
REPORT_JSON = PROJECT_ROOT / "outputs/evaluation/underwater_background_sweep_report.json"
REPORT_CSV = PROJECT_ROOT / "outputs/evaluation/underwater_background_sweep_report.csv"
BEST_CSV = PROJECT_ROOT / "outputs/evaluation/underwater_background_best_vins.csv"
BEST_POINT_CLOUD = PROJECT_ROOT / "outputs/evaluation/underwater_background_best_point_cloud.csv"


@dataclass(frozen=True)
class Candidate:
    name: str
    params: dict[str, Any]


def main() -> None:
    parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    baseline = evaluate_existing("live_current", LIVE_VINS)
    candidates = build_candidates()
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        result = run_candidate(index, len(candidates), candidate)
        results.append(result)
        if result.get("failed"):
            print(f"[{index:03d}/{len(candidates):03d}] {candidate.name} FAILED", flush=True)
            continue
        print(
            f"[{index:03d}/{len(candidates):03d}] {candidate.name} "
            f"match={result['match_percent']:.3f} dir={result['direction_error_deg']:.2f} "
            f"rmse={result['rmse_3d_m']:.3f} corr=({result['corr_x']:.3f},"
            f"{result['corr_y']:.3f},{result['corr_z']:.3f}) success={result['pose_success_count']}",
            flush=True,
        )

    valid = [row for row in results if not row.get("failed")]
    valid.sort(
        key=lambda row: (
            -float(row["match_percent"]),
            float(row["direction_error_deg"]),
            -min(float(row["corr_x"]), float(row["corr_y"]), float(row["corr_z"])),
        )
    )
    best = valid[0] if valid else None
    if best is not None:
        shutil.copyfile(best["output_csv"], BEST_CSV)
        shutil.copyfile(best["point_cloud_csv"], BEST_POINT_CLOUD)

    report = {
        "mode": "underwater_vio_background_sweep",
        "estimator_inputs": ["stereo_left", "stereo_right", "imu"],
        "dvl_usage": "offline_reference_only_not_estimator_input",
        "dataset_dir": str(DATASET_DIR),
        "dvl_reference_csv": str(DVL_REFERENCE),
        "live_baseline": baseline,
        "best": best,
        "best_outputs": {
            "csv": str(BEST_CSV) if best else "",
            "point_cloud_csv": str(BEST_POINT_CLOUD) if best else "",
        },
        "results": valid + [row for row in results if row.get("failed")],
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_csv(valid)
    print(json.dumps({"baseline": baseline, "best": best, "report_json": str(REPORT_JSON)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a background underwater stereo+IMU VIO candidate sweep. "
            "DVL is used only as an offline reference metric, never as an estimator input."
        )
    )
    return parser.parse_args()


def build_candidates() -> list[Candidate]:
    base: dict[str, Any] = {
        "feature_detector": "orb",
        "max_features": 480,
        "min_features": 120,
        "quality": 0.01,
        "min_distance": 16,
        "cell_max_features": 28,
        "orb_fast_threshold": 8,
        "fb_threshold": 0.8,
        "klt_max_error": 35.0,
        "temporal_ransac": "fundamental",
        "temporal_ransac_threshold": 0.6,
        "min_temporal_inliers": 18,
        "stereo_fb_threshold": 1.2,
        "epipolar_threshold": 2.0,
        "stereo_disparity_consistency": True,
        "stereo_disparity_mad_factor": 5.0,
        "pnp_threshold": 2.5,
        "min_pnp_inliers": 24,
        "pose_solver": "pnp",
        "pose_mode": "relative",
        "stereo_depth_mode": "klt",
        "depth_scale": 0.85,
        "max_step_m": 0.14,
        "max_speed_mps": 0.8,
        "translation_smoothing": 0.60,
    }

    candidates: list[Candidate] = []

    def add(name: str, **updates: Any) -> None:
        params = dict(base)
        params.update(updates)
        if "max_features" in updates and "min_features" not in updates:
            params["min_features"] = max(90, int(float(params["max_features"]) * 0.25))
        candidates.append(Candidate(name, params))

    add("base_cell28")
    for cell in (18, 28, 36):
        for max_features, min_distance in ((480, 16), (720, 12), (960, 10), (960, 8)):
            add(f"orb_c{cell}_f{max_features}_d{min_distance}", cell_max_features=cell, max_features=max_features, min_distance=min_distance)

    for depth_scale in (0.70, 0.78, 0.85, 0.95, 1.05, 1.15):
        add(f"depth_{depth_scale:.2f}".replace(".", "p"), depth_scale=depth_scale, max_features=720, min_distance=12)

    for focal_scale in (1.0, 1.15, 1.33):
        for baseline_scale in (0.75, 0.85, 1.0, 1.15):
            add(
                f"calib_f{focal_scale:.2f}_b{baseline_scale:.2f}".replace(".", "p"),
                focal_scale=focal_scale,
                baseline_scale=baseline_scale,
                max_features=720,
                min_distance=12,
            )

    for threshold in (235, 245, 250):
        add(f"bright_mask_{threshold}", mask_bright_threshold=threshold, mask_bright_dilate=11, max_features=720, min_distance=12)
    for ratio in (0.08, 0.12, 0.18):
        add(f"bottom_mask_{ratio:.2f}".replace(".", "p"), mask_bottom_ratio=ratio, max_features=720, min_distance=12)
    for clip in (1.5, 2.0, 3.0):
        add(f"clahe_{clip:.1f}".replace(".", "p"), clahe=True, clahe_clip_limit=clip, max_features=720, min_distance=12)
    for min_klt in (12, 24, 36):
        for clip in (1.5, 2.0):
            add(
                f"stereo_clahe_fb{min_klt}_c{clip:.1f}".replace(".", "p"),
                stereo_clahe_fallback_min_klt=min_klt,
                stereo_clahe_fallback_clip_limit=clip,
                max_features=720,
                min_distance=12,
            )
    for min_klt in (12, 24, 36):
        for min_score, margin in ((0.86, 0.06), (0.82, 0.05)):
            add(
                f"stereo_ncc_fb{min_klt}_s{min_score:.2f}_m{margin:.2f}".replace(".", "p"),
                stereo_ncc_fallback_min_klt=min_klt,
                stereo_ncc_min_score=min_score,
                stereo_ncc_min_margin=margin,
                max_features=720,
                min_distance=12,
            )

    for action in ("clamp", "reject"):
        for confirmed in (4, 8, 12):
            add(
                f"current_stereo_{action}_{confirmed}",
                min_current_stereo_confirmed=confirmed,
                low_current_stereo_action=action,
                max_features=720,
                min_distance=12,
            )
    for depth_age in (1, 2, 3):
        add(
            f"carry_stereo_gap_{depth_age}",
            carry_tracks_without_stereo=True,
            carry_track_max_depth_age=depth_age,
            max_features=720,
            min_distance=12,
        )

    for max_depth in (2.5, 3.5, 5.0, 8.0):
        add(f"max_depth_{max_depth:.1f}".replace(".", "p"), max_depth=max_depth, max_features=720, min_distance=12)
    for min_disp in (1.0, 1.5, 2.5, 4.0):
        add(f"min_disp_{min_disp:.1f}".replace(".", "p"), min_disparity=min_disp, max_features=720, min_distance=12)

    for smooth in (0.38, 0.50, 0.60, 0.72):
        add(f"smooth_{smooth:.2f}".replace(".", "p"), translation_smoothing=smooth, max_features=720, min_distance=12)

    for rel_change, abs_change in ((0.25, 0.35), (0.40, 0.50), (0.60, 0.75), (0.85, 1.00)):
        add(
            f"depth_stable_r{rel_change:.2f}_a{abs_change:.2f}".replace(".", "p"),
            track_depth_max_rel_change=rel_change,
            track_depth_max_abs_change=abs_change,
            max_features=720,
            min_distance=12,
        )

    for mode in ("hybrid", "sgbm"):
        add(
            f"depth_mode_{mode}",
            stereo_depth_mode=mode,
            sgbm_fallback_min_klt=24,
            sgbm_uniqueness=8,
            sgbm_sample_radius=2,
            max_features=720,
            min_distance=12,
        )
    for valid_ratio in (0.55, 0.75):
        for max_mad in (0.4, 0.8, 1.2):
            add(
                f"hybrid_patch_v{valid_ratio:.2f}_mad{max_mad:.1f}".replace(".", "p"),
                stereo_depth_mode="hybrid",
                sgbm_fallback_min_klt=36,
                sgbm_uniqueness=12,
                sgbm_sample_radius=2,
                sgbm_min_patch_valid_ratio=valid_ratio,
                sgbm_patch_max_mad=max_mad,
                sgbm_require_center_valid=True,
                sgbm_center_max_diff=1.0,
                max_features=720,
                min_distance=12,
            )

    unique: dict[str, Candidate] = {}
    for candidate in candidates:
        key = json.dumps(candidate.params, sort_keys=True)
        unique.setdefault(key, candidate)
    return list(unique.values())


def run_candidate(index: int, count: int, candidate: Candidate) -> dict[str, Any]:
    stem = f"{index:03d}_{candidate.name}"
    output_csv = OUTPUT_DIR / f"{stem}.csv"
    point_cloud_csv = OUTPUT_DIR / f"{stem}_point_cloud.csv"
    summary_json = OUTPUT_DIR / f"{stem}.json"
    debug_csv = OUTPUT_DIR / f"{stem}_debug.csv"
    eval_json = OUTPUT_DIR / f"{stem}_eval.json"

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/ros2_stereo_imu_vio.py"),
        "--dataset-dir",
        str(DATASET_DIR),
        "--output-csv",
        str(output_csv),
        "--point-cloud-csv",
        str(point_cloud_csv),
        "--summary-json",
        str(summary_json),
        "--debug-csv",
        str(debug_csv),
    ]
    append_params(command, candidate.params)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        return {
            "name": candidate.name,
            "params": candidate.params,
            "failed": True,
            "stage": "vio",
            "error": "\n".join(completed.stdout.splitlines()[-40:]),
        }

    eval_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_vins_dvl_overlay.py"),
        "--vins-csv",
        str(output_csv),
        "--dvl-csv",
        str(DVL_REFERENCE),
        "--summary-json",
        str(eval_json),
        "--vins-coordinate-frame",
        "ros",
        "--dvl-coordinate-frame",
        "flip-y",
        "--zero-start",
    ]
    completed = subprocess.run(eval_command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        return {
            "name": candidate.name,
            "params": candidate.params,
            "failed": True,
            "stage": "evaluate",
            "error": "\n".join(completed.stdout.splitlines()[-40:]),
        }

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    metrics = json.loads(eval_json.read_text(encoding="utf-8"))
    return {
        "name": candidate.name,
        "params": candidate.params,
        "failed": False,
        "output_csv": str(output_csv),
        "point_cloud_csv": str(point_cloud_csv),
        "summary_json": str(summary_json),
        "debug_csv": str(debug_csv),
        "eval_json": str(eval_json),
        "match_percent": float(metrics["rmse_match_percent"]),
        "rmse_3d_m": float(metrics["rmse_3d_m"]),
        "direction_error_deg": float(metrics["mean_direction_error_deg"]),
        "corr_x": float(metrics["axis_corr"]["x"]),
        "corr_y": float(metrics["axis_corr"]["y"]),
        "corr_z": float(metrics["axis_corr"]["z"]),
        "axis_rmse_m": metrics["axis_rmse_m"],
        "pose_success_count": int(summary.get("pose_success_count", 0)),
        "path_length_m": float(summary["trajectory_stats"]["path_length_m"]),
        "note_counts": summary.get("note_counts", {}),
        "candidate_index": index,
        "candidate_count": count,
    }


def append_params(command: list[str], params: dict[str, Any]) -> None:
    for key, value in params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        command.extend([flag, str(value)])


def evaluate_existing(name: str, csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {"name": name, "missing": True}
    eval_json = OUTPUT_DIR / f"{name}_eval.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_vins_dvl_overlay.py"),
        "--vins-csv",
        str(csv_path),
        "--dvl-csv",
        str(DVL_REFERENCE),
        "--summary-json",
        str(eval_json),
        "--vins-coordinate-frame",
        "ros",
        "--dvl-coordinate-frame",
        "flip-y",
        "--zero-start",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        return {"name": name, "failed": True, "error": completed.stdout}
    metrics = json.loads(eval_json.read_text(encoding="utf-8"))
    return {
        "name": name,
        "csv": str(csv_path),
        "match_percent": float(metrics["rmse_match_percent"]),
        "rmse_3d_m": float(metrics["rmse_3d_m"]),
        "direction_error_deg": float(metrics["mean_direction_error_deg"]),
        "corr_x": float(metrics["axis_corr"]["x"]),
        "corr_y": float(metrics["axis_corr"]["y"]),
        "corr_z": float(metrics["axis_corr"]["z"]),
    }


def write_report_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "match_percent",
        "rmse_3d_m",
        "direction_error_deg",
        "corr_x",
        "corr_y",
        "corr_z",
        "path_length_m",
        "pose_success_count",
        "output_csv",
        "point_cloud_csv",
        "params",
    ]
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fieldnames},
                    "params": json.dumps(row.get("params", {}), sort_keys=True),
                }
            )


if __name__ == "__main__":
    main()
