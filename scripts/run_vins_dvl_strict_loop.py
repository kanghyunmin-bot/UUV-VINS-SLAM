#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)

    if not args.allow_dvl_derived_diagnostic:
        assert_not_dvl_derived(args.physical_vins_csv, "physical VINS")

    provenance = collect_provenance(args)
    iterations: list[dict[str, Any]] = []

    physical_summary = args.work_dir / "strict_loop_physical_overlay.json"
    physical = run_eval(
        vins_csv=args.physical_vins_csv,
        dvl_csv=args.dvl_csv,
        summary_json=physical_summary,
        zero_start=False,
        match_by=args.match_by,
        vins_coordinate_frame=args.vins_coordinate_frame,
        dvl_coordinate_frame=args.dvl_coordinate_frame,
        max_axis_rmse=args.max_axis_rmse,
        min_axis_corr=args.min_axis_corr,
    )
    physical_over_90 = physical["rmse_match_percent"] >= args.min_match_percent
    iterations.append(
        {
            "iteration": 1,
            "stage": "physical_vins_fusion_overlay",
            "action": "evaluate physical/sim3 VINS-Fusion path against fixed DVL reference",
            "passed_90_percent": physical_over_90,
            "passed_strict_axis_target": physical["passed"],
            "metrics": physical,
        }
    )

    selected_csv = args.physical_vins_csv
    strict_pass = bool(physical["passed"])
    anchor_generation: dict[str, Any] | None = None
    anchor_nozero: dict[str, Any] | None = None
    anchor_zero: dict[str, Any] | None = None

    if not strict_pass and args.allow_dvl_anchor_diagnostic:
        for loop_index in range(1, args.max_control_iterations + 1):
            anchor_summary = args.work_dir / f"strict_loop_anchor_generation_iter_{loop_index}.json"
            anchor_generation = run_anchor(
                vins_csv=args.physical_vins_csv,
                dvl_csv=args.dvl_csv,
                output_csv=args.anchor_output_csv,
                summary_json=anchor_summary,
                max_axis_rmse=args.max_axis_rmse,
                min_axis_corr=args.min_axis_corr,
            )
            anchor_nozero = run_eval(
                vins_csv=args.anchor_output_csv,
                dvl_csv=args.dvl_csv,
                summary_json=args.work_dir / f"strict_loop_anchor_overlay_nozero_iter_{loop_index}.json",
                zero_start=False,
                match_by=args.match_by,
                vins_coordinate_frame=args.vins_coordinate_frame,
                dvl_coordinate_frame=args.dvl_coordinate_frame,
                max_axis_rmse=args.max_axis_rmse,
                min_axis_corr=args.min_axis_corr,
            )
            anchor_zero = run_eval(
                vins_csv=args.anchor_output_csv,
                dvl_csv=args.dvl_csv,
                summary_json=args.work_dir / f"strict_loop_anchor_overlay_zero_iter_{loop_index}.json",
                zero_start=True,
                match_by=args.match_by,
                vins_coordinate_frame=args.vins_coordinate_frame,
                dvl_coordinate_frame=args.dvl_coordinate_frame,
                max_axis_rmse=args.max_axis_rmse,
                min_axis_corr=args.min_axis_corr,
            )
            strict_pass = bool(anchor_nozero["passed"] and anchor_zero["passed"])
            iterations.append(
                {
                    "iteration": loop_index + 1,
                    "stage": "dvl_anchor_control",
                    "action": (
                        "physical strict target failed; keep DVL reference fixed, anchor VINS output positions "
                        "to DVL reference, then re-evaluate no-zero-start and zero-start overlays"
                    ),
                    "passed_strict_axis_target": strict_pass,
                    "generation_metrics": anchor_generation,
                    "overlay_no_zero_start_metrics": anchor_nozero,
                    "overlay_zero_start_metrics": anchor_zero,
                }
            )
            selected_csv = args.anchor_output_csv
            if strict_pass:
                break
    elif not strict_pass:
        iterations.append(
            {
                "iteration": 2,
                "stage": "dvl_anchor_rejected",
                "action": (
                    "strict target failed, but DVL-anchor/control output is disabled; "
                    "DVL remains a separate fixed reference and is not injected into the VINS path"
                ),
                "passed_strict_axis_target": False,
                "metrics": {},
            }
        )

    if strict_pass:
        interpretation = (
            "The selected VINS path passed the strict overlay target while keeping DVL as a separate "
            "fixed reference."
        )
    elif args.allow_dvl_anchor_diagnostic and selected_csv == args.anchor_output_csv:
        interpretation = (
            "DVL-anchor diagnostic output was explicitly enabled and selected. This is not a pure "
            "VINS accuracy claim and must not be used as the VINS result."
        )
    else:
        interpretation = (
            "DVL was kept separate as a fixed reference. The current VINS path does not meet the "
            "strict 1e-5 RMSE / 0.9999 correlation target without injecting DVL into the VINS output."
        )

    report: dict[str, Any] = {
        "flow": [
            "1. Validate stereo MP4/exported stereo frames and rosbag IMU/DVL provenance.",
            "2. Keep DVL odometry/reference fixed as a separate metric source only.",
            "3. Evaluate VINS-Fusion path against DVL for >=90% match without injecting DVL into VINS.",
            "4. Evaluate strict per-axis overlay target: RMSE <= max_axis_rmse and corr >= min_axis_corr.",
            "5. If strict target fails, inspect VINS/frontend/initialization/extrinsic/time logic first.",
            "6. DVL-anchor output is forbidden by default and only runs with --allow-dvl-anchor-diagnostic.",
        ],
        "targets": {
            "min_match_percent": args.min_match_percent,
            "max_axis_rmse_m": args.max_axis_rmse,
            "min_axis_corr": args.min_axis_corr,
        },
        "provenance": provenance,
        "physical_vins_over_90_percent": physical_over_90,
        "strict_overlay_passed": strict_pass,
        "selected_overlay_csv": str(selected_csv),
        "dvl_reference_const_sha256": provenance["files"]["dvl_reference_csv"]["sha256"],
        "dvl_reference_modified_by_loop": False,
        "dvl_anchor_diagnostic_enabled": bool(args.allow_dvl_anchor_diagnostic),
        "interpretation": interpretation,
        "iterations": iterations,
    }
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv_report(args.report_csv, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VINS and DVL as separate paths against strict overlay targets.")
    parser.add_argument("--physical-vins-csv", type=Path, default=PROJECT_ROOT / "outputs/live/latest_odometry_vins_raw_best.csv")
    parser.add_argument("--dvl-csv", type=Path, default=PROJECT_ROOT / "outputs/live/latest_odometry_dvl_timefixed_const_reference.csv")
    parser.add_argument("--anchor-output-csv", type=Path, default=PROJECT_ROOT / "outputs/live/latest_odometry_vins_dvl_anchor_control.csv")
    parser.add_argument("--dataset-metadata", type=Path, default=PROJECT_ROOT / "outputs/vins_fusion/mp4_stereo_imu_rosbag_300/metadata.json")
    parser.add_argument("--left-mp4", type=Path, default=PROJECT_ROOT / "video_src/camera_camera_infra1_image_rect_raw.mp4")
    parser.add_argument("--right-mp4", type=Path, default=PROJECT_ROOT / "video_src/camera_camera_infra2_image_rect_raw.mp4")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "outputs/evaluation/strict_loop")
    parser.add_argument("--report-json", type=Path, default=PROJECT_ROOT / "outputs/evaluation/vins_dvl_strict_loop_report.json")
    parser.add_argument("--report-csv", type=Path, default=PROJECT_ROOT / "outputs/evaluation/vins_dvl_strict_loop_report.csv")
    parser.add_argument("--min-match-percent", type=float, default=90.0)
    parser.add_argument("--max-axis-rmse", type=float, default=1e-5)
    parser.add_argument("--min-axis-corr", type=float, default=0.9999)
    parser.add_argument("--match-by", choices=("time", "index"), default="time")
    parser.add_argument("--vins-coordinate-frame", choices=("ros", "raw", "opencv", "flip-y"), default="ros")
    parser.add_argument("--dvl-coordinate-frame", choices=("ros", "raw", "opencv", "flip-y"), default="flip-y")
    parser.add_argument("--max-control-iterations", type=int, default=3)
    parser.add_argument(
        "--allow-dvl-anchor-diagnostic",
        action="store_true",
        help="Explicitly allow generating a DVL-anchored diagnostic CSV. Disabled by default.",
    )
    parser.add_argument(
        "--allow-dvl-derived-diagnostic",
        action="store_true",
        help="Allow a DVL-fit/DVL-anchor CSV as a diagnostic VINS input. Disabled by default.",
    )
    return parser.parse_args()


def assert_not_dvl_derived(path: Path, role: str) -> None:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as file:
        header = next(csv.reader(file), [])
    marker_text = f"{path},{','.join(header)}".lower()
    forbidden_markers = (
        "dvl_anchor",
        "dvl_fit",
        "dvl_guided",
        "guided",
        "oracle",
        "reference_only",
    )
    if any(marker in marker_text for marker in forbidden_markers):
        raise SystemExit(f"Refusing {role} CSV because it is DVL-derived, not pure VINS: {path}")


def run_eval(
    *,
    vins_csv: Path,
    dvl_csv: Path,
    summary_json: Path,
    zero_start: bool,
    match_by: str,
    vins_coordinate_frame: str,
    dvl_coordinate_frame: str,
    max_axis_rmse: float,
    min_axis_corr: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_vins_dvl_overlay.py"),
        "--vins-csv",
        str(vins_csv),
        "--dvl-csv",
        str(dvl_csv),
        "--match-by",
        match_by,
        "--vins-coordinate-frame",
        vins_coordinate_frame,
        "--dvl-coordinate-frame",
        dvl_coordinate_frame,
        "--summary-json",
        str(summary_json),
        "--max-axis-rmse",
        str(max_axis_rmse),
        "--min-axis-corr",
        str(min_axis_corr),
    ]
    if zero_start:
        command.append("--zero-start")
    run(command)
    return read_json(summary_json)


def run_anchor(
    *,
    vins_csv: Path,
    dvl_csv: Path,
    output_csv: Path,
    summary_json: Path,
    max_axis_rmse: float,
    min_axis_corr: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/dvl_anchor_vins_control.py"),
        "--vins-csv",
        str(vins_csv),
        "--dvl-csv",
        str(dvl_csv),
        "--output-csv",
        str(output_csv),
        "--summary-json",
        str(summary_json),
        "--match-by",
        "index",
        "--max-axis-rmse",
        str(max_axis_rmse),
        "--min-axis-corr",
        str(min_axis_corr),
    ]
    run(command)
    return read_json(summary_json)


def collect_provenance(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(args.dataset_metadata)
    image0_count = count_pngs(args.dataset_metadata.parent / "image_0")
    image1_count = count_pngs(args.dataset_metadata.parent / "image_1")
    return {
        "dataset_metadata": str(args.dataset_metadata),
        "format": metadata.get("format"),
        "bag": metadata.get("bag"),
        "left_topic": metadata.get("left_topic"),
        "right_topic": metadata.get("right_topic"),
        "imu_source": metadata.get("imu_source"),
        "imu_topics": metadata.get("imu_topics"),
        "frames_metadata": metadata.get("frames"),
        "frames_image0_count": image0_count,
        "frames_image1_count": image1_count,
        "imu_samples": metadata.get("imu_samples"),
        "duration_sec": metadata.get("duration_sec"),
        "baseline_m": metadata.get("baseline_m"),
        "files": {
            "left_mp4": file_info(args.left_mp4),
            "right_mp4": file_info(args.right_mp4),
            "dvl_reference_csv": file_info(args.dvl_csv),
            "physical_vins_csv": file_info(args.physical_vins_csv),
        },
    }


def write_csv_report(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for item in report["iterations"]:
        metrics = item.get("metrics") or item.get("overlay_zero_start_metrics") or {}
        rows.append(
            {
                "iteration": item["iteration"],
                "stage": item["stage"],
                "passed_90_percent": item.get("passed_90_percent", ""),
                "passed_strict_axis_target": item.get("passed_strict_axis_target", ""),
                "rmse_match_percent": metrics.get("rmse_match_percent", ""),
                "rmse_3d_m": metrics.get("rmse_3d_m", ""),
                "rmse_x_m": (metrics.get("axis_rmse_m") or {}).get("x", ""),
                "rmse_y_m": (metrics.get("axis_rmse_m") or {}).get("y", ""),
                "rmse_z_m": (metrics.get("axis_rmse_m") or {}).get("z", ""),
                "corr_x": (metrics.get("axis_corr") or {}).get("x", ""),
                "corr_y": (metrics.get("axis_corr") or {}).get("y", ""),
                "corr_z": (metrics.get("axis_corr") or {}).get("z", ""),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_pngs(path: Path) -> int:
    return sum(1 for _ in path.glob("*.png")) if path.exists() else 0


if __name__ == "__main__":
    main()
