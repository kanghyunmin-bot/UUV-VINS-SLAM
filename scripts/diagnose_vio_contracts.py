#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


@dataclass(frozen=True)
class PoseSample:
    t: float
    q: np.ndarray


@dataclass(frozen=True)
class ImuSample:
    t: float
    gyro: np.ndarray


@dataclass(frozen=True)
class StereoStamp:
    left_ns: int
    right_ns: int


def stamp_to_ns(stamp: object) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def q_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


def q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def q_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def R_angle(R: np.ndarray) -> float:
    c = (float(np.trace(R)) - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def R_log(R: np.ndarray) -> np.ndarray:
    angle = R_angle(R)
    if angle < 1e-9:
        return np.zeros(3, dtype=float)
    skew = (R - R.T) / (2.0 * math.sin(angle))
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=float)
    return axis * angle


def read_csv_poses(path: Path) -> list[PoseSample]:
    out: list[PoseSample] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(
                PoseSample(
                    t=float(row["timestamp_sec"]),
                    q=q_normalize(
                        np.array(
                            [
                                float(row["qx"]),
                                float(row["qy"]),
                                float(row["qz"]),
                                float(row["qw"]),
                            ],
                            dtype=float,
                        )
                    ),
                )
            )
    return out


def read_csv_imu(path: Path) -> list[ImuSample]:
    out: list[ImuSample] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(
                ImuSample(
                    t=float(row["timestamp_sec"]),
                    gyro=np.array([float(row["gx"]), float(row["gy"]), float(row["gz"])], dtype=float),
                )
            )
    return out


def read_stereo_schedule(path: Path) -> list[StereoStamp]:
    out: list[StereoStamp] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(StereoStamp(left_ns=int(row["left_ns"]), right_ns=int(row["right_ns"])))
    return out


def interp_pose(poses: list[PoseSample], t: float) -> PoseSample | None:
    if not poses or t < poses[0].t or t > poses[-1].t:
        return None
    lo, hi = 0, len(poses) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if poses[mid].t <= t:
            lo = mid
        else:
            hi = mid
    a, b = poses[lo], poses[hi]
    if abs(b.t - a.t) < 1e-9:
        return a
    u = (t - a.t) / (b.t - a.t)
    q0 = a.q
    q1 = b.q
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    dot = max(-1.0, min(1.0, float(np.dot(q0, q1))))
    if dot > 0.9995:
        q = q_normalize((1.0 - u) * q0 + u * q1)
    else:
        theta = math.acos(dot)
        q = (math.sin((1.0 - u) * theta) * q0 + math.sin(u * theta) * q1) / math.sin(theta)
    return PoseSample(t=t, q=q_normalize(q))


def corrcoef(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or len(b) < 3:
        return None
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize_vector_match(target: np.ndarray, measured: np.ndarray) -> dict[str, object]:
    err = measured - target
    return {
        "count": int(len(target)),
        "axis_corr": [corrcoef(target[:, i], measured[:, i]) for i in range(3)],
        "axis_rmse": [float(np.sqrt(np.mean(err[:, i] ** 2))) for i in range(3)],
        "vector_rmse": float(np.sqrt(np.mean(np.sum(err * err, axis=1)))),
        "mean_target_norm": float(np.mean(np.linalg.norm(target, axis=1))),
        "mean_measured_norm": float(np.mean(np.linalg.norm(measured, axis=1))),
    }


def signed_permutation_matrices() -> list[np.ndarray]:
    mats: list[np.ndarray] = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    m = np.zeros((3, 3), dtype=float)
                    signs = [sx, sy, sz]
                    for out_axis, in_axis in enumerate(perm):
                        m[out_axis, in_axis] = signs[out_axis]
                    mats.append(m)
    return mats


def diagnose_imu_axes(poses: list[PoseSample], imu: list[ImuSample]) -> dict[str, object]:
    if len(poses) < 3 or len(imu) < 3:
        return {"error": "not enough pose or imu samples"}
    imu_times = np.array([s.t for s in imu], dtype=float)
    targets = []
    measured = []
    imu_index0 = 0
    for a, b in zip(poses[:-1], poses[1:]):
        dt = b.t - a.t
        if dt <= 1e-4:
            continue
        R_a = q_to_R(a.q)
        R_b = q_to_R(b.q)
        body_rel = R_a.T @ R_b
        odom_body_gyro = R_log(body_rel) / dt
        lo = int(np.searchsorted(imu_times, a.t, side="left"))
        hi = int(np.searchsorted(imu_times, b.t, side="right"))
        if hi <= lo:
            continue
        gyro = np.mean([s.gyro for s in imu[lo:hi]], axis=0)
        if np.linalg.norm(odom_body_gyro) < 1e-4 and np.linalg.norm(gyro) < 1e-4:
            continue
        targets.append(odom_body_gyro)
        measured.append(gyro)
        imu_index0 = hi
    if not targets:
        return {"error": "no overlapping motion intervals"}
    target = np.vstack(targets)
    meas = np.vstack(measured)
    identity = summarize_vector_match(target, meas)
    best = None
    for m in signed_permutation_matrices():
        mapped = (m @ meas.T).T
        summary = summarize_vector_match(target, mapped)
        score = summary["vector_rmse"]
        if best is None or score < best["summary"]["vector_rmse"]:
            best = {
                "matrix_odom_body_from_imu": m.tolist(),
                "determinant": float(np.linalg.det(m)),
                "summary": summary,
            }
    return {"identity": identity, "best_signed_permutation": best, "used_intervals": int(len(target)), "last_imu_index": imu_index0}


def read_images_for_stamps(
    bag_db: Path,
    topic: str,
    stamps: set[int],
    time_source: str,
) -> dict[int, Image]:
    conn = sqlite3.connect(str(bag_db))
    try:
        topic_ids = {name: tid for tid, name in conn.execute("select id, name from topics")}
        if topic not in topic_ids:
            raise RuntimeError(f"missing image topic: {topic}")
        topic_id = topic_ids[topic]
        images: dict[int, Image] = {}
        for db_ns, data in conn.execute(
            "select timestamp, data from messages where topic_id = ? order by timestamp",
            (topic_id,),
        ):
            msg = deserialize_message(data, Image)
            stamp_ns = int(db_ns) if time_source == "db" else stamp_to_ns(msg.header.stamp)
            if stamp_ns in stamps:
                images[stamp_ns] = msg
                if len(images) == len(stamps):
                    break
        return images
    finally:
        conn.close()


def image_to_gray(msg: Image) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in {"mono8", "8UC1"}:
        return data.reshape((msg.height, msg.width))
    if msg.encoding in {"rgb8", "bgr8"}:
        img = data.reshape((msg.height, msg.width, 3))
        code = cv2.COLOR_RGB2GRAY if msg.encoding == "rgb8" else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(img, code)
    raise RuntimeError(f"unsupported image encoding: {msg.encoding}")


def read_camera_matrix(path: Path) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"failed to open camera yaml: {path}")
    fx = fs.getNode("projection_parameters").getNode("fx").real()
    fy = fs.getNode("projection_parameters").getNode("fy").real()
    cx = fs.getNode("projection_parameters").getNode("cx").real()
    cy = fs.getNode("projection_parameters").getNode("cy").real()
    fs.release()
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def read_body_T_cam(path: Path, key: str = "body_T_cam0") -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"failed to open config yaml: {path}")
    mat = fs.getNode(key).mat()
    fs.release()
    if mat is None or mat.shape != (4, 4):
        raise RuntimeError(f"missing {key} in {path}")
    return mat.astype(float)


def track_pair(prev: np.ndarray, curr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    prev_eq = clahe.apply(prev)
    curr_eq = clahe.apply(curr)
    pts0 = cv2.goodFeaturesToTrack(prev_eq, maxCorners=700, qualityLevel=0.01, minDistance=10, blockSize=7)
    if pts0 is None or len(pts0) < 20:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    pts1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_eq,
        curr_eq,
        pts0,
        None,
        winSize=(41, 41),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    back, st_back, _ = cv2.calcOpticalFlowPyrLK(
        curr_eq,
        prev_eq,
        pts1,
        None,
        winSize=(41, 41),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    pts0f = pts0.reshape(-1, 2)
    pts1f = pts1.reshape(-1, 2)
    backf = back.reshape(-1, 2)
    ok = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1)
    fb = np.linalg.norm(pts0f - backf, axis=1)
    ok &= fb < 2.5
    return pts0f[ok].astype(np.float32), pts1f[ok].astype(np.float32)


def diagnose_visual_rotation(
    bag_db: Path,
    left_topic: str,
    schedule: list[StereoStamp],
    origin_ns: int,
    poses: list[PoseSample],
    K: np.ndarray,
    R_bc: np.ndarray,
    max_pairs: int,
    time_offsets_s: list[float],
    time_source: str,
) -> dict[str, object]:
    if len(schedule) < 2:
        return {"error": "not enough stereo schedule rows"}
    selected = schedule[: max_pairs + 1] if max_pairs > 0 else schedule
    stamps = {s.left_ns for s in selected}
    images = read_images_for_stamps(bag_db, left_topic, stamps, time_source)
    rows = []
    candidate_errors: dict[str, list[float]] = {"current_body_T_cam": [], "inverse_rotation": [], "identity": []}
    offset_errors: dict[float, list[float]] = {offset: [] for offset in time_offsets_s}
    candidates = {
        "current_body_T_cam": R_bc,
        "inverse_rotation": R_bc.T,
        "identity": np.eye(3),
    }
    for prev_s, curr_s in zip(selected[:-1], selected[1:]):
        prev_msg = images.get(prev_s.left_ns)
        curr_msg = images.get(curr_s.left_ns)
        if prev_msg is None or curr_msg is None:
            continue
        t0 = (prev_s.left_ns - origin_ns) / 1e9
        t1 = (curr_s.left_ns - origin_ns) / 1e9
        p0 = interp_pose(poses, t0)
        p1 = interp_pose(poses, t1)
        if p0 is None or p1 is None:
            continue
        prev = image_to_gray(prev_msg)
        curr = image_to_gray(curr_msg)
        pts0, pts1 = track_pair(prev, curr)
        if len(pts0) < 30:
            continue
        E, mask = cv2.findEssentialMat(
            pts0,
            pts1,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5,
        )
        if E is None:
            continue
        _, R_vis, _, mask_pose = cv2.recoverPose(E, pts0, pts1, K)
        inliers = int(mask_pose.sum() / 255) if mask_pose is not None else int(mask.sum())
        if inliers < 25:
            continue
        R_wb0 = q_to_R(p0.q)
        R_wb1 = q_to_R(p1.q)
        row = {
            "t0": t0,
            "t1": t1,
            "tracked": int(len(pts0)),
            "inliers": inliers,
            "vision_rot_deg": math.degrees(R_angle(R_vis)),
        }
        for name, Rbc in candidates.items():
            R_pred = Rbc.T @ R_wb1.T @ R_wb0 @ Rbc
            err_deg = math.degrees(R_angle(R_vis @ R_pred.T))
            candidate_errors[name].append(err_deg)
            row[f"{name}_err_deg"] = err_deg
        for offset_s in time_offsets_s:
            op0 = interp_pose(poses, t0 + offset_s)
            op1 = interp_pose(poses, t1 + offset_s)
            if op0 is None or op1 is None:
                continue
            R_wb0_off = q_to_R(op0.q)
            R_wb1_off = q_to_R(op1.q)
            R_pred_off = R_bc.T @ R_wb1_off.T @ R_wb0_off @ R_bc
            offset_errors[offset_s].append(math.degrees(R_angle(R_vis @ R_pred_off.T)))
        rows.append(row)
    summary = {}
    for name, vals in candidate_errors.items():
        if vals:
            arr = np.array(vals, dtype=float)
            summary[name] = {
                "count": int(len(arr)),
                "median_error_deg": float(np.median(arr)),
                "mean_error_deg": float(np.mean(arr)),
                "p90_error_deg": float(np.percentile(arr, 90)),
            }
        else:
            summary[name] = {"count": 0}
    offset_summary = []
    for offset_s, vals in sorted(offset_errors.items()):
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        offset_summary.append(
            {
                "offset_ms": float(offset_s * 1000.0),
                "count": int(len(arr)),
                "median_error_deg": float(np.median(arr)),
                "mean_error_deg": float(np.mean(arr)),
                "p90_error_deg": float(np.percentile(arr, 90)),
            }
        )
    best_offset = min(offset_summary, key=lambda row: row["median_error_deg"]) if offset_summary else None
    return {"summary": summary, "time_offset_sweep": offset_summary, "best_time_offset": best_offset, "pairs": rows}


def diagnose_stereo_epipolar(
    bag_db: Path,
    left_topic: str,
    right_topic: str,
    schedule: list[StereoStamp],
    max_pairs: int,
    time_source: str,
) -> dict[str, object]:
    selected = schedule[:max_pairs] if max_pairs > 0 else schedule
    left_images = read_images_for_stamps(bag_db, left_topic, {s.left_ns for s in selected}, time_source)
    right_images = read_images_for_stamps(bag_db, right_topic, {s.right_ns for s in selected}, time_source)
    dy_all = []
    disp_all = []
    per_frame = []
    for s in selected:
        l_msg = left_images.get(s.left_ns)
        r_msg = right_images.get(s.right_ns)
        if l_msg is None or r_msg is None:
            continue
        left = image_to_gray(l_msg)
        right = image_to_gray(r_msg)
        pts_l, pts_r = track_pair(left, right)
        if len(pts_l) < 10:
            continue
        disparity = pts_l[:, 0] - pts_r[:, 0]
        dy = pts_l[:, 1] - pts_r[:, 1]
        good = (disparity > 1.0) & (disparity < 160.0) & (np.abs(dy) < 20.0)
        if not np.any(good):
            continue
        dy_good = dy[good]
        disp_good = disparity[good]
        dy_all.extend(dy_good.tolist())
        disp_all.extend(disp_good.tolist())
        per_frame.append(
            {
                "left_ns": s.left_ns,
                "tracked": int(len(pts_l)),
                "used": int(np.sum(good)),
                "median_abs_y_error_px": float(np.median(np.abs(dy_good))),
                "median_disparity_px": float(np.median(disp_good)),
            }
        )
    if not dy_all:
        return {"error": "no usable stereo matches"}
    dy_arr = np.array(dy_all, dtype=float)
    disp_arr = np.array(disp_all, dtype=float)
    return {
        "frames": int(len(per_frame)),
        "matches": int(len(dy_arr)),
        "median_abs_y_error_px": float(np.median(np.abs(dy_arr))),
        "p90_abs_y_error_px": float(np.percentile(np.abs(dy_arr), 90)),
        "median_y_error_px": float(np.median(dy_arr)),
        "median_disparity_px": float(np.median(disp_arr)),
        "p10_disparity_px": float(np.percentile(disp_arr, 10)),
        "p90_disparity_px": float(np.percentile(disp_arr, 90)),
        "per_frame": per_frame,
    }


def write_visual_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose VIO frame, stereo, and camera-IMU contracts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/rosbag_active/localization bag"))
    parser.add_argument("--config", type=Path, default=Path("VINS-Fusion-ROS2/config/realsense_d435i/realsense_stereo_mavros_imu_config.yaml"))
    parser.add_argument("--left-calib", type=Path, default=Path("VINS-Fusion-ROS2/config/realsense_d435i/left.yaml"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/evaluation/vio_contract_diagnostics.json"))
    parser.add_argument("--visual-csv", type=Path, default=Path("outputs/evaluation/vio_contract_visual_rotation.csv"))
    parser.add_argument("--max-visual-pairs", type=int, default=160)
    parser.add_argument("--max-stereo-pairs", type=int, default=120)
    parser.add_argument("--time-offset-min-ms", type=float, default=-150.0)
    parser.add_argument("--time-offset-max-ms", type=float, default=150.0)
    parser.add_argument("--time-offset-step-ms", type=float, default=10.0)
    args = parser.parse_args()

    manifest = json.loads((args.data_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    origin_ns = int(manifest["origin_stamp_ns"])
    topics = manifest["topics"]
    outputs = manifest["outputs"]
    bag_db = args.data_dir / outputs["stamp_bag"]
    schedule = read_stereo_schedule(args.data_dir / outputs["stereo_schedule_csv"])
    poses = read_csv_poses(args.data_dir / outputs["localization_odometry_csv"])
    imu = read_csv_imu(args.data_dir / outputs["imu_csv"])
    K = read_camera_matrix(args.left_calib)
    T_bc = read_body_T_cam(args.config, "body_T_cam0")
    R_bc = T_bc[:3, :3]
    if args.time_offset_step_ms <= 0:
        raise SystemExit("--time-offset-step-ms must be positive")
    time_offsets_s = [
        ms / 1000.0
        for ms in np.arange(
            args.time_offset_min_ms,
            args.time_offset_max_ms + 0.5 * args.time_offset_step_ms,
            args.time_offset_step_ms,
        )
    ]

    imu_axes = diagnose_imu_axes(poses, imu)
    visual = diagnose_visual_rotation(
        bag_db=bag_db,
        left_topic=topics["left"],
        schedule=schedule,
        origin_ns=origin_ns,
        poses=poses,
        K=K,
        R_bc=R_bc,
        max_pairs=args.max_visual_pairs,
        time_offsets_s=time_offsets_s,
        time_source=manifest.get("time_source", "db"),
    )
    stereo = diagnose_stereo_epipolar(
        bag_db=bag_db,
        left_topic=topics["left"],
        right_topic=topics["right"],
        schedule=schedule,
        max_pairs=args.max_stereo_pairs,
        time_source=manifest.get("time_source", "db"),
    )

    result = {
        "mode": "vio_contract_diagnostics",
        "dvl_usage": "not_used",
        "data_dir": str(args.data_dir),
        "bag_db": str(bag_db),
        "config": str(args.config),
        "origin_stamp_ns": origin_ns,
        "topics": topics,
        "camera": {
            "K": K.tolist(),
            "body_T_cam0": T_bc.tolist(),
            "det_R_body_cam0": float(np.linalg.det(R_bc)),
        },
        "imu_vs_localization_gyro": imu_axes,
        "left_image_visual_rotation_vs_localization_body_T_cam": visual["summary"],
        "image_localization_time_offset_sweep": visual.get("time_offset_sweep", []),
        "best_image_localization_time_offset": visual.get("best_time_offset"),
        "stereo_epipolar_contract": {k: v for k, v in stereo.items() if k != "per_frame"},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if "pairs" in visual:
        write_visual_csv(args.visual_csv, visual["pairs"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
