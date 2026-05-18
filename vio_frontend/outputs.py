from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import OutputConfig
from .dataset import Frame
from .feature_tracker import OpticalFlowResult
from .geometry import VerificationResult
from .odometry import OdometryEstimate


@dataclass(frozen=True)
class OutputPaths:
    tracked_video: Path
    tracks_csv: Path
    odometry_csv: Path
    summary_json: Path

    @classmethod
    def build(cls, output_dir: Path, input_stem: str) -> "OutputPaths":
        return cls(
            tracked_video=output_dir / f"{input_stem}_tracked.mp4",
            tracks_csv=output_dir / f"{input_stem}_tracks.csv",
            odometry_csv=output_dir / f"{input_stem}_odometry.csv",
            summary_json=output_dir / f"{input_stem}_summary.json",
        )


class TrackCsvWriter:
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "track_id",
        "x",
        "y",
        "prev_x",
        "prev_y",
        "fb_error",
        "geometry_inlier",
        "geometry_model",
        "event",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def close(self) -> None:
        self._file.close()

    def write_detection(self, frame: Frame, track_id: int, point: np.ndarray) -> None:
        self._writer.writerow(
            {
                "frame_index": frame.index,
                "timestamp_sec": f"{frame.timestamp_sec:.6f}",
                "track_id": track_id,
                "x": f"{point[0]:.3f}",
                "y": f"{point[1]:.3f}",
                "prev_x": "",
                "prev_y": "",
                "fb_error": "",
                "geometry_inlier": "",
                "geometry_model": "",
                "event": "detected",
            }
        )

    def write_tracking(
        self,
        frame: Frame,
        track_id: int,
        prev_point: np.ndarray,
        curr_point: np.ndarray,
        fb_error: float,
        inlier: bool,
        model: str,
    ) -> None:
        self._writer.writerow(
            {
                "frame_index": frame.index,
                "timestamp_sec": f"{frame.timestamp_sec:.6f}",
                "track_id": track_id,
                "x": f"{curr_point[0]:.3f}",
                "y": f"{curr_point[1]:.3f}",
                "prev_x": f"{prev_point[0]:.3f}",
                "prev_y": f"{prev_point[1]:.3f}",
                "fb_error": f"{float(fb_error):.4f}",
                "geometry_inlier": int(bool(inlier)),
                "geometry_model": model,
                "event": "tracked" if inlier else "rejected",
            }
        )

    def __enter__(self) -> "TrackCsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class VisualizationWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int, config: OutputConfig) -> None:
        self.config = config
        self.path = path
        self._writer = None
        if config.write_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {path}")

    def write(
        self,
        frame: Frame,
        active_count: int,
        flow: OpticalFlowResult,
        verification: VerificationResult,
        new_points: list[np.ndarray],
    ) -> None:
        if self._writer is None:
            return
        self._writer.write(
            draw_frontend_overlay(frame, active_count, flow, verification, new_points, self.config)
        )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()


class OdometryCsvWriter:
    fieldnames = [
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
        "scale_mode",
        "note",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def write(self, estimate: OdometryEstimate) -> None:
        x, y, z = estimate.position
        qx, qy, qz, qw = estimate.quaternion_xyzw
        self._writer.writerow(
            {
                "frame_index": estimate.frame_index,
                "timestamp_sec": f"{estimate.timestamp_sec:.6f}",
                "x": f"{float(x):.6f}",
                "y": f"{float(y):.6f}",
                "z": f"{float(z):.6f}",
                "qx": f"{float(qx):.8f}",
                "qy": f"{float(qy):.8f}",
                "qz": f"{float(qz):.8f}",
                "qw": f"{float(qw):.8f}",
                "pose_success": int(estimate.success),
                "pose_inliers": estimate.pose_inliers,
                "pose_inlier_ratio": f"{estimate.pose_inlier_ratio:.6f}",
                "scale_mode": estimate.scale_mode,
                "note": estimate.note,
            }
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "OdometryCsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def draw_frontend_overlay(
    frame: Frame,
    active_count: int,
    flow: OpticalFlowResult,
    verification: VerificationResult,
    new_points: list[np.ndarray],
    config: OutputConfig,
) -> np.ndarray:
    canvas = frame.bgr.copy()
    inliers = verification.inlier_mask

    for prev, curr, keep in zip(flow.prev_points, flow.curr_points, inliers):
        p0 = tuple(np.round(prev).astype(int))
        p1 = tuple(np.round(curr).astype(int))
        if keep:
            cv2.line(canvas, p0, p1, (0, 220, 80), 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, (0, 255, 120), -1, cv2.LINE_AA)
        elif config.draw_rejected:
            cv2.line(canvas, p0, p1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, (0, 0, 255), -1, cv2.LINE_AA)

    for point in new_points:
        x, y = tuple(np.round(point).astype(int))
        cv2.circle(canvas, (x, y), 2, (255, 180, 0), -1, cv2.LINE_AA)

    tracked_count = len(inliers)
    inlier_count = int(inliers.sum()) if tracked_count else 0
    ratio = inlier_count / tracked_count if tracked_count else 0.0
    text = (
        f"frame {frame.index}  active {active_count}  "
        f"inliers {inlier_count}/{tracked_count} ({ratio:.2f})  {verification.model_used}"
    )
    cv2.rectangle(canvas, (10, 10), (760, 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
