from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .config import PipelineConfig
from .dataset import VideoFrameSource
from .feature_tracker import (
    ActiveTracks,
    DescriptorFeatureMatcher,
    KltOpticalFlowTracker,
    OpticalFlowResult,
    ShiTomasiDetector,
)
from .geometry import GeometricVerifier, VerificationResult
from .imu import CsvImuSource
from .odometry import VisualOdometryEstimator
from .outputs import OdometryCsvWriter, OutputPaths, TrackCsvWriter, VisualizationWriter, write_summary


class VioFrontendPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self) -> dict[str, object]:
        input_path = self.config.dataset.input_path
        output_dir = self.config.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = OutputPaths.build(output_dir, input_path.stem)

        imu_measurements = CsvImuSource(self.config.imu).load()
        source = VideoFrameSource(self.config.dataset, self.config.preprocess)
        frames = iter(source)

        try:
            first_frame = next(frames)
        except StopIteration as exc:
            raise RuntimeError(f"No frames could be read from: {input_path}") from exc

        camera = self.config.camera.scaled(source.scale_x, source.scale_y)
        use_klt = self.config.features.method == "shi_tomasi_klt"
        detector = ShiTomasiDetector(self.config.features)
        tracker = KltOpticalFlowTracker(self.config.optical_flow)
        descriptor_matcher = None if use_klt else DescriptorFeatureMatcher(self.config.features)
        verifier = GeometricVerifier(self.config.geometry, camera)
        odometry = VisualOdometryEstimator(
            self.config.odometry,
            camera,
            source.width,
            source.height,
        )
        active_tracks = ActiveTracks()

        visualizer = VisualizationWriter(
            paths.tracked_video,
            source.fps,
            source.width,
            source.height,
            self.config.output,
        )

        frame_stats: list[dict[str, object]] = []
        frames_processed = 1
        previous_frame = first_frame

        pose_success_count = 0
        odometry_estimates = []
        descriptor_next_track_id = 0

        with TrackCsvWriter(paths.tracks_csv) as track_log, OdometryCsvWriter(paths.odometry_csv) as odom_log:
            if use_klt:
                initial_points = detector.detect(
                    first_frame.gray,
                    [],
                    self.config.features.max_features,
                )
            else:
                initial_points = descriptor_matcher.detect_points(first_frame.gray)
            for track_id, point in active_tracks.add_points(initial_points):
                track_log.write_detection(first_frame, track_id, point)
            descriptor_next_track_id = active_tracks.next_track_id

            initial_odometry = odometry.initial_estimate(first_frame)
            odom_log.write(initial_odometry)
            odometry_estimates.append(initial_odometry)

            empty_flow = OpticalFlowResult(
                track_ids=[],
                prev_points=np.empty((0, 2), dtype=np.float32),
                curr_points=np.empty((0, 2), dtype=np.float32),
                forward_backward_errors=np.empty((0,), dtype=np.float32),
            )
            empty_verification = VerificationResult(
                inlier_mask=np.empty((0,), dtype=bool),
                model_used="none",
                ransac_used=False,
            )
            visualizer.write(
                first_frame,
                len(active_tracks),
                empty_flow,
                empty_verification,
                initial_points,
            )

            for frame in frames:
                if use_klt:
                    track_ids, prev_points = active_tracks.ids_and_points()
                    flow = tracker.track(previous_frame.gray, frame.gray, track_ids, prev_points)
                else:
                    flow = descriptor_matcher.match(
                        previous_frame.gray,
                        frame.gray,
                        descriptor_next_track_id,
                    )
                    descriptor_next_track_id += len(flow.track_ids)

                verification = verifier.verify(flow.prev_points, flow.curr_points)
                odometry_estimate = odometry.update(frame, flow, verification)
                odom_log.write(odometry_estimate)
                odometry_estimates.append(odometry_estimate)
                if odometry_estimate.success:
                    pose_success_count += 1

                next_active = {}
                for track_id, prev_point, curr_point, fb_error, inlier in zip(
                    flow.track_ids,
                    flow.prev_points,
                    flow.curr_points,
                    flow.forward_backward_errors,
                    verification.inlier_mask,
                ):
                    track_log.write_tracking(
                        frame,
                        track_id,
                        prev_point,
                        curr_point,
                        float(fb_error),
                        bool(inlier),
                        verification.model_used,
                    )
                    if use_klt and inlier:
                        next_active[track_id] = curr_point.astype(np.float32)

                if use_klt:
                    active_tracks.replace(next_active)
                    new_points = self._refill_tracks(frame.gray, active_tracks, detector, frames_processed)
                    for track_id, point in active_tracks.add_points(new_points):
                        track_log.write_detection(frame, track_id, point)
                    active_count = len(active_tracks)
                else:
                    new_points = []
                    active_count = len(flow.track_ids)

                visualizer.write(frame, active_count, flow, verification, new_points)
                frame_stats.append(
                    self._make_frame_stats(frame.index, active_count, flow, verification, new_points)
                )
                previous_frame = frame
                frames_processed += 1

        visualizer.close()
        total_tracks_created = active_tracks.next_track_id if use_klt else descriptor_next_track_id
        summary = self._make_summary(
            paths=paths,
            source=source,
            frames_processed=frames_processed,
            total_tracks_created=total_tracks_created,
            frame_stats=frame_stats,
            imu_measurement_count=len(imu_measurements),
            pose_success_count=pose_success_count,
            odometry_estimates=odometry_estimates,
        )
        write_summary(paths.summary_json, summary)
        return summary

    def _refill_tracks(
        self,
        gray: np.ndarray,
        active_tracks: ActiveTracks,
        detector: ShiTomasiDetector,
        frames_processed: int,
    ) -> list[np.ndarray]:
        should_refill = len(active_tracks) < self.config.features.min_features
        interval = self.config.features.detect_interval
        if interval > 0 and frames_processed % interval == 0:
            should_refill = True
        if not should_refill:
            return []

        max_new = max(0, self.config.features.max_features - len(active_tracks))
        return detector.detect(gray, active_tracks.points.values(), max_new)

    def _make_frame_stats(
        self,
        frame_index: int,
        active_count: int,
        flow: OpticalFlowResult,
        verification: VerificationResult,
        new_points: list[np.ndarray],
    ) -> dict[str, object]:
        tracked = int(len(verification.inlier_mask))
        inliers = int(verification.inlier_mask.sum()) if tracked else 0
        return {
            "frame_index": frame_index,
            "active_tracks": active_count,
            "tracked_candidates": tracked,
            "geometry_inliers": inliers,
            "geometry_inlier_ratio": inliers / tracked if tracked else 0.0,
            "new_features": len(new_points),
            "geometry_model": verification.model_used,
            "ransac_used": verification.ransac_used,
        }

    def _make_summary(
        self,
        paths: OutputPaths,
        source: VideoFrameSource,
        frames_processed: int,
        total_tracks_created: int,
        frame_stats: list[dict[str, object]],
        imu_measurement_count: int,
        pose_success_count: int,
        odometry_estimates,
    ) -> dict[str, object]:
        mean_active = _mean(row["active_tracks"] for row in frame_stats)
        mean_inliers = _mean(row["geometry_inliers"] for row in frame_stats)
        mean_ratio = _mean(
            row["geometry_inlier_ratio"]
            for row in frame_stats
            if int(row["tracked_candidates"]) > 0
        )
        return {
            "mode": "vio_frontend_visual_only" if imu_measurement_count == 0 else "vio_frontend_with_imu_input",
            "input": str(self.config.dataset.input_path),
            "frames_processed": frames_processed,
            "fps": source.fps,
            "width": source.width,
            "height": source.height,
            "feature_method": self.config.features.method,
            "camera_intrinsics_available": self.config.camera.has_intrinsics,
            "imu_measurements_loaded": imu_measurement_count,
            "odometry_enabled": self.config.odometry.enabled,
            "pose_success_count": pose_success_count,
            "pose_success_ratio": pose_success_count / max(1, frames_processed - 1),
            "trajectory_scale_mode": (
                odometry_estimates[-1].scale_mode
                if odometry_estimates
                else "none"
            ),
            "total_tracks_created": total_tracks_created,
            "mean_active_tracks": mean_active,
            "mean_geometry_inliers": mean_inliers,
            "mean_geometry_inlier_ratio": mean_ratio,
            "output_video": str(paths.tracked_video) if self.config.output.write_video else "",
            "output_csv": str(paths.tracks_csv),
            "output_odometry_csv": str(paths.odometry_csv),
            "output_summary": str(paths.summary_json),
            "frontend_stages": [
                "video_frame_source",
                "grayscale_preprocess",
                (
                    "shi_tomasi_detection"
                    if self.config.features.method == "shi_tomasi_klt"
                    else f"{self.config.features.method}_detection"
                ),
                (
                    "klt_forward_backward_tracking"
                    if self.config.features.method == "shi_tomasi_klt"
                    else "descriptor_ratio_matching"
                ),
                "essential_or_fundamental_ransac",
                "track_log_export",
            ],
            "full_vio_requirements_not_in_mp4": [
                "camera_intrinsics_and_distortion",
                "imu_gyro_accel_measurements",
                "camera_imu_extrinsic",
                "camera_imu_time_offset",
                "backend_state_estimator",
            ],
            "config": _json_safe_asdict(self.config),
            "frame_stats": frame_stats,
        }


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(values))


def _json_safe_asdict(config: PipelineConfig) -> dict[str, object]:
    data = asdict(config)
    return json.loads(json.dumps(data, default=str))
