from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraModel, OdometryConfig
from .dataset import Frame
from .feature_tracker import OpticalFlowResult
from .geometry import VerificationResult


@dataclass
class OdometryEstimate:
    frame_index: int
    timestamp_sec: float
    success: bool
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    pose_inliers: int
    pose_inlier_ratio: float
    scale_mode: str
    note: str


class VisualOdometryEstimator:
    def __init__(
        self,
        config: OdometryConfig,
        camera: CameraModel,
        width: int,
        height: int,
    ) -> None:
        self.config = config
        self.camera = camera
        self.width = width
        self.height = height
        self.position = np.zeros(3, dtype=np.float64)
        self.rotation_world_camera = np.eye(3, dtype=np.float64)
        self.camera_matrix, self.scale_mode = self._make_camera_matrix()

    def initial_estimate(self, frame: Frame) -> OdometryEstimate:
        return OdometryEstimate(
            frame_index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            success=True,
            position=self.position.copy(),
            quaternion_xyzw=_rotation_to_quaternion(self.rotation_world_camera),
            pose_inliers=0,
            pose_inlier_ratio=0.0,
            scale_mode=self.scale_mode,
            note="initial_pose",
        )

    def update(
        self,
        frame: Frame,
        flow: OpticalFlowResult,
        verification: VerificationResult,
    ) -> OdometryEstimate:
        if not self.config.enabled:
            return self._estimate(frame, False, 0, 0.0, "odometry_disabled")

        if len(verification.inlier_mask) == 0:
            return self._estimate(frame, False, 0, 0.0, "no_verified_tracks")

        points_prev = flow.prev_points[verification.inlier_mask]
        points_curr = flow.curr_points[verification.inlier_mask]
        if len(points_prev) < self.config.min_pose_inliers:
            return self._estimate(
                frame,
                False,
                len(points_prev),
                0.0,
                "not_enough_pose_inliers",
            )

        try:
            essential, mask = cv2.findEssentialMat(
                np.ascontiguousarray(points_prev.reshape(-1, 2), dtype=np.float32),
                np.ascontiguousarray(points_curr.reshape(-1, 2), dtype=np.float32),
                cameraMatrix=self.camera_matrix,
                method=cv2.RANSAC,
                prob=self.config.pose_ransac_confidence,
                threshold=self.config.pose_ransac_threshold,
            )
        except cv2.error:
            return self._estimate(frame, False, 0, 0.0, "essential_error")
        if essential is None or mask is None:
            return self._estimate(frame, False, 0, 0.0, "essential_degenerate")

        try:
            pose_inliers, relative_rotation, relative_translation, pose_mask = cv2.recoverPose(
                essential,
                np.ascontiguousarray(points_prev.reshape(-1, 2), dtype=np.float32),
                np.ascontiguousarray(points_curr.reshape(-1, 2), dtype=np.float32),
                cameraMatrix=self.camera_matrix,
                mask=mask,
            )
        except cv2.error:
            return self._estimate(frame, False, 0, 0.0, "recover_pose_error")
        pose_inliers = int(pose_inliers)
        ratio = pose_inliers / len(points_prev) if len(points_prev) else 0.0
        if pose_inliers < self.config.min_pose_inliers:
            return self._estimate(frame, False, pose_inliers, ratio, "recover_pose_low_inliers")

        relative_translation = relative_translation.reshape(3) * self.config.motion_scale
        camera_motion_prev = -relative_rotation.T @ relative_translation
        previous_rotation = self.rotation_world_camera.copy()
        self.position = self.position + previous_rotation @ camera_motion_prev
        self.rotation_world_camera = previous_rotation @ relative_rotation.T
        return self._estimate(frame, True, pose_inliers, ratio, "relative_pose_integrated")

    def _estimate(
        self,
        frame: Frame,
        success: bool,
        pose_inliers: int,
        pose_inlier_ratio: float,
        note: str,
    ) -> OdometryEstimate:
        return OdometryEstimate(
            frame_index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            success=success,
            position=self.position.copy(),
            quaternion_xyzw=_rotation_to_quaternion(self.rotation_world_camera),
            pose_inliers=pose_inliers,
            pose_inlier_ratio=pose_inlier_ratio,
            scale_mode=self.scale_mode,
            note=note,
        )

    def _make_camera_matrix(self) -> tuple[np.ndarray, str]:
        if self.camera.has_intrinsics:
            return (
                np.array(
                    [
                        [float(self.camera.fx), 0.0, float(self.camera.cx)],
                        [0.0, float(self.camera.fy), float(self.camera.cy)],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                "calibrated_intrinsics_arbitrary_translation_scale",
            )

        focal = max(self.width, self.height) * self.config.assumed_focal_scale
        return (
            np.array(
                [
                    [focal, 0.0, self.width * 0.5],
                    [0.0, focal, self.height * 0.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            "assumed_intrinsics_arbitrary_translation_scale",
        )


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s

    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm
