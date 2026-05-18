from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraModel, GeometryConfig


@dataclass
class VerificationResult:
    inlier_mask: np.ndarray
    model_used: str
    ransac_used: bool


class GeometricVerifier:
    def __init__(self, config: GeometryConfig, camera: CameraModel) -> None:
        self.config = config
        self.camera = camera

    def verify(self, prev_points: np.ndarray, curr_points: np.ndarray) -> VerificationResult:
        count = len(prev_points)
        if count == 0:
            return VerificationResult(np.empty((0,), dtype=bool), "none", False)
        if self.config.model == "none":
            return VerificationResult(np.ones(count, dtype=bool), "none", False)

        model = self._selected_model()
        min_points = 5 if model == "essential" else 8
        if count < min_points:
            return VerificationResult(np.zeros(count, dtype=bool), f"{model}:not_enough_points", False)

        if model == "essential":
            mask = self._essential_inliers(prev_points, curr_points)
        else:
            mask = self._fundamental_inliers(prev_points, curr_points)

        if mask is None:
            # Pure rotation, repeated pool tiles, or low parallax can make the model degenerate.
            return VerificationResult(np.zeros(count, dtype=bool), f"{model}:degenerate", False)
        return VerificationResult(mask, model, True)

    def _selected_model(self) -> str:
        if self.config.model == "auto":
            return "essential" if self.camera.has_intrinsics else "fundamental"
        if self.config.model not in {"essential", "fundamental"}:
            raise ValueError(f"Unsupported geometry model: {self.config.model}")
        if self.config.model == "essential" and not self.camera.has_intrinsics:
            raise ValueError("Essential-matrix verification requires camera intrinsics")
        return self.config.model

    def _fundamental_inliers(self, prev_points: np.ndarray, curr_points: np.ndarray) -> np.ndarray | None:
        try:
            _, mask = cv2.findFundamentalMat(
                np.ascontiguousarray(prev_points.reshape(-1, 2), dtype=np.float32),
                np.ascontiguousarray(curr_points.reshape(-1, 2), dtype=np.float32),
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
                confidence=self.config.ransac_confidence,
            )
        except cv2.error:
            return None
        if mask is None:
            return None
        if len(mask.reshape(-1)) != len(prev_points):
            return None
        return mask.reshape(-1).astype(bool)

    def _essential_inliers(self, prev_points: np.ndarray, curr_points: np.ndarray) -> np.ndarray | None:
        camera_matrix = np.array(
            [
                [float(self.camera.fx), 0.0, float(self.camera.cx)],
                [0.0, float(self.camera.fy), float(self.camera.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = (
            np.array(self.camera.distortion, dtype=np.float64).reshape(-1, 1)
            if self.camera.distortion
            else None
        )

        if dist_coeffs is not None:
            prev = cv2.undistortPoints(prev_points.reshape(-1, 1, 2), camera_matrix, dist_coeffs)
            curr = cv2.undistortPoints(curr_points.reshape(-1, 1, 2), camera_matrix, dist_coeffs)
            normalized_threshold = self.config.ransac_threshold / (
                (float(self.camera.fx) + float(self.camera.fy)) * 0.5
            )
            try:
                _, mask = cv2.findEssentialMat(
                    np.ascontiguousarray(prev.reshape(-1, 2), dtype=np.float32),
                    np.ascontiguousarray(curr.reshape(-1, 2), dtype=np.float32),
                    cameraMatrix=np.eye(3, dtype=np.float64),
                    method=cv2.RANSAC,
                    prob=self.config.ransac_confidence,
                    threshold=normalized_threshold,
                )
            except cv2.error:
                return None
        else:
            try:
                _, mask = cv2.findEssentialMat(
                    np.ascontiguousarray(prev_points.reshape(-1, 2), dtype=np.float32),
                    np.ascontiguousarray(curr_points.reshape(-1, 2), dtype=np.float32),
                    cameraMatrix=camera_matrix,
                    method=cv2.RANSAC,
                    prob=self.config.ransac_confidence,
                    threshold=self.config.ransac_threshold,
                )
            except cv2.error:
                return None

        if mask is None:
            return None
        if len(mask.reshape(-1)) != len(prev_points):
            return None
        return mask.reshape(-1).astype(bool)
