from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from .config import FeatureConfig, OpticalFlowConfig


@dataclass
class ActiveTracks:
    points: dict[int, np.ndarray] = field(default_factory=dict)
    next_track_id: int = 0

    def add_points(self, points: Iterable[np.ndarray]) -> list[tuple[int, np.ndarray]]:
        added = []
        for point in points:
            track_id = self.next_track_id
            self.points[track_id] = point.astype(np.float32)
            self.next_track_id += 1
            added.append((track_id, self.points[track_id]))
        return added

    def replace(self, points: dict[int, np.ndarray]) -> None:
        self.points = points

    def ids_and_points(self) -> tuple[list[int], np.ndarray]:
        ids = list(self.points.keys())
        points = np.array([self.points[track_id] for track_id in ids], dtype=np.float32)
        return ids, points.reshape(-1, 1, 2)

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class OpticalFlowResult:
    track_ids: list[int]
    prev_points: np.ndarray
    curr_points: np.ndarray
    forward_backward_errors: np.ndarray


class ShiTomasiDetector:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def detect(
        self,
        gray: np.ndarray,
        existing_points: Iterable[np.ndarray],
        max_new: int,
    ) -> list[np.ndarray]:
        if max_new <= 0:
            return []

        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for point in existing_points:
            x, y = point
            cv2.circle(
                mask,
                (int(round(x)), int(round(y))),
                self.config.min_distance,
                0,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_new,
            qualityLevel=self.config.quality,
            minDistance=self.config.min_distance,
            mask=mask,
            blockSize=self.config.block_size,
            useHarrisDetector=False,
        )
        if corners is None:
            return []
        return [corner.reshape(2).astype(np.float32) for corner in corners]


class KltOpticalFlowTracker:
    def __init__(self, config: OpticalFlowConfig) -> None:
        self.config = config

    def track(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        track_ids: list[int],
        prev_points: np.ndarray,
    ) -> OpticalFlowResult:
        empty_points = np.empty((0, 2), dtype=np.float32)
        empty_errors = np.empty((0,), dtype=np.float32)
        if len(track_ids) == 0:
            return OpticalFlowResult([], empty_points, empty_points, empty_errors)

        lk_kwargs = {
            "winSize": (self.config.window_size, self.config.window_size),
            "maxLevel": self.config.max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.config.max_iterations,
                self.config.epsilon,
            ),
        }
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_points,
            None,
            **lk_kwargs,
        )
        if curr_points is None or status is None:
            return OpticalFlowResult([], empty_points, empty_points, empty_errors)

        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray,
            prev_gray,
            curr_points,
            None,
            **lk_kwargs,
        )
        if back_points is None or back_status is None:
            return OpticalFlowResult([], empty_points, empty_points, empty_errors)

        status = status.reshape(-1).astype(bool)
        back_status = back_status.reshape(-1).astype(bool)
        finite = _finite_points(curr_points, curr_gray.shape)
        fb_errors = np.linalg.norm(prev_points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1)
        good = (
            status
            & back_status
            & finite
            & (fb_errors <= self.config.forward_backward_threshold)
        )

        kept_ids = [track_id for track_id, keep in zip(track_ids, good) if keep]
        return OpticalFlowResult(
            track_ids=kept_ids,
            prev_points=prev_points.reshape(-1, 2)[good],
            curr_points=curr_points.reshape(-1, 2)[good],
            forward_backward_errors=fb_errors[good].astype(np.float32),
        )


class DescriptorFeatureMatcher:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.method = config.method
        self.detector, self.norm_type = self._create_detector()

    def detect_points(self, gray: np.ndarray) -> list[np.ndarray]:
        keypoints, _ = self.detector.detectAndCompute(gray, None)
        keypoints = self._select_keypoints(keypoints or [])
        return [np.array(kp.pt, dtype=np.float32) for kp in keypoints]

    def match(self, prev_gray: np.ndarray, curr_gray: np.ndarray, first_track_id: int) -> OpticalFlowResult:
        empty_points = np.empty((0, 2), dtype=np.float32)
        empty_errors = np.empty((0,), dtype=np.float32)

        prev_keypoints, prev_descriptors = self.detector.detectAndCompute(prev_gray, None)
        curr_keypoints, curr_descriptors = self.detector.detectAndCompute(curr_gray, None)
        if prev_descriptors is None or curr_descriptors is None:
            return OpticalFlowResult([], empty_points, empty_points, empty_errors)
        prev_descriptors = np.ascontiguousarray(prev_descriptors)
        curr_descriptors = np.ascontiguousarray(curr_descriptors)
        if prev_descriptors.ndim != 2 or curr_descriptors.ndim != 2:
            return OpticalFlowResult([], empty_points, empty_points, empty_errors)
        if len(prev_keypoints or []) != len(prev_descriptors) or len(curr_keypoints or []) != len(curr_descriptors):
            return OpticalFlowResult([], empty_points, empty_points, empty_errors)

        prev_keypoints, prev_descriptors = self._select_keypoints_and_descriptors(
            prev_keypoints or [],
            prev_descriptors,
        )
        curr_keypoints, curr_descriptors = self._select_keypoints_and_descriptors(
            curr_keypoints or [],
            curr_descriptors,
        )
        # BFMatcher.knnMatch(k=2) asserts inside OpenCV when the train set has
        # fewer than two descriptors. Sparse underwater frames can hit this.
        if len(prev_keypoints) == 0 or len(curr_keypoints) < 2:
            return OpticalFlowResult([], empty_points, empty_points, empty_errors)
        if len(prev_descriptors) == 0 or len(curr_descriptors) < 2:
            return OpticalFlowResult([], empty_points, empty_points, empty_errors)

        matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)
        try:
            raw_matches = matcher.knnMatch(prev_descriptors, curr_descriptors, k=2)
        except cv2.error:
            return OpticalFlowResult([], empty_points, empty_points, empty_errors)
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
            best, second = match_pair
            if best.distance <= self.config.descriptor_ratio * second.distance:
                good_matches.append(best)

        if not good_matches:
            return OpticalFlowResult([], empty_points, empty_points, empty_errors)

        good_matches = sorted(good_matches, key=lambda match: match.distance)[: self.config.max_features]
        prev_points = np.array([prev_keypoints[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        curr_points = np.array([curr_keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float32)
        distances = np.array([m.distance for m in good_matches], dtype=np.float32)
        track_ids = list(range(first_track_id, first_track_id + len(good_matches)))
        return OpticalFlowResult(
            track_ids=track_ids,
            prev_points=prev_points,
            curr_points=curr_points,
            forward_backward_errors=distances,
        )

    def _create_detector(self):
        if self.method == "orb_match":
            return cv2.ORB_create(nfeatures=self.config.max_features * 3), cv2.NORM_HAMMING
        if self.method == "sift_match":
            return cv2.SIFT_create(nfeatures=self.config.max_features * 3), cv2.NORM_L2
        if self.method == "akaze_match":
            return cv2.AKAZE_create(), cv2.NORM_HAMMING
        raise ValueError(f"Unsupported descriptor feature method: {self.method}")

    def _select_keypoints_and_descriptors(self, keypoints, descriptors):
        indexed = list(enumerate(keypoints))
        selected_indices = [
            index
            for index, _ in self._select_indexed_keypoints(indexed)
            if index < len(descriptors)
        ]
        if not selected_indices:
            return [], descriptors[:0]
        selected_descriptors = np.ascontiguousarray(descriptors[np.array(selected_indices, dtype=np.intp)])
        return [keypoints[index] for index in selected_indices], selected_descriptors

    def _select_keypoints(self, keypoints):
        return [kp for _, kp in self._select_indexed_keypoints(list(enumerate(keypoints)))]

    def _select_indexed_keypoints(self, indexed_keypoints):
        selected = []
        selected_points = []
        sorted_keypoints = sorted(indexed_keypoints, key=lambda item: item[1].response, reverse=True)
        min_distance_sq = float(self.config.min_distance * self.config.min_distance)
        for index, keypoint in sorted_keypoints:
            point = np.array(keypoint.pt, dtype=np.float32)
            if any(float(np.sum((point - existing) ** 2)) < min_distance_sq for existing in selected_points):
                continue
            selected.append((index, keypoint))
            selected_points.append(point)
            if len(selected) >= self.config.max_features:
                break
        return selected


def _finite_points(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    xy = points.reshape(-1, 2)
    return (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
