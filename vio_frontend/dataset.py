from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import PreprocessConfig, VideoDatasetConfig


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_sec: float
    bgr: np.ndarray
    gray: np.ndarray


class VideoFrameSource:
    def __init__(self, dataset: VideoDatasetConfig, preprocess: PreprocessConfig) -> None:
        self.dataset = dataset
        self.preprocess = preprocess
        self.fps = dataset.fallback_fps
        self.original_width = 0
        self.original_height = 0
        self.width = 0
        self.height = 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._clahe = None

    def __iter__(self):
        if not self.dataset.input_path.exists():
            raise FileNotFoundError(self.dataset.input_path)

        cap = cv2.VideoCapture(str(self.dataset.input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {self.dataset.input_path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if math.isfinite(source_fps) and source_fps > 0:
            self.fps = source_fps

        if self.dataset.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.dataset.start_frame)

        yielded = 0
        source_index = self.dataset.start_frame
        try:
            while True:
                if self.dataset.max_frames and yielded >= self.dataset.max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                yield self._make_frame(frame, source_index)
                yielded += 1
                source_index += 1
        finally:
            cap.release()

    def _make_frame(self, frame: np.ndarray, source_index: int) -> Frame:
        self.original_height, self.original_width = frame.shape[:2]
        bgr = self._resize(frame)
        self.height, self.width = bgr.shape[:2]
        self.scale_x = self.width / self.original_width
        self.scale_y = self.height / self.original_height
        gray = self._to_gray(bgr)
        return Frame(
            index=source_index,
            timestamp_sec=source_index / self.fps,
            bgr=bgr,
            gray=gray,
        )

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        width = self.preprocess.resize_width
        if width <= 0 or frame.shape[1] == width:
            return frame
        scale = width / frame.shape[1]
        height = max(1, int(round(frame.shape[0] * scale)))
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not self.preprocess.clahe:
            return gray
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return self._clahe.apply(gray)
