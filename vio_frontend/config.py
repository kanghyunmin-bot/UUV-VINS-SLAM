from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoDatasetConfig:
    input_path: Path = Path("video_src/camera_camera_color_image_raw.mp4")
    start_frame: int = 0
    max_frames: int = 0
    fallback_fps: float = 30.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "VideoDatasetConfig":
        defaults = cls()
        return cls(
            input_path=Path(data.get("input_path", defaults.input_path)),
            start_frame=int(data.get("start_frame", defaults.start_frame)),
            max_frames=int(data.get("max_frames", defaults.max_frames)),
            fallback_fps=float(data.get("fallback_fps", defaults.fallback_fps)),
        )


@dataclass
class CameraModel:
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    distortion: list[float] = field(default_factory=list)

    @property
    def has_intrinsics(self) -> bool:
        return self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None

    def scaled(self, sx: float, sy: float) -> "CameraModel":
        if not self.has_intrinsics:
            return self
        return CameraModel(
            fx=float(self.fx) * sx,
            fy=float(self.fy) * sy,
            cx=float(self.cx) * sx,
            cy=float(self.cy) * sy,
            distortion=list(self.distortion),
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "CameraModel":
        distortion = data.get("distortion", [])
        return cls(
            fx=_optional_float(data.get("fx")),
            fy=_optional_float(data.get("fy")),
            cx=_optional_float(data.get("cx")),
            cy=_optional_float(data.get("cy")),
            distortion=[float(value) for value in distortion] if distortion else [],
        )


@dataclass
class PreprocessConfig:
    resize_width: int = 0
    clahe: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PreprocessConfig":
        defaults = cls()
        return cls(
            resize_width=int(data.get("resize_width", defaults.resize_width)),
            clahe=bool(data.get("clahe", defaults.clahe)),
        )


@dataclass
class FeatureConfig:
    method: str = "shi_tomasi_klt"
    max_features: int = 200
    min_features: int = 80
    detect_interval: int = 5
    quality: float = 0.01
    min_distance: int = 30
    block_size: int = 7
    descriptor_ratio: float = 0.75

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "FeatureConfig":
        defaults = cls()
        return cls(
            method=str(data.get("method", defaults.method)),
            max_features=int(data.get("max_features", defaults.max_features)),
            min_features=int(data.get("min_features", defaults.min_features)),
            detect_interval=int(data.get("detect_interval", defaults.detect_interval)),
            quality=float(data.get("quality", defaults.quality)),
            min_distance=int(data.get("min_distance", defaults.min_distance)),
            block_size=int(data.get("block_size", defaults.block_size)),
            descriptor_ratio=float(data.get("descriptor_ratio", defaults.descriptor_ratio)),
        )


@dataclass
class OpticalFlowConfig:
    window_size: int = 21
    max_level: int = 3
    max_iterations: int = 30
    epsilon: float = 0.01
    forward_backward_threshold: float = 0.8

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "OpticalFlowConfig":
        defaults = cls()
        return cls(
            window_size=int(data.get("window_size", defaults.window_size)),
            max_level=int(data.get("max_level", defaults.max_level)),
            max_iterations=int(data.get("max_iterations", defaults.max_iterations)),
            epsilon=float(data.get("epsilon", defaults.epsilon)),
            forward_backward_threshold=float(
                data.get("forward_backward_threshold", defaults.forward_backward_threshold)
            ),
        )


@dataclass
class GeometryConfig:
    model: str = "auto"
    ransac_threshold: float = 0.75
    ransac_confidence: float = 0.999

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "GeometryConfig":
        defaults = cls()
        return cls(
            model=str(data.get("model", defaults.model)),
            ransac_threshold=float(data.get("ransac_threshold", defaults.ransac_threshold)),
            ransac_confidence=float(data.get("ransac_confidence", defaults.ransac_confidence)),
        )


@dataclass
class OdometryConfig:
    enabled: bool = True
    assumed_focal_scale: float = 1.2
    motion_scale: float = 0.05
    min_pose_inliers: int = 20
    pose_ransac_threshold: float = 1.0
    pose_ransac_confidence: float = 0.999

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "OdometryConfig":
        defaults = cls()
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            assumed_focal_scale=float(data.get("assumed_focal_scale", defaults.assumed_focal_scale)),
            motion_scale=float(data.get("motion_scale", defaults.motion_scale)),
            min_pose_inliers=int(data.get("min_pose_inliers", defaults.min_pose_inliers)),
            pose_ransac_threshold=float(
                data.get("pose_ransac_threshold", defaults.pose_ransac_threshold)
            ),
            pose_ransac_confidence=float(
                data.get("pose_ransac_confidence", defaults.pose_ransac_confidence)
            ),
        )


@dataclass
class ImuConfig:
    csv_path: Path | None = None
    timestamp_column: str = "timestamp_sec"
    gyro_columns: tuple[str, str, str] = ("gyro_x", "gyro_y", "gyro_z")
    accel_columns: tuple[str, str, str] = ("accel_x", "accel_y", "accel_z")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ImuConfig":
        defaults = cls()
        path_value = data.get("csv_path")
        gyro_columns = tuple(data.get("gyro_columns", defaults.gyro_columns))
        accel_columns = tuple(data.get("accel_columns", defaults.accel_columns))
        return cls(
            csv_path=Path(path_value) if path_value else None,
            timestamp_column=str(data.get("timestamp_column", defaults.timestamp_column)),
            gyro_columns=(str(gyro_columns[0]), str(gyro_columns[1]), str(gyro_columns[2])),
            accel_columns=(str(accel_columns[0]), str(accel_columns[1]), str(accel_columns[2])),
        )


@dataclass
class OutputConfig:
    output_dir: Path = Path("outputs")
    write_video: bool = True
    draw_rejected: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "OutputConfig":
        defaults = cls()
        return cls(
            output_dir=Path(data.get("output_dir", defaults.output_dir)),
            write_video=bool(data.get("write_video", defaults.write_video)),
            draw_rejected=bool(data.get("draw_rejected", defaults.draw_rejected)),
        )


@dataclass
class PipelineConfig:
    dataset: VideoDatasetConfig = field(default_factory=VideoDatasetConfig)
    camera: CameraModel = field(default_factory=CameraModel)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    odometry: OdometryConfig = field(default_factory=OdometryConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PipelineConfig":
        return cls(
            dataset=VideoDatasetConfig.from_mapping(data.get("dataset", {})),
            camera=CameraModel.from_mapping(data.get("camera", {})),
            preprocess=PreprocessConfig.from_mapping(data.get("preprocess", {})),
            features=FeatureConfig.from_mapping(data.get("features", {})),
            optical_flow=OpticalFlowConfig.from_mapping(data.get("optical_flow", {})),
            geometry=GeometryConfig.from_mapping(data.get("geometry", {})),
            odometry=OdometryConfig.from_mapping(data.get("odometry", {})),
            imu=ImuConfig.from_mapping(data.get("imu", {})),
            output=OutputConfig.from_mapping(data.get("output", {})),
        )


def load_pipeline_config(path: Path) -> PipelineConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return PipelineConfig.from_mapping(data)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
