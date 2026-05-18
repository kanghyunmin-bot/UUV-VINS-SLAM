#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import deque
from collections import defaultdict
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


FeaturePoint = tuple[int, float, float, float, int]


class OdometryCsvPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("vio_frontend_odometry_csv_publisher")
        self.csv_path = args.csv
        self.watch = args.watch
        self.last_mtime_ns = 0
        self.index = 0
        self.loop = args.loop
        self.frame_id = args.frame_id
        self.child_frame_id = args.child_frame_id
        self.coordinate_frame = args.coordinate_frame
        self.yaw_offset_rad = math.radians(args.yaw_offset_deg)
        self.orientation_source = args.orientation_source
        self.success_only_path = args.success_only_path
        self.rows = []
        self.finished = False
        self.cloud_csv_path = args.point_cloud_csv
        self.cloud_rows_by_frame: dict[int, list[FeaturePoint]] = {}
        self.cloud_mtime_ns = 0
        self.cloud_source_odom_csv_path = args.point_cloud_source_odom_csv
        self.cloud_source_rows_by_frame: dict[int, dict[str, str]] = {}
        self.cloud_source_mtime_ns = 0
        self.feature_history_frames = args.feature_history_frames
        self.feature_track_history = args.feature_track_history
        self.feature_track_publish_stride = max(1, int(args.feature_track_publish_stride))
        self.point_cloud_display_mode = args.point_cloud_display_mode
        self.point_cloud_transform_mode = args.point_cloud_transform_mode
        self.point_cloud_min_track_age = args.point_cloud_min_track_age
        self.point_cloud_min_depth = args.point_cloud_min_depth
        self.point_cloud_max_depth = args.point_cloud_max_depth
        self.point_cloud_max_range = args.point_cloud_max_range
        self.point_cloud_max_path_distance = args.point_cloud_max_path_distance
        self.point_cloud_local_scale = args.point_cloud_local_scale
        self.history_frames: deque[tuple[int, list[tuple[float, float, float]]]] = deque()
        self.track_history: dict[int, deque[tuple[float, float, float]]] = defaultdict(self._new_track_history)
        self.track_last_frame: dict[int, int] = {}
        self.force_marker_publish = True
        self.zero_start = args.zero_start
        self.success_only_cloud = args.success_only_cloud
        self.origin_position: tuple[float, float, float] | None = None

        self.odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)
        self.path_pub = self.create_publisher(RosPath, args.path_topic, 10)
        self.vins_odom_pub = self.create_publisher(Odometry, args.vins_odom_topic, 10)
        self.vins_path_pub = self.create_publisher(RosPath, args.vins_path_topic, 10)
        self.point_cloud_pub = self.create_publisher(PointCloud2, args.point_cloud_topic, 10)
        self.history_cloud_pub = self.create_publisher(PointCloud2, args.history_point_cloud_topic, 10)
        self.feature_track_pub = self.create_publisher(MarkerArray, args.feature_track_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.path_msg = RosPath()
        self.path_msg.header.frame_id = self.frame_id
        self.vins_path_msg = RosPath()
        self.vins_path_msg.header.frame_id = self.frame_id

        self._reload_rows(initial=True)
        self._reload_cloud_rows(initial=True)
        self._reload_cloud_source_rows(initial=True)
        if not self.rows and not self.watch:
            raise RuntimeError(f"No rows in odometry CSV: {args.csv}")

        period = 1.0 / max(args.rate, 0.1)
        self.timer = self.create_timer(period, self._publish_next)
        playback_seconds = len(self.rows) / max(args.rate, 0.1) if self.rows else 0.0
        self.get_logger().info(
            f"Publishing {len(self.rows)} odometry rows at {args.rate:.1f} Hz "
            f"(~{playback_seconds:.1f}s per pass) to {args.odom_topic} and {args.path_topic}"
        )

    def _publish_next(self) -> None:
        if self.watch:
            self._reload_rows(initial=False)
            self._reload_cloud_rows(initial=False)
            self._reload_cloud_source_rows(initial=False)
        if not self.rows:
            return

        if self.index >= len(self.rows):
            if self.loop:
                self.index = 0
                self.path_msg.poses.clear()
                self.vins_path_msg.poses.clear()
                self._clear_feature_history()
            else:
                self.get_logger().info("Finished odometry CSV playback")
                self.finished = True
                return

        row = self.rows[self.index]
        stamp = self.get_clock().now().to_msg()
        odom = self._make_odom(row, self.index, stamp)
        pose = self._make_pose(row, self.index, stamp)
        transform = self._make_transform(row, self.index, stamp)

        self.odom_pub.publish(odom)
        self.vins_odom_pub.publish(odom)
        self.tf_broadcaster.sendTransform(transform)
        if not self.success_only_path or row["pose_success"] == "1":
            self.path_msg.header.stamp = stamp
            self.path_msg.poses.append(pose)
            self.vins_path_msg.header.stamp = stamp
            self.vins_path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)
        self.vins_path_pub.publish(self.vins_path_msg)
        self._publish_point_cloud(row, stamp)
        self.index += 1

    def _reload_rows(self, initial: bool) -> None:
        if not self.csv_path.exists():
            if initial:
                self.get_logger().warn(f"Waiting for odometry CSV: {self.csv_path}")
            return
        mtime_ns = self.csv_path.stat().st_mtime_ns
        if not initial and mtime_ns == self.last_mtime_ns:
            return
        rows = _load_rows(self.csv_path)
        if not rows:
            return
        self.rows = rows
        self.index = 0
        self.origin_position = _raw_position(rows[0]) if self.zero_start else None
        self.path_msg = RosPath()
        self.path_msg.header.frame_id = self.frame_id
        self.vins_path_msg = RosPath()
        self.vins_path_msg.header.frame_id = self.frame_id
        self.last_mtime_ns = mtime_ns
        self.get_logger().info(f"Loaded {len(self.rows)} odometry rows from {self.csv_path}")

    def _reload_cloud_rows(self, initial: bool) -> None:
        if self.cloud_csv_path is None:
            return
        if not self.cloud_csv_path.exists():
            if initial:
                self.get_logger().warn(f"Point cloud CSV not found yet: {self.cloud_csv_path}")
            return
        mtime_ns = self.cloud_csv_path.stat().st_mtime_ns
        if not initial and mtime_ns == self.cloud_mtime_ns:
            return
        self.cloud_rows_by_frame = _load_cloud_rows(self.cloud_csv_path)
        if not initial:
            self._clear_feature_history()
        self.cloud_mtime_ns = mtime_ns
        cloud_points = sum(len(points) for points in self.cloud_rows_by_frame.values())
        self.get_logger().info(
            f"Loaded {cloud_points} sparse landmark rows from {self.cloud_csv_path} "
            f"across {len(self.cloud_rows_by_frame)} frames"
        )

    def _reload_cloud_source_rows(self, initial: bool) -> None:
        if self.cloud_source_odom_csv_path is None:
            return
        if not self.cloud_source_odom_csv_path.exists():
            if initial:
                self.get_logger().warn(f"Point cloud source odometry CSV not found yet: {self.cloud_source_odom_csv_path}")
            return
        mtime_ns = self.cloud_source_odom_csv_path.stat().st_mtime_ns
        if not initial and mtime_ns == self.cloud_source_mtime_ns:
            return
        self.cloud_source_rows_by_frame = _load_rows_by_frame(self.cloud_source_odom_csv_path)
        if not initial:
            self._clear_feature_history()
        self.cloud_source_mtime_ns = mtime_ns
        self.get_logger().info(
            f"Loaded {len(self.cloud_source_rows_by_frame)} point-cloud source odometry rows "
            f"from {self.cloud_source_odom_csv_path}"
        )

    def _make_odom(self, row: dict[str, str], index: int, stamp) -> Odometry:
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        _fill_pose(
            odom.pose.pose,
            self.rows,
            index,
            self.coordinate_frame,
            self.origin_position,
            self.orientation_source,
            self.yaw_offset_rad,
        )
        return odom

    def _make_pose(self, row: dict[str, str], index: int, stamp) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        _fill_pose(
            pose.pose,
            self.rows,
            index,
            self.coordinate_frame,
            self.origin_position,
            self.orientation_source,
            self.yaw_offset_rad,
        )
        return pose

    def _make_transform(self, row: dict[str, str], index: int, stamp) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.frame_id
        transform.child_frame_id = self.child_frame_id
        position, quaternion = _row_pose_for_index(
            self.rows,
            index,
            self.coordinate_frame,
            self.origin_position,
            self.orientation_source,
            self.yaw_offset_rad,
        )
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        return transform

    def _publish_point_cloud(self, row: dict[str, str], stamp) -> None:
        if not self.cloud_rows_by_frame:
            return
        if self.success_only_cloud and row.get("pose_success") != "1":
            self.point_cloud_pub.publish(
                _make_point_cloud2([], stamp, self.frame_id, self.coordinate_frame, self.origin_position)
            )
            history_points = [point for _, frame_points in self.history_frames for point in frame_points]
            self.history_cloud_pub.publish(
                _make_point_cloud2(
                    history_points,
                    stamp,
                    self.frame_id,
                    self.coordinate_frame,
                    self.origin_position,
                    self.yaw_offset_rad,
                )
            )
            return
        try:
            frame_index = int(row["frame_index"])
        except (KeyError, ValueError):
            return
        if self.point_cloud_display_mode == "world-map":
            features = self._world_map_features_for_frame(frame_index)
        else:
            features = self._feature_points_for_target_pose(frame_index, row)
        points = _feature_points(features)
        cloud = _make_point_cloud2(
            points,
            stamp,
            self.frame_id,
            self.coordinate_frame,
            self.origin_position,
            self.yaw_offset_rad,
        )
        self.point_cloud_pub.publish(cloud)
        self._update_feature_history(frame_index, features)
        history_points = [point for _, frame_points in self.history_frames for point in frame_points]
        self.history_cloud_pub.publish(
            _make_point_cloud2(
                history_points,
                stamp,
                self.frame_id,
                self.coordinate_frame,
                self.origin_position,
                self.yaw_offset_rad,
            )
        )
        should_publish_tracks = self.force_marker_publish or (self.index % self.feature_track_publish_stride == 0)
        if should_publish_tracks:
            self.feature_track_pub.publish(
                _make_feature_track_markers(
                    self.track_history,
                    stamp,
                    self.frame_id,
                    self.coordinate_frame,
                    self.origin_position,
                    self.yaw_offset_rad,
                )
            )
            self.force_marker_publish = False

    def _feature_points_for_target_pose(self, frame_index: int, target_row: dict[str, str]) -> list[FeaturePoint]:
        features = self.cloud_rows_by_frame.get(frame_index, [])
        if not features or not self.cloud_source_rows_by_frame:
            return features
        source_row = self.cloud_source_rows_by_frame.get(frame_index)
        if source_row is None:
            return features
        return _transform_features_between_poses(
            features,
            source_row,
            target_row,
            mode=self.point_cloud_transform_mode,
            min_track_age=self.point_cloud_min_track_age,
            min_depth=self.point_cloud_min_depth,
            max_depth=self.point_cloud_max_depth,
            max_range=self.point_cloud_max_range,
            max_path_distance=self.point_cloud_max_path_distance,
            local_scale=self.point_cloud_local_scale,
        )

    def _world_map_features_for_frame(self, frame_index: int) -> list[FeaturePoint]:
        features = self.cloud_rows_by_frame.get(frame_index, [])
        if not features:
            return features
        source_row = self.cloud_source_rows_by_frame.get(frame_index)
        if source_row is None:
            return [
                feature
                for feature in features
                if feature[4] >= self.point_cloud_min_track_age
            ]
        return _filter_world_features_from_source_camera(
            features,
            source_row,
            min_track_age=self.point_cloud_min_track_age,
            min_depth=self.point_cloud_min_depth,
            max_depth=self.point_cloud_max_depth,
            max_range=self.point_cloud_max_range,
            max_path_distance=self.point_cloud_max_path_distance,
        )

    def _new_track_history(self) -> deque[tuple[float, float, float]]:
        maxlen = self.feature_track_history if self.feature_track_history > 0 else None
        return deque(maxlen=maxlen)

    def _clear_feature_history(self) -> None:
        self.history_frames.clear()
        self.track_history.clear()
        self.track_last_frame.clear()
        self.force_marker_publish = True

    def _update_feature_history(self, frame_index: int, features: list[FeaturePoint]) -> None:
        points = _feature_points(features)
        self.history_frames.append((frame_index, points))
        if self.feature_history_frames > 0:
            first_kept = frame_index - self.feature_history_frames + 1
            while self.history_frames and self.history_frames[0][0] < first_kept:
                self.history_frames.popleft()

        for track_id, x, y, z, _age in features:
            if track_id < 0:
                continue
            self.track_history[track_id].append((x, y, z))
            self.track_last_frame[track_id] = frame_index
        if self.feature_history_frames > 0:
            stale_ids = [
                track_id
                for track_id, last_frame in self.track_last_frame.items()
                if last_frame < first_kept
            ]
            for track_id in stale_ids:
                self.track_history.pop(track_id, None)
                self.track_last_frame.pop(track_id, None)


def _fill_pose(
    pose,
    rows: list[dict[str, str]],
    index: int,
    coordinate_frame: str = "ros",
    origin_position: tuple[float, float, float] | None = None,
    orientation_source: str = "csv",
    yaw_offset_rad: float = 0.0,
) -> None:
    position, quaternion = _row_pose_for_index(
        rows,
        index,
        coordinate_frame,
        origin_position,
        orientation_source,
        yaw_offset_rad,
    )
    pose.position.x = position[0]
    pose.position.y = position[1]
    pose.position.z = position[2]
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]


def _row_pose_for_index(
    rows: list[dict[str, str]],
    index: int,
    coordinate_frame: str,
    origin_position: tuple[float, float, float] | None = None,
    orientation_source: str = "csv",
    yaw_offset_rad: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    row = rows[index]
    position, csv_quaternion = _row_pose(row, coordinate_frame, origin_position, yaw_offset_rad)
    if not _should_use_tangent_orientation(row, orientation_source):
        return position, csv_quaternion
    return position, _tangent_quaternion_for_index(
        rows,
        index,
        coordinate_frame,
        origin_position,
        yaw_offset_rad,
    )


def _row_pose(
    row: dict[str, str],
    coordinate_frame: str,
    origin_position: tuple[float, float, float] | None = None,
    yaw_offset_rad: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    position = _raw_position(row)
    if origin_position is not None:
        position = _sub(position, origin_position)
    quaternion = (
        float(row.get("qx") or 0.0),
        float(row.get("qy") or 0.0),
        float(row.get("qz") or 0.0),
        float(row.get("qw") or 1.0),
    )
    if coordinate_frame in {"opencv", "raw"}:
        return _apply_yaw_offset(position, _normalize_quaternion(quaternion), yaw_offset_rad)
    if coordinate_frame == "flip-y":
        return _apply_yaw_offset(
            (position[0], -position[1], position[2]),
            _normalize_quaternion(quaternion),
            yaw_offset_rad,
        )
    if coordinate_frame != "ros":
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")

    # CSV odometry is in OpenCV optical coordinates: x right, y down, z forward.
    # RViz expects ROS body/map style: x forward, y left, z up.
    ros_position = (position[2], -position[0], -position[1])
    rotation = _quaternion_to_matrix(quaternion)
    converted = _matmul(_matmul(_OPENCV_OPTICAL_TO_ROS, rotation), _transpose(_OPENCV_OPTICAL_TO_ROS))
    return _apply_yaw_offset(ros_position, _matrix_to_quaternion(converted), yaw_offset_rad)


def _apply_yaw_offset(
    position: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    yaw_offset_rad: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    if abs(yaw_offset_rad) <= 1e-12:
        return position, quaternion
    rotation = _yaw_rotation_matrix(yaw_offset_rad)
    rotated_position = _matvec(rotation, position)
    rotated_quaternion = _matrix_to_quaternion(_matmul(rotation, _quaternion_to_matrix(quaternion)))
    return rotated_position, rotated_quaternion


def _apply_yaw_to_position(
    position: tuple[float, float, float],
    yaw_offset_rad: float,
) -> tuple[float, float, float]:
    if abs(yaw_offset_rad) <= 1e-12:
        return position
    return _matvec(_yaw_rotation_matrix(yaw_offset_rad), position)


def _yaw_rotation_matrix(yaw_rad: float) -> tuple[tuple[float, float, float], ...]:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    return (
        (cos_yaw, -sin_yaw, 0.0),
        (sin_yaw, cos_yaw, 0.0),
        (0.0, 0.0, 1.0),
    )


def _should_use_tangent_orientation(row: dict[str, str], orientation_source: str) -> bool:
    if orientation_source == "tangent":
        return True
    if orientation_source == "csv":
        return False
    if orientation_source != "auto":
        raise ValueError(f"Unsupported orientation source: {orientation_source}")
    return not _row_has_quaternion(row)


def _row_has_quaternion(row: dict[str, str]) -> bool:
    return all(row.get(name) not in {None, ""} for name in ("qx", "qy", "qz", "qw"))


def _tangent_quaternion_for_index(
    rows: list[dict[str, str]],
    index: int,
    coordinate_frame: str,
    origin_position: tuple[float, float, float] | None = None,
    yaw_offset_rad: float = 0.0,
) -> tuple[float, float, float, float]:
    if len(rows) < 2:
        return (0.0, 0.0, 0.0, 1.0)
    if index == 0:
        left_index, right_index = 0, 1
    elif index == len(rows) - 1:
        left_index, right_index = index - 1, index
    else:
        left_index, right_index = index - 1, index + 1
    left_position, _ = _row_pose(rows[left_index], coordinate_frame, origin_position, yaw_offset_rad)
    right_position, _ = _row_pose(rows[right_index], coordinate_frame, origin_position, yaw_offset_rad)
    return _direction_quaternion(_sub(right_position, left_position))


def _direction_quaternion(delta: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_axis = _normalize_vector(delta)
    if x_axis is None:
        return (0.0, 0.0, 0.0, 1.0)
    up = (0.0, 0.0, 1.0)
    if abs(_dot(x_axis, up)) > 0.95:
        up = (0.0, 1.0, 0.0)
    y_axis = _normalize_vector(_cross(up, x_axis))
    if y_axis is None:
        return (0.0, 0.0, 0.0, 1.0)
    z_axis = _cross(x_axis, y_axis)
    rotation = (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )
    return _matrix_to_quaternion(rotation)


def _convert_point(
    point: tuple[float, float, float],
    coordinate_frame: str,
    origin_position: tuple[float, float, float] | None = None,
    yaw_offset_rad: float = 0.0,
) -> tuple[float, float, float]:
    if origin_position is not None:
        point = _sub(point, origin_position)
    if coordinate_frame in {"opencv", "raw"}:
        return _apply_yaw_to_position(point, yaw_offset_rad)
    if coordinate_frame == "flip-y":
        return _apply_yaw_to_position((point[0], -point[1], point[2]), yaw_offset_rad)
    if coordinate_frame != "ros":
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    return _apply_yaw_to_position((point[2], -point[0], -point[1]), yaw_offset_rad)


def _make_point_cloud2(
    points: list[tuple[float, float, float]],
    stamp,
    frame_id: str,
    coordinate_frame: str,
    origin_position: tuple[float, float, float] | None = None,
    yaw_offset_rad: float = 0.0,
) -> PointCloud2:
    cloud = PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = frame_id
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 12
    cloud.row_step = cloud.point_step * cloud.width
    cloud.is_dense = True
    cloud.data = b"".join(
        struct.pack("<fff", *_convert_point(point, coordinate_frame, origin_position, yaw_offset_rad))
        for point in points
    )
    return cloud


def _make_feature_track_markers(
    track_history: dict[int, deque[tuple[float, float, float]]],
    stamp,
    frame_id: str,
    coordinate_frame: str,
    origin_position: tuple[float, float, float] | None = None,
    yaw_offset_rad: float = 0.0,
) -> MarkerArray:
    markers = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)
    for marker_id, (track_id, points) in enumerate(sorted(track_history.items())):
        if len(points) < 2:
            continue
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "feature_track_history"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.color.r = 0.1
        marker.color.g = 0.85
        marker.color.b = 1.0
        marker.color.a = 0.55
        marker.pose.orientation.w = 1.0
        marker.points = [
            _make_point(_convert_point(point, coordinate_frame, origin_position, yaw_offset_rad))
            for point in points
        ]
        markers.markers.append(marker)
    return markers


def _make_point(point: tuple[float, float, float]) -> Point:
    msg = Point()
    msg.x = float(point[0])
    msg.y = float(point[1])
    msg.z = float(point[2])
    return msg


_OPENCV_OPTICAL_TO_ROS = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)


def _quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], ...]:
    qx, qy, qz, qw = _normalize_quaternion(quaternion)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _matrix_to_quaternion(rotation: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float, float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2][1] - rotation[1][2]) / scale
        qy = (rotation[0][2] - rotation[2][0]) / scale
        qz = (rotation[1][0] - rotation[0][1]) / scale
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        qw = (rotation[2][1] - rotation[1][2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0][1] + rotation[1][0]) / scale
        qz = (rotation[0][2] + rotation[2][0]) / scale
    elif rotation[1][1] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        qw = (rotation[0][2] - rotation[2][0]) / scale
        qx = (rotation[0][1] + rotation[1][0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1][2] + rotation[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        qw = (rotation[1][0] - rotation[0][1]) / scale
        qx = (rotation[0][2] + rotation[2][0]) / scale
        qy = (rotation[1][2] + rotation[2][1]) / scale
        qz = 0.25 * scale
    return _normalize_quaternion((qx, qy, qz, qw))


def _normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(value / norm for value in quaternion)


def _normalize_vector(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = _norm(vector)
    if norm <= 1e-12:
        return None
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _raw_position(row: dict[str, str]) -> tuple[float, float, float]:
    return (float(row["x"]), float(row["y"]), float(row["z"]))


def _sub(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _add(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _scale(vector: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def _matmul(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def _transpose(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[row][col] for row in range(3)) for col in range(3))


def _matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))


def _transform_features_between_poses(
    features: list[FeaturePoint],
    source_row: dict[str, str],
    target_row: dict[str, str],
    *,
    mode: str,
    min_track_age: int,
    min_depth: float,
    max_depth: float,
    max_range: float,
    max_path_distance: float,
    local_scale: float,
) -> list[FeaturePoint]:
    source_position, source_quaternion = _row_pose(source_row, "raw")
    target_position, target_quaternion = _row_pose(target_row, "raw")
    source_rotation = _quaternion_to_matrix(source_quaternion)
    target_rotation = _quaternion_to_matrix(target_quaternion)
    source_camera_from_world = _transpose(source_rotation)

    transformed: list[FeaturePoint] = []
    for track_id, x, y, z, age in features:
        if age < min_track_age:
            continue
        source_world_point = (x, y, z)
        camera_point = _matvec(source_camera_from_world, _sub(source_world_point, source_position))
        if not _passes_camera_point_filter(camera_point, min_depth, max_depth, max_range):
            continue
        scaled_camera_point = _scale(camera_point, max(0.0, local_scale))
        if mode == "pose":
            target_world_point = _add(target_position, _matvec(target_rotation, scaled_camera_point))
        elif mode == "translate-only":
            target_world_point = _add(target_position, scaled_camera_point)
        else:
            raise ValueError(f"Unsupported point-cloud transform mode: {mode}")
        if max_path_distance > 0.0 and _norm(_sub(target_world_point, target_position)) > max_path_distance:
            continue
        transformed.append((track_id, target_world_point[0], target_world_point[1], target_world_point[2], age))
    return transformed


def _filter_world_features_from_source_camera(
    features: list[FeaturePoint],
    source_row: dict[str, str],
    *,
    min_track_age: int,
    min_depth: float,
    max_depth: float,
    max_range: float,
    max_path_distance: float,
) -> list[FeaturePoint]:
    source_position, source_quaternion = _row_pose(source_row, "raw")
    source_rotation = _quaternion_to_matrix(source_quaternion)
    source_camera_from_world = _transpose(source_rotation)
    filtered: list[FeaturePoint] = []
    for track_id, x, y, z, age in features:
        if age < min_track_age:
            continue
        source_world_point = (x, y, z)
        camera_point = _matvec(source_camera_from_world, _sub(source_world_point, source_position))
        if not _passes_camera_point_filter(camera_point, min_depth, max_depth, max_range):
            continue
        if max_path_distance > 0.0 and _norm(_sub(source_world_point, source_position)) > max_path_distance:
            continue
        filtered.append((track_id, x, y, z, age))
    return filtered


def _passes_camera_point_filter(
    camera_point: tuple[float, float, float],
    min_depth: float,
    max_depth: float,
    max_range: float,
) -> bool:
    x, y, z = camera_point
    if z < min_depth:
        return False
    if max_depth > 0.0 and z > max_depth:
        return False
    if max_range > 0.0 and _norm(camera_point) > max_range:
        return False
    return True


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _load_rows_by_frame(path: Path) -> dict[int, dict[str, str]]:
    rows_by_frame: dict[int, dict[str, str]] = {}
    for row in _load_rows(path):
        try:
            rows_by_frame[int(row["frame_index"])] = row
        except (KeyError, ValueError):
            continue
    return rows_by_frame


def _load_cloud_rows(path: Path) -> dict[int, list[FeaturePoint]]:
    grouped: dict[int, list[FeaturePoint]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                frame_index = int(row["frame_index"])
                track_id = int(row.get("track_id", -1))
                point = (
                    track_id,
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    int(row.get("age", 1) or 1),
                )
            except (KeyError, ValueError):
                continue
            if all(math.isfinite(value) for value in point[1:4]):
                grouped[frame_index].append(point)
    return dict(grouped)


def _feature_points(features: list[FeaturePoint]) -> list[tuple[float, float, float]]:
    return [(x, y, z) for _, x, y, z, _age in features]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish VIO frontend odometry CSV to ROS2/RViz.")
    parser.add_argument("--csv", type=Path, default=Path("outputs/live/latest_odometry.csv"))
    parser.add_argument("--point-cloud-csv", type=Path, default=Path("outputs/live/latest_point_cloud.csv"))
    parser.add_argument(
        "--point-cloud-source-odom-csv",
        type=Path,
        help=(
            "Odometry CSV whose poses generated the point-cloud CSV. When set, feature world points are "
            "converted source-world -> source-camera -> current CSV world so the cloud follows the displayed path."
        ),
    )
    parser.add_argument(
        "--point-cloud-display-mode",
        choices=("camera-local-history", "world-map"),
        default="camera-local-history",
        help=(
            "camera-local-history re-attaches feature observations to the current displayed pose for frontend debugging. "
            "world-map publishes filtered feature world points without re-attaching them to the current camera pose."
        ),
    )
    parser.add_argument(
        "--point-cloud-transform-mode",
        choices=("translate-only", "pose"),
        default="translate-only",
        help=(
            "translate-only: attach source camera-local feature points to the target position without using target rotation. "
            "pose: rotate source camera-local points by the target quaternion."
        ),
    )
    parser.add_argument("--point-cloud-min-track-age", type=int, default=2)
    parser.add_argument("--point-cloud-min-depth", type=float, default=0.25)
    parser.add_argument("--point-cloud-max-depth", type=float, default=3.0)
    parser.add_argument("--point-cloud-max-range", type=float, default=3.0)
    parser.add_argument("--point-cloud-max-path-distance", type=float, default=3.0)
    parser.add_argument(
        "--point-cloud-local-scale",
        type=float,
        default=1.0,
        help="Scale camera-local feature offsets for RViz display only; odometry and evaluation stay unchanged.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=60.0,
        help="Odometry rows published per second. 60 Hz replays 3100 frame poses in about 52 seconds.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--zero-start", action="store_true", help="Subtract the first CSV pose from odometry, path, and cloud points.")
    parser.add_argument("--success-only-path", action="store_true")
    parser.add_argument(
        "--success-only-cloud",
        action="store_true",
        help="Do not add feature cloud/history samples from rows where pose_success is false.",
    )
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--child-frame-id", default="camera_link")
    parser.add_argument(
        "--coordinate-frame",
        choices=("ros", "opencv", "raw", "flip-y"),
        default="ros",
        help=(
            "ros: convert CSV OpenCV optical poses to RViz/map coordinates. "
            "opencv/raw: publish CSV coordinates directly with no axis conversion. "
            "flip-y: publish direct coordinates with only Y sign inverted."
        ),
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=0.0,
        help=(
            "Rotate the displayed odometry, path, point cloud, and feature tracks around RViz/map Z after "
            "zero-start alignment. This is a display/calibration offset only; it does not feed DVL into VIO."
        ),
    )
    parser.add_argument(
        "--orientation-source",
        choices=("csv", "tangent", "auto"),
        default="csv",
        help=(
            "csv: publish quaternion stored in the CSV. "
            "tangent: derive orientation from trajectory direction. "
            "auto: use CSV quaternion when present, otherwise derive trajectory direction."
        ),
    )
    parser.add_argument("--odom-topic", default="/visual_odom")
    parser.add_argument("--path-topic", default="/local_path")
    parser.add_argument("--vins-odom-topic", default="/vins_estimator/odometry")
    parser.add_argument("--vins-path-topic", default="/vins_estimator/path")
    parser.add_argument("--point-cloud-topic", default="/vins_estimator/point_cloud")
    parser.add_argument("--history-point-cloud-topic", default="/vins_estimator/feature_history_cloud")
    parser.add_argument("--feature-track-topic", default="/vins_estimator/feature_tracks")
    parser.add_argument(
        "--feature-history-frames",
        type=int,
        default=0,
        help="Number of frames kept in the accumulated feature point cloud. 0 keeps the full playback pass.",
    )
    parser.add_argument(
        "--feature-track-history",
        type=int,
        default=80,
        help="Number of recent 3D positions kept per feature track line. 0 keeps the full track.",
    )
    parser.add_argument(
        "--feature-track-publish-stride",
        type=int,
        default=1,
        help="Publish feature track line markers every N odometry frames while still publishing cloud points every frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = OdometryCsvPublisher(args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
