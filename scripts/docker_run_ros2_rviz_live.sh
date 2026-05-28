#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-uuv-vins-ros2:humble}"
CONTAINER="${CONTAINER:-uuv-vins-ros2-live}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
PLAY_SECONDS="${PLAY_SECONDS:-30}"
RUN_RVIZ="${RUN_RVIZ:-1}"
RVIZ_BACKEND="${RVIZ_BACKEND:-}"
DETACH="${DETACH:-0}"
LOG_DIR="${LOG_DIR:-/workspace/under_water_image_match/outputs/docker_live}"
NOVNC_HOST_PORT="${NOVNC_HOST_PORT:-6080}"
EXPOSE_VNC_PORT="${EXPOSE_VNC_PORT:-0}"
VNC_HOST_PORT="${VNC_HOST_PORT:-5900}"
DOCKER_NETWORK_MODE="${DOCKER_NETWORK_MODE:-}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
ROS_TCP_BRIDGE_PORT="${ROS_TCP_BRIDGE_PORT:-}"

if ! docker info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Docker daemon is not running.
Open Docker Desktop first, wait until Docker is running, then retry.
EOF
  exit 2
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  "${ROOT}/scripts/docker_build_ros2_humble.sh"
fi

display_args=()
network_args=()
device_args=()
run_mode_args=()
remove_args=()
port_args=()
env_args=()

if [[ "$(uname -s)" == "Darwin" ]]; then
  RVIZ_BACKEND="${RVIZ_BACKEND:-vnc}"
  if [[ -z "${DISPLAY:-}" || "${DISPLAY}" == /var/run/* || "${DISPLAY}" == :* ]]; then
    DISPLAY="host.docker.internal:0"
  fi
  if command -v xhost >/dev/null 2>&1; then
    DISPLAY=:0 xhost + >/dev/null 2>&1 || true
    DISPLAY=:0 xhost + 127.0.0.1 >/dev/null 2>&1 || true
    DISPLAY=:0 xhost + localhost >/dev/null 2>&1 || true
    DISPLAY=:0 xhost + "$(hostname)" >/dev/null 2>&1 || true
  fi
else
  RVIZ_BACKEND="${RVIZ_BACKEND:-x11}"
  DISPLAY="${DISPLAY:-:0}"
  network_args+=(--net=host)
  display_args+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  if [[ -d /dev/dri ]]; then
    device_args+=(--device /dev/dri)
  fi
fi

if [[ -n "${DOCKER_NETWORK_MODE}" ]]; then
  network_args=(--network "${DOCKER_NETWORK_MODE}")
fi

if [[ "${RVIZ_BACKEND}" == "vnc" && "${DOCKER_NETWORK_MODE}" != "host" ]]; then
  port_args+=(-p "${NOVNC_HOST_PORT}:6080")
  if [[ "${EXPOSE_VNC_PORT}" == "1" ]]; then
    port_args+=(-p "${VNC_HOST_PORT}:5900")
  fi
fi
if [[ -n "${ROS_TCP_BRIDGE_PORT}" ]]; then
  port_args+=(-p "${ROS_TCP_BRIDGE_PORT}:${ROS_TCP_BRIDGE_PORT}")
fi

if [[ -n "${ROS_DISCOVERY_SERVER:-}" ]]; then
  env_args+=(-e "ROS_DISCOVERY_SERVER=${ROS_DISCOVERY_SERVER}")
fi
if [[ -n "${CYCLONEDDS_URI:-}" ]]; then
  env_args+=(-e "CYCLONEDDS_URI=${CYCLONEDDS_URI}")
fi
if [[ -n "${FASTDDS_DEFAULT_PROFILES_FILE:-}" ]]; then
  env_args+=(-e "FASTDDS_DEFAULT_PROFILES_FILE=${FASTDDS_DEFAULT_PROFILES_FILE}")
fi

if [[ "${DETACH}" == "1" ]]; then
  run_mode_args+=(-d)
elif [[ -t 0 && -t 1 ]]; then
  run_mode_args+=(-it)
else
  run_mode_args+=(-i)
fi

if [[ "${DETACH}" != "1" ]]; then
  remove_args+=(--rm)
fi

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

docker run \
  "${remove_args[@]+"${remove_args[@]}"}" \
  "${run_mode_args[@]+"${run_mode_args[@]}"}" \
  --name "${CONTAINER}" \
  "${network_args[@]+"${network_args[@]}"}" \
  "${port_args[@]+"${port_args[@]}"}" \
  "${display_args[@]+"${display_args[@]}"}" \
  "${device_args[@]+"${device_args[@]}"}" \
  -e DISPLAY="${DISPLAY}" \
  -e CONFIG="${CONFIG:-}" \
  -e RUN_ONCE="${RUN_ONCE:-0}" \
  -e QT_X11_NO_MITSHM=1 \
  -e LIBGL_ALWAYS_INDIRECT=1 \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -e ROS_LOCALHOST_ONLY=0 \
  "${env_args[@]+"${env_args[@]}"}" \
  -e PLAY_SECONDS="${PLAY_SECONDS}" \
  -e PLAY_RATE="${PLAY_RATE:-1.0}" \
  -e RESTAMP_NOW="${RESTAMP_NOW:-1}" \
  -e ROSBAG_INPUT_DIR="${ROSBAG_INPUT_DIR:-/workspace/under_water_image_match/data/rosbag_active/localization bag}" \
  -e IMAGE_SOURCE_TYPE="${IMAGE_SOURCE_TYPE:-mp4}" \
  -e STAMP_TIME_SOURCE="${STAMP_TIME_SOURCE:-}" \
  -e STEREO_SYNC_TOLERANCE_MS="${STEREO_SYNC_TOLERANCE_MS:-3}" \
  -e STEREO_SCHEDULE_SOURCE_CSV="${STEREO_SCHEDULE_SOURCE_CSV:-}" \
  -e SWAP_STEREO_MP4="${SWAP_STEREO_MP4:-0}" \
  -e INFRA1_MP4="${INFRA1_MP4:-}" \
  -e INFRA2_MP4="${INFRA2_MP4:-}" \
  -e LEFT_MP4="${LEFT_MP4:-}" \
  -e RIGHT_MP4="${RIGHT_MP4:-}" \
  -e DVL_CSV="${DVL_CSV:-}" \
  -e DVL_COORDINATE_FRAME="${DVL_COORDINATE_FRAME:-flip-y}" \
  -e DVL_REFERENCE_ALIGN_MODE="${DVL_REFERENCE_ALIGN_MODE:-none}" \
  -e STAMP_DB="${STAMP_DB:-}" \
  -e BAG_DIR="${BAG_DIR:-}" \
  -e LOCALIZATION_REFERENCE_CSV="${LOCALIZATION_REFERENCE_CSV:-}" \
  -e IMU_RESAMPLE_HZ="${IMU_RESAMPLE_HZ:-0}" \
  -e IMU_SOURCE_TOPIC="${IMU_SOURCE_TOPIC-/mavros/imu/data}" \
  -e IMU_SOURCE_TYPE="${IMU_SOURCE_TYPE:-sensor_msgs/msg/Imu}" \
  -e IMU_OUTPUT_TOPIC="${IMU_OUTPUT_TOPIC:-/vins/imu/data}" \
  -e IMU_FRAME_ID="${IMU_FRAME_ID:-fcu_link}" \
  -e IMU_FRAME_TRANSFORM="${IMU_FRAME_TRANSFORM:-identity}" \
  -e UNIFORM_STEREO_TIMESTAMPS="${UNIFORM_STEREO_TIMESTAMPS:-auto}" \
  -e PUBLISH_PIXHAWK_CAMERA_TF="${PUBLISH_PIXHAWK_CAMERA_TF:-1}" \
  -e RUN_EKF_LOCALIZATION="${RUN_EKF_LOCALIZATION:-0}" \
  -e RUN_BAG_LOCALIZATION_PATH="${RUN_BAG_LOCALIZATION_PATH:-1}" \
  -e RUN_TF_LOCALIZED_PATH="${RUN_TF_LOCALIZED_PATH:-0}" \
  -e SYNC_REPLAY_PATHS="${SYNC_REPLAY_PATHS:-1}" \
  -e DVL_REPLAY_HZ="${DVL_REPLAY_HZ:-10.0}" \
  -e LOCALIZATION_ALIGN_MODE="${LOCALIZATION_ALIGN_MODE:-start_yaw}" \
  -e LOCALIZATION_ALIGN_YAW_DEG="${LOCALIZATION_ALIGN_YAW_DEG:--90.0}" \
  -e VINS_DRAIN_TIMEOUT_SEC="${VINS_DRAIN_TIMEOUT_SEC:-25}" \
  -e VINS_DRAIN_IDLE_SEC="${VINS_DRAIN_IDLE_SEC:-5}" \
  -e EKF_CONFIG="${EKF_CONFIG:-/workspace/under_water_image_match/config/ekf_vins_pixhawk.yaml}" \
  -e BUILD_ON_START="${BUILD_ON_START:-1}" \
  -e RUN_RVIZ="${RUN_RVIZ}" \
  -e RVIZ_BACKEND="${RVIZ_BACKEND}" \
  -e LOG_DIR="${LOG_DIR}" \
  -v "${ROOT}:/workspace/under_water_image_match:rw" \
  -w /workspace/under_water_image_match \
  "${IMAGE}" \
  bash /workspace/under_water_image_match/scripts/docker_ros2_live_rviz.sh
