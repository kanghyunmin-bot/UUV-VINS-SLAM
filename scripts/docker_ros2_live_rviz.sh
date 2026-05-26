#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/under_water_image_match}"
PLAY_SECONDS="${PLAY_SECONDS:-30}"
PLAY_RATE="${PLAY_RATE:-1.0}"
RESTAMP_NOW="${RESTAMP_NOW:-1}"
UNIFORM_STEREO_TIMESTAMPS="${UNIFORM_STEREO_TIMESTAMPS:-0}"
IMAGE_SOURCE_TYPE="${IMAGE_SOURCE_TYPE:-mp4}"
RUN_ONCE="${RUN_ONCE:-0}"
IMU_RESAMPLE_HZ="${IMU_RESAMPLE_HZ:-100}"
LOG_DIR="${LOG_DIR:-/tmp/uuv_ros2_live}"
CONFIG="${CONFIG:-${ROOT}/VINS-Fusion-ROS2/config/realsense_d435i/underwater_realsense_stereo_mavros_imu_config.yaml}"
RVIZ_CONFIG="${RVIZ_CONFIG:-${ROOT}/rviz/uuv_vins_minimal.rviz}"
EKF_CONFIG="${EKF_CONFIG:-${ROOT}/config/ekf_vins_pixhawk.yaml}"
ROSBAG_INPUT_DIR="${ROSBAG_INPUT_DIR:-${ROOT}/data/rosbag_active/localization bag}"
DVL_CSV="${DVL_CSV:-}"
DVL_COORDINATE_FRAME="${DVL_COORDINATE_FRAME:-flip-y}"
DVL_REFERENCE_ALIGN_MODE="${DVL_REFERENCE_ALIGN_MODE:-none}"
INFRA1_MP4="${INFRA1_MP4:-${ROSBAG_INPUT_DIR}/infra1_image_rect_raw.mp4}"
INFRA2_MP4="${INFRA2_MP4:-${ROSBAG_INPUT_DIR}/infra2_image_rect_raw.mp4}"
BAG_DIR="${BAG_DIR:-${ROSBAG_INPUT_DIR}}"
STAMP_DB="${STAMP_DB:-${ROSBAG_INPUT_DIR}/stamp_bag.db3}"
BUNDLE_MANIFEST="${BUNDLE_MANIFEST:-${ROSBAG_INPUT_DIR}/bundle_manifest.json}"
MANIFEST_START_OFFSET_SEC=""
MANIFEST_DVL_CSV=""
MANIFEST_TIME_SOURCE=""
MANIFEST_STEREO_SCHEDULE_CSV=""
MANIFEST_STEREO_SCHEDULE_SWAPPED_CSV=""
MANIFEST_LOCALIZATION_ODOMETRY_CSV=""
if [[ -f "${BUNDLE_MANIFEST}" ]]; then
  MANIFEST_START_OFFSET_SEC="$(python3 - "${BUNDLE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data.get("start_offset_sec", ""))
PY
)"
  MANIFEST_DVL_CSV="$(python3 - "${BUNDLE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print((data.get("outputs") or {}).get("dvl_reference_csv", ""))
PY
)"
  MANIFEST_TIME_SOURCE="$(python3 - "${BUNDLE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data.get("time_source", ""))
PY
)"
  MANIFEST_STEREO_SCHEDULE_CSV="$(python3 - "${BUNDLE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print((data.get("outputs") or {}).get("stereo_schedule_csv", ""))
PY
)"
  MANIFEST_STEREO_SCHEDULE_SWAPPED_CSV="$(python3 - "${BUNDLE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print((data.get("outputs") or {}).get("stereo_schedule_swapped_csv", ""))
PY
)"
  MANIFEST_LOCALIZATION_ODOMETRY_CSV="$(python3 - "${BUNDLE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print((data.get("outputs") or {}).get("localization_odometry_csv", ""))
PY
)"
fi
if [[ -z "${DVL_CSV}" ]]; then
  if [[ -n "${MANIFEST_DVL_CSV}" && -f "${ROSBAG_INPUT_DIR}/${MANIFEST_DVL_CSV}" ]]; then
    DVL_CSV="${ROSBAG_INPUT_DIR}/${MANIFEST_DVL_CSV}"
  else
    DVL_CSV="${ROSBAG_INPUT_DIR}/dvl_reference_0_30s.csv"
  fi
fi
LOCALIZATION_REFERENCE_CSV="${LOCALIZATION_REFERENCE_CSV:-}"
if [[ -z "${LOCALIZATION_REFERENCE_CSV}" ]]; then
  if [[ -n "${MANIFEST_LOCALIZATION_ODOMETRY_CSV}" && -f "${ROSBAG_INPUT_DIR}/${MANIFEST_LOCALIZATION_ODOMETRY_CSV}" ]]; then
    LOCALIZATION_REFERENCE_CSV="${ROSBAG_INPUT_DIR}/${MANIFEST_LOCALIZATION_ODOMETRY_CSV}"
  else
    LOCALIZATION_REFERENCE_CSV="$(find "${ROSBAG_INPUT_DIR}" -maxdepth 1 -name 'localization_odometry_*.csv' -print -quit 2>/dev/null || true)"
  fi
fi
PUBLISH_START_OFFSET_SEC="${PUBLISH_START_OFFSET_SEC:-${MANIFEST_START_OFFSET_SEC:-0}}"
VIDEO_START_INDEX_OVERRIDE="${VIDEO_START_INDEX_OVERRIDE:-}"
STAMP_TIME_SOURCE="${STAMP_TIME_SOURCE:-${MANIFEST_TIME_SOURCE:-header}}"
STEREO_SYNC_TOLERANCE_MS="${STEREO_SYNC_TOLERANCE_MS:-3}"
if [[ -z "${VIDEO_START_INDEX_OVERRIDE}" && -n "${MANIFEST_START_OFFSET_SEC}" ]]; then
  if python3 - "${MANIFEST_START_OFFSET_SEC}" <<'PY'
import sys
raise SystemExit(0 if abs(float(sys.argv[1])) > 1e-9 else 1)
PY
  then
    VIDEO_START_INDEX_OVERRIDE="0"
  fi
fi
DEFAULT_LEFT_MP4="${INFRA1_MP4}"
DEFAULT_RIGHT_MP4="${INFRA2_MP4}"
DEFAULT_LEFT_STAMP_TOPIC="/camera/camera/infra1/image_rect_raw"
DEFAULT_RIGHT_STAMP_TOPIC="/camera/camera/infra2/image_rect_raw"
SWAP_STEREO_MP4="${SWAP_STEREO_MP4:-0}"
if [[ "${SWAP_STEREO_MP4}" == "1" ]]; then
  DEFAULT_LEFT_MP4="${INFRA2_MP4}"
  DEFAULT_RIGHT_MP4="${INFRA1_MP4}"
  DEFAULT_LEFT_STAMP_TOPIC="/camera/camera/infra2/image_rect_raw"
  DEFAULT_RIGHT_STAMP_TOPIC="/camera/camera/infra1/image_rect_raw"
fi
STEREO_SCHEDULE_SOURCE_CSV="${STEREO_SCHEDULE_SOURCE_CSV:-}"
if [[ -z "${STEREO_SCHEDULE_SOURCE_CSV}" ]]; then
  if [[ "${SWAP_STEREO_MP4}" == "1" && -n "${MANIFEST_STEREO_SCHEDULE_SWAPPED_CSV}" && -f "${ROSBAG_INPUT_DIR}/${MANIFEST_STEREO_SCHEDULE_SWAPPED_CSV}" ]]; then
    STEREO_SCHEDULE_SOURCE_CSV="${ROSBAG_INPUT_DIR}/${MANIFEST_STEREO_SCHEDULE_SWAPPED_CSV}"
  elif [[ -n "${MANIFEST_STEREO_SCHEDULE_CSV}" && -f "${ROSBAG_INPUT_DIR}/${MANIFEST_STEREO_SCHEDULE_CSV}" ]]; then
    STEREO_SCHEDULE_SOURCE_CSV="${ROSBAG_INPUT_DIR}/${MANIFEST_STEREO_SCHEDULE_CSV}"
  fi
fi
if [[ -n "${STEREO_SCHEDULE_SOURCE_CSV}" ]]; then
  PUBLISH_START_OFFSET_SEC="0"
  VIDEO_START_INDEX_OVERRIDE="${VIDEO_START_INDEX_OVERRIDE:-0}"
fi
LEFT_MP4="${LEFT_MP4:-${DEFAULT_LEFT_MP4}}"
RIGHT_MP4="${RIGHT_MP4:-${DEFAULT_RIGHT_MP4}}"
LEFT_STAMP_TOPIC="${LEFT_STAMP_TOPIC:-${DEFAULT_LEFT_STAMP_TOPIC}}"
RIGHT_STAMP_TOPIC="${RIGHT_STAMP_TOPIC:-${DEFAULT_RIGHT_STAMP_TOPIC}}"
IMU_SOURCE_TOPIC="${IMU_SOURCE_TOPIC-/mavros/imu/data}"
IMU_SOURCE_TYPE="${IMU_SOURCE_TYPE:-sensor_msgs/msg/Imu}"
IMU_OUTPUT_TOPIC="${IMU_OUTPUT_TOPIC:-/vins/imu/data}"
EKF_CLOCK_TOPIC="${EKF_CLOCK_TOPIC:-/odometry}"
EKF_CLOCK_MSG_TYPE="${EKF_CLOCK_MSG_TYPE:-nav_msgs.msg.Odometry}"
EKF_CLOCKED_ODOM_TOPIC="${EKF_CLOCKED_ODOM_TOPIC:-/vins/odometry_clocked}"
IMU_FRAME_ID="${IMU_FRAME_ID:-fcu_link}"
IMU_FRAME_TRANSFORM="${IMU_FRAME_TRANSFORM:-identity}"
PUBLISH_PIXHAWK_CAMERA_TF="${PUBLISH_PIXHAWK_CAMERA_TF:-1}"
RUN_EKF_LOCALIZATION="${RUN_EKF_LOCALIZATION:-0}"
RUN_BAG_LOCALIZATION_PATH="${RUN_BAG_LOCALIZATION_PATH:-1}"
RUN_TF_LOCALIZED_PATH="${RUN_TF_LOCALIZED_PATH:-0}"
SYNC_REPLAY_PATHS="${SYNC_REPLAY_PATHS:-1}"
DVL_REPLAY_HZ="${DVL_REPLAY_HZ:-10.0}"
LOCALIZATION_ALIGN_MODE="${LOCALIZATION_ALIGN_MODE:-none}"
LOCALIZATION_ALIGN_YAW_DEG="${LOCALIZATION_ALIGN_YAW_DEG:-0.0}"
FCU_TO_CAM0_X="${FCU_TO_CAM0_X:--0.03103}"
FCU_TO_CAM0_Y="${FCU_TO_CAM0_Y:--0.023425138206636054}"
FCU_TO_CAM0_Z="${FCU_TO_CAM0_Z:--0.06254}"
FCU_TO_CAM1_X="${FCU_TO_CAM1_X:--0.03103}"
FCU_TO_CAM1_Y="${FCU_TO_CAM1_Y:--0.07346077639431987}"
FCU_TO_CAM1_Z="${FCU_TO_CAM1_Z:--0.06254}"
FCU_TO_CAMERA_QX="${FCU_TO_CAMERA_QX:--0.5}"
FCU_TO_CAMERA_QY="${FCU_TO_CAMERA_QY:-0.5}"
FCU_TO_CAMERA_QZ="${FCU_TO_CAMERA_QZ:--0.5}"
FCU_TO_CAMERA_QW="${FCU_TO_CAMERA_QW:-0.5}"
BUILD_ON_START="${BUILD_ON_START:-1}"
RUN_RVIZ="${RUN_RVIZ:-1}"
RVIZ_BACKEND="${RVIZ_BACKEND:-x11}"
DOCKER_BUILD_BASE="${DOCKER_BUILD_BASE:-build_docker}"
DOCKER_INSTALL_BASE="${DOCKER_INSTALL_BASE:-install_docker}"
DOCKER_LOG_BASE="${DOCKER_LOG_BASE:-log_docker}"

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export VINS_OUTPUT_PATH="${VINS_OUTPUT_PATH:-${ROOT}/outputs/ros2_vins_fusion_live}"

mkdir -p "${LOG_DIR}" "${ROOT}/outputs/ros2_vins_fusion_live"
case "${IMAGE_SOURCE_TYPE}" in
  mp4|bag) ;;
  *)
    echo "Unsupported IMAGE_SOURCE_TYPE=${IMAGE_SOURCE_TYPE}; expected mp4 or bag" >&2
    exit 2
    ;;
esac

set +u
source /opt/ros/humble/setup.bash
set -u

VINS_INSTALL_ROOT="${ROOT}/VINS-Fusion-ROS2/${DOCKER_INSTALL_BASE}"
VINS_NODE="${VINS_INSTALL_ROOT}/vins/lib/vins/vins_node"

if [[ "${BUILD_ON_START}" == "1" || ! -f "${VINS_NODE}" ]]; then
  cd "${ROOT}/VINS-Fusion-ROS2"
  colcon --log-base "${DOCKER_LOG_BASE}" build \
    --packages-select camera_models vins \
    --build-base "${DOCKER_BUILD_BASE}" \
    --install-base "${DOCKER_INSTALL_BASE}" \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    > "${LOG_DIR}/colcon_build.log" 2>&1
fi

set +u
source "${VINS_INSTALL_ROOT}/setup.bash"
set -u

dvl_pid=""
tf_path_pid=""
ekf_pid=""
filtered_path_pid=""
clock_pid=""
bag_localization_pid=""
loop_pid=""
rviz_pid=""
vins_pid=""
mp4_pid=""
xvfb_pid=""
fluxbox_pid=""
x11vnc_pid=""
novnc_pid=""
static_tf_pids=()

cleanup() {
  for pid in "${static_tf_pids[@]}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  for pid in "${rviz_pid}" "${novnc_pid}" "${x11vnc_pid}" "${fluxbox_pid}" "${xvfb_pid}" "${loop_pid}" "${bag_localization_pid}" "${filtered_path_pid}" "${ekf_pid}" "${clock_pid}" "${tf_path_pid}" "${dvl_pid}" "${vins_pid}" "${mp4_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

stop_child() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then
    return
  fi
  kill "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return
    fi
    sleep 0.1
  done
  kill -9 "${pid}" >/dev/null 2>&1 || true
}

wait_for_vins_drain() {
  local vins_pid_arg="${1:-}"
  local vio_file="${ROOT}/outputs/ros2_vins_fusion_live/vio.csv"
  local timeout_sec="${VINS_DRAIN_TIMEOUT_SEC:-25}"
  local idle_sec="${VINS_DRAIN_IDLE_SEC:-5}"
  local start_sec now_sec last_change_sec size last_size

  if [[ -z "${vins_pid_arg}" ]]; then
    return
  fi

  start_sec="$(date +%s)"
  last_change_sec="${start_sec}"
  last_size="-1"
  echo "waiting for VINS output drain: timeout=${timeout_sec}s idle=${idle_sec}s"

  while kill -0 "${vins_pid_arg}" >/dev/null 2>&1; do
    now_sec="$(date +%s)"
    if [[ -f "${vio_file}" ]]; then
      size="$(wc -c < "${vio_file}" 2>/dev/null || echo 0)"
    else
      size="0"
    fi

    if [[ "${size}" != "${last_size}" ]]; then
      last_size="${size}"
      last_change_sec="${now_sec}"
    elif (( size > 0 && now_sec - last_change_sec >= idle_sec )); then
      echo "VINS output drain idle for ${idle_sec}s"
      break
    fi

    if (( now_sec - start_sec >= timeout_sec )); then
      echo "VINS output drain timeout after ${timeout_sec}s"
      break
    fi

    sleep 0.5
  done
}

stop_orphan_live_nodes() {
  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -f "${VINS_NODE} ${CONFIG}" >/dev/null 2>&1 || true
    pkill -TERM -f "ros2_publish_stereo_mp4.py" >/dev/null 2>&1 || true
    sleep 0.2
    pkill -9 -f "${VINS_NODE} ${CONFIG}" >/dev/null 2>&1 || true
    pkill -9 -f "ros2_publish_stereo_mp4.py" >/dev/null 2>&1 || true
  fi
}

dvl_align_args=()
if [[ "${DVL_REFERENCE_ALIGN_MODE}" != "none" && -n "${LOCALIZATION_REFERENCE_CSV}" && -f "${LOCALIZATION_REFERENCE_CSV}" ]]; then
  dvl_align_args+=(
    --align-target-csv "${LOCALIZATION_REFERENCE_CSV}"
    --align-target-coordinate-frame raw
    --align-mode "${DVL_REFERENCE_ALIGN_MODE}"
  )
fi

start_dvl_reference_path() {
  local mode_label="${1:-static}"
  local progressive_args=()
  if [[ "${mode_label}" == "progressive" ]]; then
    progressive_args+=(
      --progressive
      --playback-rate "${PLAY_RATE}"
      --max-duration-sec "${PLAY_SECONDS}"
      --publish-empty-on-start
      --reset-time-zero
    )
  fi
  python3 "${ROOT}/scripts/ros2_publish_static_path.py" \
    --csv "${DVL_CSV}" \
    --path-topic /dvl_reference_path \
    --marker-topic /dvl_reference_line_marker \
    --direction-marker-topic /dvl_reference_direction_markers \
    --frame-id world \
    --coordinate-frame "${DVL_COORDINATE_FRAME}" \
    "${dvl_align_args[@]}" \
    --rate "${DVL_REPLAY_HZ}" \
    "${progressive_args[@]}" \
    --line-width 0.055 \
    --direction-arrow-stride 28 \
    --direction-arrow-length 0.12 \
    --direction-arrow-shaft-diameter 0.018 \
    --direction-arrow-head-diameter 0.055 \
    --direction-arrow-head-length 0.065 \
    > "${LOG_DIR}/dvl_reference.log" 2>&1 &
  dvl_pid=$!
}

if [[ "${SYNC_REPLAY_PATHS}" != "1" ]]; then
  start_dvl_reference_path static
fi

if [[ "${RUN_TF_LOCALIZED_PATH}" == "1" ]]; then
  python3 "${ROOT}/scripts/ros2_publish_tf_path.py" \
    --tf-topic /tf \
    --parent-frame world \
    --child-frame fcu_link \
    --path-topic /tf_localized_path \
    --odom-topic /tf_localized_odometry \
    --min-step-m 0.02 \
    > "${LOG_DIR}/tf_localized_path.log" 2>&1 &
  tf_path_pid=$!
fi

if [[ "${RUN_EKF_LOCALIZATION}" == "1" ]]; then
python3 "${ROOT}/scripts/ros2_clocked_odom_relay.py" \
  --input-topic "${EKF_CLOCK_TOPIC}" \
  --output-topic "${EKF_CLOCKED_ODOM_TOPIC}" \
  --clock-topic /clock \
  --allow-backward-jump \
  > "${LOG_DIR}/clocked_odom_relay.log" 2>&1 &
clock_pid=$!

  ros2 run robot_localization ekf_node \
    --ros-args \
    --params-file "${EKF_CONFIG}" \
    > "${LOG_DIR}/ekf_filter_node.log" 2>&1 &
  ekf_pid=$!

  python3 "${ROOT}/scripts/ros2_publish_odom_path.py" \
    --odom-topic /odometry/filtered \
    --path-topic /localization/path \
    --frame-id world \
    --min-step-m 0.02 \
    > "${LOG_DIR}/filtered_path.log" 2>&1 &
  filtered_path_pid=$!
fi

start_static_tf() {
  local name="$1"
  local parent_frame="$2"
  local child_frame="$3"
  local x="$4"
  local y="$5"
  local z="$6"
  local qx="$7"
  local qy="$8"
  local qz="$9"
  local qw="${10}"
  ros2 run tf2_ros static_transform_publisher \
    --x "${x}" --y "${y}" --z "${z}" \
    --qx "${qx}" --qy "${qy}" --qz "${qz}" --qw "${qw}" \
    --frame-id "${parent_frame}" \
    --child-frame-id "${child_frame}" \
    > "${LOG_DIR}/${name}.log" 2>&1 &
  static_tf_pids+=("$!")
}

if [[ "${PUBLISH_PIXHAWK_CAMERA_TF}" == "1" ]]; then
  start_static_tf "tf_fcu_to_camera_infra1" \
    "fcu_link" "camera_infra1_optical_frame" \
    "${FCU_TO_CAM0_X}" "${FCU_TO_CAM0_Y}" "${FCU_TO_CAM0_Z}" \
    "${FCU_TO_CAMERA_QX}" "${FCU_TO_CAMERA_QY}" "${FCU_TO_CAMERA_QZ}" "${FCU_TO_CAMERA_QW}"
  start_static_tf "tf_fcu_to_camera_infra2" \
    "fcu_link" "camera_infra2_optical_frame" \
    "${FCU_TO_CAM1_X}" "${FCU_TO_CAM1_Y}" "${FCU_TO_CAM1_Z}" \
    "${FCU_TO_CAMERA_QX}" "${FCU_TO_CAMERA_QY}" "${FCU_TO_CAMERA_QZ}" "${FCU_TO_CAMERA_QW}"
  start_static_tf "tf_fcu_to_imu_link" \
    "fcu_link" "imu_link" \
    "0.0" "0.0" "0.0" \
    "0.0" "0.0" "0.0" "1.0"
fi

SCHEDULE_DIR="${LOG_DIR}/schedule"
publish_window_args=(--start-offset-sec "${PUBLISH_START_OFFSET_SEC}")
stamp_match_args=(--time-source "${STAMP_TIME_SOURCE}" --sync-tolerance-ms "${STEREO_SYNC_TOLERANCE_MS}")
stereo_timing_args=()
if [[ "${UNIFORM_STEREO_TIMESTAMPS}" == "1" ]]; then
  stereo_timing_args+=(--uniform-stereo-timestamps)
fi
if [[ -n "${STEREO_SCHEDULE_SOURCE_CSV}" ]]; then
  stamp_match_args+=(--stereo-schedule-csv "${STEREO_SCHEDULE_SOURCE_CSV}")
fi
if [[ -n "${VIDEO_START_INDEX_OVERRIDE}" ]]; then
  publish_window_args+=(--video-start-index-override "${VIDEO_START_INDEX_OVERRIDE}")
fi
rm -rf "${SCHEDULE_DIR}"
schedule_imu_args=(
  --imu-source-type "${IMU_SOURCE_TYPE}"
  --imu-start-margin-sec 0.03
  --imu-end-margin-sec 0.03
  --publish-imu-from-stamp-bag
  --imu-output-topic "${IMU_OUTPUT_TOPIC}"
  --imu-frame-id "${IMU_FRAME_ID}"
  --imu-frame-transform "${IMU_FRAME_TRANSFORM}"
  --imu-max-delta-sec 0.025
  --imu-resample-hz "${IMU_RESAMPLE_HZ}"
  --imu-resample-max-gap-sec 1.0
  --imu-ahead-sec "${MP4_IMU_AHEAD_SEC:-0.12}"
)
if [[ -n "${IMU_SOURCE_TOPIC}" ]]; then
  schedule_imu_args+=(--imu-source-topic "${IMU_SOURCE_TOPIC}")
else
  schedule_imu_args+=(
    --gyro-stamp-topic /camera/camera/gyro/sample
    --accel-stamp-topic /camera/camera/accel/sample
  )
fi

python3 "${ROOT}/scripts/ros2_publish_stereo_mp4.py" \
  --image-source "${IMAGE_SOURCE_TYPE}" \
  --left-mp4 "${LEFT_MP4}" \
  --right-mp4 "${RIGHT_MP4}" \
  --stamp-bag "${STAMP_DB}" \
  --left-stamp-topic "${LEFT_STAMP_TOPIC}" \
  --right-stamp-topic "${RIGHT_STAMP_TOPIC}" \
  "${stamp_match_args[@]}" \
  --auto-trim-to-imu \
  "${publish_window_args[@]}" \
  "${stereo_timing_args[@]}" \
  "${schedule_imu_args[@]}" \
  --max-duration-sec "${PLAY_SECONDS}" \
  --rate "${PLAY_RATE}" \
  --export-schedule-dir "${SCHEDULE_DIR}" \
  --export-schedule-only \
  > "${LOG_DIR}/mp4_schedule_export.log" 2>&1
# shellcheck disable=SC1091
source "${SCHEDULE_DIR}/metadata.env"

start_bag_localization_path() {
  localization_start_stamp_ns="0"
  if [[ -n "${STEREO_SCHEDULE_CSV:-}" && -f "${STEREO_SCHEDULE_CSV}" ]]; then
    localization_start_stamp_ns="$(awk -F, 'NR==2 {print $1}' "${STEREO_SCHEDULE_CSV}")"
  fi
  local loop_args=()
  if [[ "${SYNC_REPLAY_PATHS}" != "1" ]]; then
    loop_args+=(--loop)
  fi
  local restamp_args=()
  if [[ "${RESTAMP_NOW}" == "1" ]]; then
    restamp_args+=(--restamp-now)
  fi
  python3 "${ROOT}/scripts/ros2_publish_bag_odometry_path.py" \
    --bag-db "${STAMP_DB}" \
    --source-topic /odometry/filtered \
    --odom-topic /localization/odometry \
    --path-topic /localization/path \
    --frame-id world \
    --child-frame-id fcu_link \
    --time-source header \
    --start-stamp-ns "${localization_start_stamp_ns}" \
    --max-duration-sec "${PLAY_SECONDS}" \
    --rate "${PLAY_RATE}" \
    --align-csv "${DVL_CSV}" \
    --align-csv-coordinate-frame "${DVL_COORDINATE_FRAME}" \
    --align-mode "${LOCALIZATION_ALIGN_MODE}" \
    --align-yaw-deg "${LOCALIZATION_ALIGN_YAW_DEG}" \
    --publish-empty-on-start \
    "${restamp_args[@]}" \
    "${loop_args[@]}" \
    > "${LOG_DIR}/bag_localization_path.log" 2>&1 &
  bag_localization_pid=$!
}

if [[ "${RUN_BAG_LOCALIZATION_PATH}" == "1" && "${SYNC_REPLAY_PATHS}" != "1" ]]; then
  start_bag_localization_path
fi

run_vins_stereo_loop() {
  while true; do
    stop_orphan_live_nodes
    if [[ "${SYNC_REPLAY_PATHS}" == "1" ]]; then
      stop_child "${bag_localization_pid}"
      stop_child "${dvl_pid}"
      bag_localization_pid=""
      dvl_pid=""
    fi
    rm -f "${ROOT}/outputs/ros2_vins_fusion_live/vio.csv"
    rm -f "${ROOT}/outputs/ros2_vins_fusion_live/extrinsic_parameter.csv"

    "${VINS_NODE}" "${CONFIG}" \
      > "${LOG_DIR}/vins_node.log" 2>&1 &
    vins_pid=$!

    sleep 3

    if [[ "${SYNC_REPLAY_PATHS}" == "1" ]]; then
      start_dvl_reference_path progressive
      if [[ "${RUN_BAG_LOCALIZATION_PATH}" == "1" ]]; then
        start_bag_localization_path
      fi
    fi

    publish_imu_args=(
      --imu-source-type "${IMU_SOURCE_TYPE}"
      --imu-start-margin-sec 0.03
      --imu-end-margin-sec 0.03
      --publish-imu-from-stamp-bag
      --imu-output-topic "${IMU_OUTPUT_TOPIC}"
      --imu-frame-id "${IMU_FRAME_ID}"
      --imu-frame-transform "${IMU_FRAME_TRANSFORM}"
      --imu-max-delta-sec 0.025
      --imu-resample-hz "${IMU_RESAMPLE_HZ}"
      --imu-resample-max-gap-sec 1.0
      --imu-ahead-sec "${MP4_IMU_AHEAD_SEC:-0.12}"
    )
    if [[ -n "${IMU_SOURCE_TOPIC}" ]]; then
      publish_imu_args+=(--imu-source-topic "${IMU_SOURCE_TOPIC}")
    else
      publish_imu_args+=(
        --gyro-stamp-topic /camera/camera/gyro/sample
        --accel-stamp-topic /camera/camera/accel/sample
      )
    fi

    python3 "${ROOT}/scripts/ros2_publish_stereo_mp4.py" \
      --image-source "${IMAGE_SOURCE_TYPE}" \
      --left-mp4 "${LEFT_MP4}" \
      --right-mp4 "${RIGHT_MP4}" \
      --stamp-bag "${STAMP_DB}" \
      --left-topic /camera/camera/infra1/image_rect_raw \
      --right-topic /camera/camera/infra2/image_rect_raw \
      --left-stamp-topic "${LEFT_STAMP_TOPIC}" \
      --right-stamp-topic "${RIGHT_STAMP_TOPIC}" \
      "${stamp_match_args[@]}" \
      --auto-trim-to-imu \
      "${publish_window_args[@]}" \
      "${publish_imu_args[@]}" \
      --max-duration-sec "${PLAY_SECONDS}" \
      --rate "${PLAY_RATE}" \
      > "${LOG_DIR}/mp4_publish.log" 2>&1 &
    mp4_pid=$!

    # In Docker Desktop on Apple Silicon the amd64 ROS image is often slower
    # than real time. Let the MP4 publisher finish its own max-duration window
    # instead of killing it after PLAY_SECONDS wall-clock seconds.
    wait "${mp4_pid}" >/dev/null 2>&1 || true
    wait_for_vins_drain "${vins_pid}"

    if [[ -f "${ROOT}/outputs/ros2_vins_fusion_live/vio.csv" ]]; then
      cp "${ROOT}/outputs/ros2_vins_fusion_live/vio.csv" \
        "${ROOT}/outputs/ros2_vins_fusion_live/last_completed_vio.csv"
    fi

    stop_child "${mp4_pid}"
    stop_child "${vins_pid}"
    wait "${mp4_pid}" "${vins_pid}" >/dev/null 2>&1 || true
    vins_pid=""
    mp4_pid=""
    if [[ "${SYNC_REPLAY_PATHS}" == "1" ]]; then
      stop_child "${bag_localization_pid}"
      stop_child "${dvl_pid}"
      bag_localization_pid=""
      dvl_pid=""
    fi
    stop_orphan_live_nodes
    if [[ "${RUN_ONCE}" == "1" ]]; then
      break
    fi
    sleep 2
  done
}

run_vins_stereo_loop &
loop_pid=$!

cat <<EOF
Docker ROS2 live stack started.
ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}
LOG_DIR=${LOG_DIR}
RViz config=${RVIZ_CONFIG}
RViz backend=${RVIZ_BACKEND}
Restamp replay to wall time=${RESTAMP_NOW}
Uniform stereo timestamps=${UNIFORM_STEREO_TIMESTAMPS}
Image source type=${IMAGE_SOURCE_TYPE}
Swap stereo source inputs=${SWAP_STEREO_MP4}
Run once=${RUN_ONCE}
VINS config=${CONFIG}
IMU source topic=${IMU_SOURCE_TOPIC:-gyro+accel}
IMU source type=${IMU_SOURCE_TYPE}
IMU publish topic=${IMU_OUTPUT_TOPIC}
Stamp time source=${STAMP_TIME_SOURCE}
Stereo schedule source=${STEREO_SCHEDULE_SOURCE_CSV:-stamp-bag rematch}
Publish start offset sec=${PUBLISH_START_OFFSET_SEC}
Video start index override=${VIDEO_START_INDEX_OVERRIDE:-none}
DVL coordinate frame=${DVL_COORDINATE_FRAME}
DVL reference align mode=${DVL_REFERENCE_ALIGN_MODE}
Localization reference CSV=${LOCALIZATION_REFERENCE_CSV:-none}
EKF clock topic=${EKF_CLOCK_TOPIC}
EKF odom input topic=${EKF_CLOCKED_ODOM_TOPIC}
IMU frame=${IMU_FRAME_ID}
Pixhawk-camera TF=${PUBLISH_PIXHAWK_CAMERA_TF}
EKF localization=${RUN_EKF_LOCALIZATION}
Bag localization path=${RUN_BAG_LOCALIZATION_PATH}
Synchronized replay paths=${SYNC_REPLAY_PATHS}
TF localized debug path=${RUN_TF_LOCALIZED_PATH}
Localization align mode=${LOCALIZATION_ALIGN_MODE}
Localization align yaw deg=${LOCALIZATION_ALIGN_YAW_DEG}
EKF config=${EKF_CONFIG}
EOF

if [[ "${RUN_RVIZ}" == "1" ]]; then
  if [[ "${RVIZ_BACKEND}" == "vnc" ]]; then
    export DISPLAY=:99
    export LIBGL_ALWAYS_SOFTWARE=1
    unset LIBGL_ALWAYS_INDIRECT
    export XDG_RUNTIME_DIR=/tmp/runtime-root
    mkdir -p "${XDG_RUNTIME_DIR}"
    chmod 700 "${XDG_RUNTIME_DIR}"

    Xvfb :99 -screen 0 1600x1000x24 +extension GLX +render -noreset \
      > "${LOG_DIR}/xvfb.log" 2>&1 &
    xvfb_pid=$!
    sleep 1

    fluxbox > "${LOG_DIR}/fluxbox.log" 2>&1 &
    fluxbox_pid=$!

    x11vnc -display :99 -forever -shared -nopw -rfbport 5900 \
      > "${LOG_DIR}/x11vnc.log" 2>&1 &
    x11vnc_pid=$!

    websockify --web=/usr/share/novnc/ 6080 localhost:5900 \
      > "${LOG_DIR}/novnc.log" 2>&1 &
    novnc_pid=$!

    echo "noVNC URL: http://localhost:6080/vnc.html?autoconnect=1&resize=scale"
  fi

  rviz2 -d "${RVIZ_CONFIG}" > "${LOG_DIR}/rviz.log" 2>&1 &
  rviz_pid=$!
  wait "${rviz_pid}"
else
  wait "${loop_pid}"
fi
