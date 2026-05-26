#!/usr/bin/env zsh
set -e

ROOT=/Users/kanghyunmin/Desktop/under_water_image_match
ROSBAG_INPUT_DIR=${ROSBAG_INPUT_DIR:-$ROOT/data/rosbag_active}
BAG=${BAG:-$ROSBAG_INPUT_DIR}
CONFIG=${CONFIG:-$ROOT/VINS-Fusion-ROS2/config/realsense_d435i/realsense_stereo_mavros_imu_config.yaml}
LOG_DIR=${LOG_DIR:-/tmp/uuv_rviz_loop}
PLAY_SECONDS=${PLAY_SECONDS:-30}
INFRA1_MP4=${INFRA1_MP4:-$ROSBAG_INPUT_DIR/infra1_image_rect_raw.mp4}
INFRA2_MP4=${INFRA2_MP4:-$ROSBAG_INPUT_DIR/infra2_image_rect_raw.mp4}
LEFT_MP4=${LEFT_MP4:-$INFRA1_MP4}
RIGHT_MP4=${RIGHT_MP4:-$INFRA2_MP4}
MP4_START_DELAY=${MP4_START_DELAY:-0.05}
AUTO_TRIM_TO_IMU=${AUTO_TRIM_TO_IMU:-1}
MP4_PUBLISH_IMU=${MP4_PUBLISH_IMU:-1}
MP4_MANUAL_START_OFFSET=${MP4_MANUAL_START_OFFSET:-0.0}
CLEAR_VINS_LOG=${CLEAR_VINS_LOG:-1}
IMU_SOURCE_TOPIC=${IMU_SOURCE_TOPIC:-/mavros/imu/data}
IMU_OUTPUT_TOPIC=${IMU_OUTPUT_TOPIC:-$IMU_SOURCE_TOPIC}
IMU_FRAME_ID=${IMU_FRAME_ID:-fcu_link}
IMU_FRAME_TRANSFORM=${IMU_FRAME_TRANSFORM:-identity}
if [[ -d "$BAG" ]]; then
  STAMP_DB=${STAMP_DB:-$BAG/stamp_bag.db3}
else
  STAMP_DB=${STAMP_DB:-$BAG}
fi

mkdir -p "$LOG_DIR"
LOCK_DIR="$LOG_DIR/run_rviz_vins_loop.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[$(date)] another run_rviz_vins_loop is already active; exiting" >> "$LOG_DIR/loop_controller.log"
  exit 0
fi

cleanup() {
  for pid in ${bag_pid:-} ${vins_pid:-} ${imu_pid:-} ${mp4_pid:-}; do
    if [[ -n "$pid" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  rm -rf "$LOCK_DIR"
}
trap cleanup EXIT INT TERM

cd "$ROOT/VINS-Fusion-ROS2" || exit 1

set +u
source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311
set -u
source "$ROOT/ros2_overlay_ws/install/setup.zsh"
source "$ROOT/VINS-Fusion-ROS2/install/setup.zsh"

export ROS_LOCALHOST_ONLY=1
export DYLD_LIBRARY_PATH=/Users/kanghyunmin/miniconda3/envs/ros2_h311/lib:/Users/kanghyunmin/miniconda3/envs/ros2_h311/opt/rviz_ogre_vendor/lib:${DYLD_LIBRARY_PATH:-}
VINS_DYLD_INSERT=/Users/kanghyunmin/miniconda3/envs/ros2_h311/lib/libpython3.11.dylib

while true; do
  echo "[$(date)] loop start" > "$LOG_DIR/loop_controller.log"
  if [[ "$CLEAR_VINS_LOG" == "1" ]]; then
    rm -f "$ROOT/outputs/ros2_vins_fusion_live/vio.csv"
  fi

  env DYLD_INSERT_LIBRARIES="$VINS_DYLD_INSERT" \
    "$ROOT/VINS-Fusion-ROS2/install/vins/lib/vins/vins_node" "$CONFIG" \
    > "$LOG_DIR/vins_node.log" 2>&1 &
  vins_pid=$!
  echo "vins_pid=$vins_pid" >> "$LOG_DIR/loop_controller.log"

  sleep 3

  if [[ "$MP4_PUBLISH_IMU" != "1" ]]; then
    python "$ROOT/scripts/ros2_fuse_realsense_imu.py" \
      --gyro-topic /camera/camera/gyro/sample \
      --accel-topic /camera/camera/accel/sample \
      --output-topic /camera/camera/imu \
      --frame-id camera_imu \
      --frame-transform identity \
      --max-delta-sec 0.025 \
      > "$LOG_DIR/imu_fuse.log" 2>&1 &
    imu_pid=$!
    echo "imu_pid=$imu_pid" >> "$LOG_DIR/loop_controller.log"

    ros2 bag play "$BAG" --rate 1.0 \
      --topics /camera/camera/gyro/sample /camera/camera/accel/sample \
      > "$LOG_DIR/bag_play.log" 2>&1 &
    bag_pid=$!
    echo "bag_pid=$bag_pid" >> "$LOG_DIR/loop_controller.log"
  else
    echo "imu_pid=integrated_mp4_publisher" >> "$LOG_DIR/loop_controller.log"
    echo "bag_pid=disabled_integrated_mp4_publisher" >> "$LOG_DIR/loop_controller.log"
  fi

  sleep "$MP4_START_DELAY"

  mp4_cmd=(
    python "$ROOT/scripts/ros2_publish_stereo_mp4.py"
    --left-mp4 "$LEFT_MP4" \
    --right-mp4 "$RIGHT_MP4" \
    --stamp-bag "$STAMP_DB" \
    --left-topic /camera/camera/infra1/image_rect_raw \
    --right-topic /camera/camera/infra2/image_rect_raw \
    --left-stamp-topic /camera/camera/infra1/image_rect_raw \
    --right-stamp-topic /camera/camera/infra2/image_rect_raw \
    --max-duration-sec "$PLAY_SECONDS" \
    --rate 1.0 \
    --queue-size "${ROS_TOPIC_QUEUE_SIZE:-2000}"
  )
  if [[ "$AUTO_TRIM_TO_IMU" == "1" ]]; then
    mp4_cmd+=(
      --auto-trim-to-imu
      --imu-start-margin-sec 0.03
      --imu-end-margin-sec 0.03
    )
    if [[ -n "$IMU_SOURCE_TOPIC" ]]; then
      mp4_cmd+=(--imu-source-topic "$IMU_SOURCE_TOPIC")
    else
      mp4_cmd+=(
        --gyro-stamp-topic /camera/camera/gyro/sample
        --accel-stamp-topic /camera/camera/accel/sample
      )
    fi
  else
    mp4_cmd+=(--start-offset-sec "$MP4_MANUAL_START_OFFSET")
  fi
  if [[ "$MP4_PUBLISH_IMU" == "1" ]]; then
    mp4_cmd+=(
      --publish-imu-from-stamp-bag
      --imu-output-topic "$IMU_OUTPUT_TOPIC"
      --imu-frame-id "$IMU_FRAME_ID"
      --imu-frame-transform "$IMU_FRAME_TRANSFORM"
      --imu-max-delta-sec 0.025
      --imu-resample-hz 200
      --imu-resample-max-gap-sec 1.0
      --imu-ahead-sec "${MP4_IMU_AHEAD_SEC:-0.12}"
    )
  fi
  "${mp4_cmd[@]}" > "$LOG_DIR/mp4_publish.log" 2>&1 &
  mp4_pid=$!
  echo "mp4_pid=$mp4_pid" >> "$LOG_DIR/loop_controller.log"

  sleep "$PLAY_SECONDS"

  if [[ -f "$ROOT/outputs/ros2_vins_fusion_live/vio.csv" ]]; then
    cp "$ROOT/outputs/ros2_vins_fusion_live/vio.csv" "$ROOT/outputs/ros2_vins_fusion_live/last_completed_vio.csv"
  fi

  kill ${bag_pid:-} "$vins_pid" ${imu_pid:-} "$mp4_pid" >/dev/null 2>&1 || true
  wait ${bag_pid:-} "$vins_pid" ${imu_pid:-} "$mp4_pid" >/dev/null 2>&1 || true
  echo "[$(date)] loop restart" >> "$LOG_DIR/loop_controller.log"
  sleep 2
done
