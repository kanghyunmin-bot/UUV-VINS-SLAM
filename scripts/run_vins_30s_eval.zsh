#!/usr/bin/env zsh
set -e

ROOT=/Users/kanghyunmin/Desktop/under_water_image_match
ROSBAG_INPUT_DIR=${ROSBAG_INPUT_DIR:-$ROOT/data/rosbag_active/localization bag}
BAG=${BAG:-$ROSBAG_INPUT_DIR}
CONFIG=${CONFIG:-$ROOT/VINS-Fusion-ROS2/config/realsense_d435i/underwater_realsense_stereo_mavros_imu_config.yaml}
LOG_DIR=/tmp/uuv_rviz_run
PLAY_SECONDS=${PLAY_SECONDS:-36}
VINS_EVAL_SECONDS=${VINS_EVAL_SECONDS:-30}
USE_CONFIG_IMU=$(awk '/^[[:space:]]*imu:[[:space:]]*/ {print $2; exit}' "$CONFIG")
IMU_FRAME_TRANSFORM=${IMU_FRAME_TRANSFORM:-identity}

mkdir -p "$LOG_DIR"
for s in uuv_vins_node uuv_bag_play uuv_imu_fuse; do
  screen -S "$s" -X quit >/dev/null 2>&1 || true
done
pkill -f '/vins_node .*underwater_realsense' >/dev/null 2>&1 || true
pkill -f 'ros2 bag play .*rosbag_all_topics_90s' >/dev/null 2>&1 || true
pkill -f 'ros2_fuse_realsense_imu.py' >/dev/null 2>&1 || true
rm -f "$ROOT/outputs/ros2_vins_fusion_live/vio.csv" "$LOG_DIR/vins_node.log" "$LOG_DIR/bag_play.log" "$LOG_DIR/imu_fuse.log"

if [[ "$USE_CONFIG_IMU" == "1" ]]; then
  screen -dmS uuv_imu_fuse zsh -lc "cd '$ROOT' && \
    source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311 && \
    source '$ROOT/ros2_overlay_ws/install/setup.zsh' && \
    export ROS_LOCALHOST_ONLY=1 && \
    /Users/kanghyunmin/miniconda3/envs/ros2_h311/bin/python '$ROOT/scripts/ros2_fuse_realsense_imu.py' \
      --gyro-topic /camera/camera/gyro/sample \
      --accel-topic /camera/camera/accel/sample \
      --output-topic /camera/camera/imu \
      --frame-id camera_imu \
      --frame-transform '$IMU_FRAME_TRANSFORM' \
      --max-delta-sec 0.025 > '$LOG_DIR/imu_fuse.log' 2>&1"
fi

screen -dmS uuv_vins_node zsh -lc "cd '$ROOT/VINS-Fusion-ROS2' && \
  source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311 && \
  source '$ROOT/ros2_overlay_ws/install/setup.zsh' && \
  source '$ROOT/VINS-Fusion-ROS2/install/setup.zsh' && \
  export ROS_LOCALHOST_ONLY=1 && \
  export DYLD_LIBRARY_PATH=/Users/kanghyunmin/miniconda3/envs/ros2_h311/lib:/Users/kanghyunmin/miniconda3/envs/ros2_h311/opt/rviz_ogre_vendor/lib:\${DYLD_LIBRARY_PATH:-} && \
  export DYLD_INSERT_LIBRARIES=/Users/kanghyunmin/miniconda3/envs/ros2_h311/lib/libpython3.11.dylib && \
  '$ROOT/VINS-Fusion-ROS2/install/vins/lib/vins/vins_node' '$CONFIG' > '$LOG_DIR/vins_node.log' 2>&1"

sleep 3

screen -dmS uuv_bag_play zsh -lc "source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311 && \
  source '$ROOT/ros2_overlay_ws/install/setup.zsh' && \
  export ROS_LOCALHOST_ONLY=1 && \
  ros2 bag play '$BAG' --rate 1.0 > '$LOG_DIR/bag_play.log' 2>&1 & \
  pid=\$!; sleep '$PLAY_SECONDS'; kill \$pid 2>/dev/null || true; wait \$pid 2>/dev/null || true; \
  echo 'bag playback stopped at ${PLAY_SECONDS}s wall-clock limit' >> '$LOG_DIR/bag_play.log'"

sleep $((PLAY_SECONDS + 6))

# Freeze the output log before scoring. VINS can keep draining its queue after
# rosbag playback stops, which makes RMSE/correlation depend on read timing.
screen -S uuv_bag_play -X quit >/dev/null 2>&1 || true
sleep 4
screen -S uuv_vins_node -X quit >/dev/null 2>&1 || true
screen -S uuv_imu_fuse -X quit >/dev/null 2>&1 || true
pkill -f '/vins_node .*underwater_realsense' >/dev/null 2>&1 || true
pkill -f 'ros2_fuse_realsense_imu.py' >/dev/null 2>&1 || true

/Users/kanghyunmin/miniconda3/envs/ros2_h311/bin/python "$ROOT/scripts/evaluate_current_vins_dvl.py"
/Users/kanghyunmin/miniconda3/envs/ros2_h311/bin/python "$ROOT/scripts/overlay_vins_dvl_from_logs.py" \
  --vins-log "$ROOT/outputs/ros2_vins_fusion_live/vio.csv" \
  --dvl-csv "$ROSBAG_INPUT_DIR/dvl_reference_34_85s.csv" \
  --max-vins-duration-sec "$VINS_EVAL_SECONDS" \
  --auto-vins-yaw-sweep \
  --yaw-sweep-objective time \
  --overlay-csv "$ROOT/outputs/evaluation/current_vins_dvl_log_overlay.csv" \
  --summary-json "$ROOT/outputs/evaluation/current_vins_dvl_log_overlay_summary.json" \
  --plot-png "$ROOT/outputs/evaluation/current_vins_dvl_log_overlay.png" \
  --single-xy-plot-png "$ROOT/outputs/evaluation/current_vins_dvl_log_overlay_xy.png"
