#!/usr/bin/env zsh
set -e

ROOT=/Users/kanghyunmin/Desktop/under_water_image_match
LOG_DIR=${LOG_DIR:-/tmp/uuv_rviz_loop}
PLAY_SECONDS=${PLAY_SECONDS:-30}
ROSBAG_INPUT_DIR=${ROSBAG_INPUT_DIR:-$ROOT/data/rosbag_active}
DVL_CSV=${DVL_CSV:-$ROSBAG_INPUT_DIR/dvl_reference_0_30s.csv}
RVIZ_CONFIG=${RVIZ_CONFIG:-$ROOT/rviz/vins_ros2_live_with_dvl.rviz}

mkdir -p "$LOG_DIR"

for session in uuv_recent_finalturn uuv_dvl_ref uuv_vins_loop uuv_rviz; do
  screen -S "$session" -X quit >/dev/null 2>&1 || true
done

pkill -f 'ros2_publish_path_overlay.py.*vins_path_overlay' >/dev/null 2>&1 || true
pkill -f 'ros2_publish_odometry.py.*feature_cloud_reference' >/dev/null 2>&1 || true
pkill -f '[r]un_rviz_vins_loop.zsh' >/dev/null 2>&1 || true
pkill -f 'ros2_publish_static_path.py.*dvl_reference_path' >/dev/null 2>&1 || true
pkill -f '[r]os2_publish_stereo_mp4.py' >/dev/null 2>&1 || true
pkill -f '[r]os2_fuse_realsense_imu.py' >/dev/null 2>&1 || true
pkill -f '[r]os2 bag play .*rosbag_all_topics_90s' >/dev/null 2>&1 || true
pkill -f '[v]ins_node .*underwater_realsense_stereo_imu_config.yaml' >/dev/null 2>&1 || true
rm -rf "$LOG_DIR/run_rviz_vins_loop.lock" "$LOG_DIR/recent_finalturn_overlay.lock"

common_setup="set +u; source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311; set -u; source '$ROOT/ros2_overlay_ws/install/setup.zsh'; source '$ROOT/VINS-Fusion-ROS2/install/setup.zsh'; export ROS_LOCALHOST_ONLY=1; export DYLD_LIBRARY_PATH=/Users/kanghyunmin/miniconda3/envs/ros2_h311/lib:/Users/kanghyunmin/miniconda3/envs/ros2_h311/opt/rviz_ogre_vendor/lib:\${DYLD_LIBRARY_PATH:-}"

screen -dmS uuv_dvl_ref zsh -lc "
  cd '$ROOT' &&
  $common_setup &&
  python '$ROOT/scripts/ros2_publish_static_path.py' \
    --csv '$DVL_CSV' \
    --path-topic /dvl_reference_path \
    --marker-topic /dvl_reference_line_marker \
    --direction-marker-topic /dvl_reference_direction_markers \
    --frame-id world \
    --coordinate-frame flip-y \
    --line-width 0.08 \
    --direction-arrow-stride 16 \
    > '$LOG_DIR/dvl_reference.log' 2>&1
"

screen -dmS uuv_vins_loop zsh -lc "
  cd '$ROOT' &&
  ROSBAG_INPUT_DIR='$ROSBAG_INPUT_DIR' \
  PLAY_SECONDS='$PLAY_SECONDS' \
  AUTO_TRIM_TO_IMU=1 \
  CLEAR_VINS_LOG=1 \
  '$ROOT/scripts/run_rviz_vins_loop.zsh' \
    > '$LOG_DIR/vins_loop_screen.log' 2>&1
"

screen -dmS uuv_rviz zsh -lc "
  cd '$ROOT' &&
  $common_setup &&
  rviz2 -d '$RVIZ_CONFIG' \
    > '$LOG_DIR/rviz.log' 2>&1
"

echo "Started realtime VINS-Fusion RViz sessions:"
screen -ls | sed -n '/uuv_dvl_ref/p;/uuv_vins_loop/p;/uuv_rviz/p'
