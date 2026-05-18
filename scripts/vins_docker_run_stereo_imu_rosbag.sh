#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="${1:-$PROJECT_ROOT/outputs/vins_fusion/underwater_stereo_imu_rosbag}"
CONFIG_FILE="${2:-$PROJECT_ROOT/outputs/vins_fusion/config_stereo_imu/underwater_stereo_imu_config.yaml}"
OUTPUT_DIR="${3:-$PROJECT_ROOT/outputs/vins_fusion/vins_output_stereo_imu}"
REALTIME_FACTOR="${VINS_REPLAY_REALTIME_FACTOR:-8}"
INPUT_BAG="$OUTPUT_DIR/stereo_imu_input.bag"

if ! docker info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Docker daemon is not running.
Open Docker Desktop first, wait until it says Docker is running, then retry.
EOF
  exit 2
fi

if [[ -z "$(docker image ls -q underwater-vins-fusion:kinetic)" ]]; then
  echo "Missing Docker image: underwater-vins-fusion:kinetic" >&2
  echo "Run scripts/vins_docker_build.sh first." >&2
  exit 2
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing VINS-Fusion stereo+IMU config: $CONFIG_FILE" >&2
  exit 2
fi

if [[ ! -f "$DATASET_DIR/times.txt" || ! -f "$DATASET_DIR/imu.csv" ]]; then
  echo "Missing exported stereo+IMU dataset: $DATASET_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

docker run --rm \
  --platform linux/amd64 \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$PROJECT_ROOT" \
  underwater-vins-fusion:kinetic \
  /bin/bash -lc "
    set -e
    source /opt/ros/kinetic/setup.bash
    source /root/catkin_ws/devel/setup.bash
    mkdir -p '$OUTPUT_DIR'
    python '$PROJECT_ROOT/scripts/vins_write_stereo_imu_ros1_bag.py' '$DATASET_DIR' \
      --output-bag '$INPUT_BAG' \
      --force >'$OUTPUT_DIR/bag_writer.log' 2>&1
    roscore >'$OUTPUT_DIR/roscore.log' 2>&1 &
    ROSCORE_PID=\$!
    sleep 2
    rosrun vins vins_node '$CONFIG_FILE' >'$OUTPUT_DIR/vins_node.log' 2>&1 &
    VINS_PID=\$!
    sleep 4
    rosbag play '$INPUT_BAG' -r '$REALTIME_FACTOR' >'$OUTPUT_DIR/rosbag_play.log' 2>&1 || PLAY_STATUS=\$?
    sleep 3
    rosnode kill /vins_estimator >/dev/null 2>&1 || true
    kill \$VINS_PID >/dev/null 2>&1 || true
    kill \$ROSCORE_PID >/dev/null 2>&1 || true
    wait \$VINS_PID >/dev/null 2>&1 || true
    wait \$ROSCORE_PID >/dev/null 2>&1 || true
    if [ \"\${PLAY_STATUS:-0}\" != \"0\" ]; then
      cat '$OUTPUT_DIR/rosbag_play.log' >&2
      exit \"\$PLAY_STATUS\"
    fi
    if [ ! -s '$OUTPUT_DIR/vio.csv' ]; then
      echo 'VINS-Fusion did not produce vio.csv. Check vins_node.log, rosbag_play.log, and bag_writer.log in the output directory.' >&2
      tail -120 '$OUTPUT_DIR/vins_node.log' >&2 || true
      exit 1
    fi
  "
