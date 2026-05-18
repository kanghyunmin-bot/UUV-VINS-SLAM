#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VINS_WS="${VINS_WS:-$HOME/vins_fusion_ws}"
DATASET_DIR="${1:-$PROJECT_ROOT/outputs/vins_fusion/underwater_stereo_kitti}"
CONFIG_FILE="${2:-$PROJECT_ROOT/outputs/vins_fusion/config/underwater_stereo_config.yaml}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing VINS-Fusion config: $CONFIG_FILE" >&2
  echo "Run scripts/vins_make_stereo_config.py first." >&2
  exit 2
fi
if [[ ! -f "$DATASET_DIR/times.txt" ]]; then
  echo "Missing VINS-Fusion KITTI-style dataset: $DATASET_DIR" >&2
  echo "Run scripts/export_vins_kitti_stereo.py first." >&2
  exit 2
fi

if [[ ! -f "$VINS_WS/devel/setup.bash" ]]; then
  echo "Missing VINS-Fusion catkin workspace: $VINS_WS" >&2
  echo "Build it with scripts/vins_prepare_workspace.sh on Ubuntu ROS1." >&2
  exit 2
fi

source "$VINS_WS/devel/setup.bash"
rosrun vins kitti_odom_test "$CONFIG_FILE" "$DATASET_DIR"
