#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VINS_WS="${VINS_WS:-$HOME/vins_fusion_ws}"
VINS_SRC="$VINS_WS/src"
VINS_REPO="$VINS_SRC/VINS-Fusion"

if [[ "$(uname -s)" != "Linux" ]]; then
  cat >&2 <<'EOF'
VINS-Fusion is a ROS1/catkin project. Build it on Ubuntu with ROS Kinetic/Melodic/Noetic.
This Mac can prepare datasets/configs, but it cannot build/run VINS-Fusion with the current ROS2 conda environment.
EOF
  exit 2
fi

if ! command -v catkin_make >/dev/null 2>&1; then
  echo "catkin_make not found. Source your ROS1 setup.bash first." >&2
  exit 2
fi

mkdir -p "$VINS_SRC"
if [[ ! -d "$VINS_REPO/.git" ]]; then
  git clone https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git "$VINS_REPO"
fi

cd "$VINS_WS"
catkin_make
echo "Built VINS-Fusion workspace: $VINS_WS"
