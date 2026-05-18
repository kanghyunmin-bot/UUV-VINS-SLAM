#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="${1:-$PROJECT_ROOT/outputs/vins_fusion/underwater_stereo_kitti}"
CONFIG_FILE="${2:-$PROJECT_ROOT/outputs/vins_fusion/config/underwater_stereo_config.yaml}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/vins_fusion/vins_output"

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
  echo "Missing VINS-Fusion config: $CONFIG_FILE" >&2
  echo "Run .venv/bin/python scripts/vins_make_stereo_config.py first." >&2
  exit 2
fi

if [[ ! -f "$DATASET_DIR/times.txt" ]]; then
  echo "Missing exported stereo dataset: $DATASET_DIR" >&2
  echo "Run .venv/bin/python scripts/export_vins_kitti_stereo.py --force first." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

docker run --rm \
  --platform linux/amd64 \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$PROJECT_ROOT" \
  underwater-vins-fusion:kinetic \
  /bin/bash -lc "
    source /opt/ros/kinetic/setup.bash
    source /root/catkin_ws/devel/setup.bash
    roscore >/tmp/roscore.log 2>&1 &
    sleep 2
    rosrun vins kitti_odom_test '$CONFIG_FILE' '$DATASET_DIR'
  "
