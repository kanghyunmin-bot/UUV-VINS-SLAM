#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

set +u
source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311
set -u

# Keep RViz and its CSV publishers in the same ROS discovery domain.
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

CSV="${1:-outputs/vins_fusion/vins_odometry.csv}"
POINT_CLOUD_CSV="${2:-outputs/live/latest_point_cloud.csv}"
if [[ ! -f "$CSV" ]]; then
  echo "Missing VINS odometry CSV: $CSV" >&2
  echo "Run scripts/convert_vins_vio_to_csv.py after VINS-Fusion produces vio.txt." >&2
  exit 2
fi

if pgrep -fl "ros2_publish_odometry.py" >/dev/null 2>&1; then
  pkill -f "ros2_publish_odometry.py" || true
fi

python scripts/ros2_publish_odometry.py \
  --csv "$CSV" \
  --point-cloud-csv "$POINT_CLOUD_CSV" \
  --rate "${RVIZ_ODOM_RATE:-60}" \
  --loop \
  --coordinate-frame ros \
  --feature-history-frames "${RVIZ_FEATURE_HISTORY_FRAMES:-0}" \
  --feature-track-history "${RVIZ_FEATURE_TRACK_HISTORY:-80}" &

exec rviz2 -d rviz/vio_frontend.rviz
