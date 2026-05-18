#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Conda's ROS activation script reads CONDA_BUILD even when it is unset.
# Keep strict mode for the script, but relax nounset while conda activates.
set +u
source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311
set -u

# Keep RViz and its CSV publishers in the same ROS discovery domain.
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

if pgrep -fl "ros2_publish_odometry.py" >/dev/null 2>&1; then
  pkill -f "ros2_publish_odometry.py" || true
fi

python scripts/ros2_publish_odometry.py \
  --csv outputs/live/latest_odometry.csv \
  --point-cloud-csv outputs/live/latest_point_cloud.csv \
  --rate "${RVIZ_ODOM_RATE:-60}" \
  --loop \
  --coordinate-frame ros \
  --feature-history-frames "${RVIZ_FEATURE_HISTORY_FRAMES:-0}" \
  --feature-track-history "${RVIZ_FEATURE_TRACK_HISTORY:-80}" \
  --watch &

exec rviz2 -d rviz/vio_frontend.rviz
