#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

set +u
source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311
set -u

# Keep RViz and all ROS2 publishers in the same local discovery domain.
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

VINS_CSV="${1:-outputs/live/latest_odometry_vins_pure_stereo_imu.csv}"
DVL_CSV="${2:-outputs/evaluation/dvl_xyz_30s_reference.csv}"
CONTROL_CSV="${3:-}"
FEATURE_ODOM_CSV="${4:-$VINS_CSV}"
FEATURE_POINT_CLOUD_CSV="${5:-outputs/live/latest_point_cloud.csv}"
FEATURE_TARGET_CSV="${6:-$VINS_CSV}"
mkdir -p outputs/logs

reject_dvl_derived_csv() {
  local role="$1"
  local csv="$2"
  if [[ -z "$csv" || ! -f "$csv" ]]; then
    return
  fi
  local header
  header="$(head -n 1 "$csv")"
  if [[ "$csv" == *dvl_anchor* ]] || grep -q "dvl_anchor" <<< "$header"; then
    echo "Refusing $role CSV because it is DVL-anchored, not pure VINS: $csv" >&2
    echo "Set ALLOW_DVL_DERIVED_DIAGNOSTIC=1 only for explicit diagnostics." >&2
    exit 3
  fi
  if [[ "$csv" == *dvl_fit* ]] || grep -q "dvl_fit" <<< "$header"; then
    echo "Refusing $role CSV because it is DVL-fit/postprocessed, not pure VINS: $csv" >&2
    echo "Set ALLOW_DVL_DERIVED_DIAGNOSTIC=1 only for explicit diagnostics." >&2
    exit 3
  fi
}

if [[ ! -f "$VINS_CSV" ]]; then
  echo "Missing VINS odometry CSV: $VINS_CSV" >&2
  exit 2
fi
if [[ ! -f "$DVL_CSV" ]]; then
  echo "Missing DVL reference CSV: $DVL_CSV" >&2
  exit 2
fi
if [[ "${ALLOW_DVL_DERIVED_DIAGNOSTIC:-0}" != "1" ]]; then
  reject_dvl_derived_csv "VINS" "$VINS_CSV"
  reject_dvl_derived_csv "control" "$CONTROL_CSV"
  reject_dvl_derived_csv "feature target" "$FEATURE_TARGET_CSV"
fi
HAS_CONTROL=0
if [[ -n "$CONTROL_CSV" && -f "$CONTROL_CSV" ]]; then
  HAS_CONTROL=1
fi

if pgrep -fl "ros2_publish_path_overlay.py" >/dev/null 2>&1; then
  pkill -f "ros2_publish_path_overlay.py" || true
fi
if pgrep -fl "ros2_publish_odometry.py" >/dev/null 2>&1; then
  pkill -f "ros2_publish_odometry.py" || true
fi

if [[ -f "$FEATURE_TARGET_CSV" && -f "$FEATURE_ODOM_CSV" && -f "$FEATURE_POINT_CLOUD_CSV" ]]; then
  python scripts/ros2_publish_odometry.py \
    --csv "$FEATURE_TARGET_CSV" \
    --point-cloud-csv "$FEATURE_POINT_CLOUD_CSV" \
    --point-cloud-source-odom-csv "$FEATURE_ODOM_CSV" \
    --point-cloud-display-mode "${RVIZ_POINT_CLOUD_DISPLAY_MODE:-world-map}" \
    --point-cloud-transform-mode "${RVIZ_POINT_CLOUD_TRANSFORM_MODE:-translate-only}" \
    --point-cloud-local-scale "${RVIZ_POINT_CLOUD_LOCAL_SCALE:-1.0}" \
    --point-cloud-min-track-age "${RVIZ_POINT_CLOUD_MIN_TRACK_AGE:-3}" \
    --point-cloud-min-depth "${RVIZ_POINT_CLOUD_MIN_DEPTH:-0.20}" \
    --point-cloud-max-depth "${RVIZ_POINT_CLOUD_MAX_DEPTH:-8.00}" \
    --point-cloud-max-range "${RVIZ_POINT_CLOUD_MAX_RANGE:-8.00}" \
    --point-cloud-max-path-distance "${RVIZ_POINT_CLOUD_MAX_PATH_DISTANCE:-0.00}" \
    --rate "${RVIZ_FEATURE_RATE:-12}" \
    --loop \
    --zero-start \
    --coordinate-frame ros \
    --orientation-source "${RVIZ_ODOM_ORIENTATION_SOURCE:-tangent}" \
    --odom-topic /feature_cloud_reference_odom \
    --path-topic /feature_cloud_reference_path \
    --vins-odom-topic /feature_cloud_reference_vins_odom \
    --vins-path-topic /feature_cloud_reference_vins_path \
    --feature-history-frames "${RVIZ_FEATURE_HISTORY_FRAMES:-0}" \
    --feature-track-history "${RVIZ_FEATURE_TRACK_HISTORY:-1}" \
    --success-only-cloud \
    > outputs/logs/feature_overlay_publisher.log 2>&1 &
  FEATURE_PID=$!
else
  echo "Skipping feature publisher; missing $FEATURE_TARGET_CSV, $FEATURE_ODOM_CSV, or $FEATURE_POINT_CLOUD_CSV" >&2
  FEATURE_PID=""
fi

if [[ "$HAS_CONTROL" -eq 1 ]]; then
  python scripts/ros2_publish_path_overlay.py \
    --vins-csv "$VINS_CSV" \
    --dvl-csv "$DVL_CSV" \
    --vins-topic /vins_path_overlay \
    --dvl-topic /dvl_reference_path \
    --vins-coordinate-frame "${RVIZ_VINS_COORDINATE_FRAME:-ros}" \
    --dvl-coordinate-frame "${RVIZ_DVL_COORDINATE_FRAME:-flip-y}" \
    --vins-orientation-source "${RVIZ_VINS_ORIENTATION_SOURCE:-tangent}" \
    --dvl-orientation-source "${RVIZ_DVL_ORIENTATION_SOURCE:-tangent}" \
    --control-csv "$CONTROL_CSV" \
    --control-topic /control_best_path \
    --control-coordinate-frame "${RVIZ_CONTROL_COORDINATE_FRAME:-ros}" \
    --control-orientation-source "${RVIZ_CONTROL_ORIENTATION_SOURCE:-tangent}" \
    --success-only-vins \
    --crop-reference-to-vins-time \
    --sample-reference-at-vins-times \
    > outputs/logs/path_overlay_publisher.log 2>&1 &
else
  python scripts/ros2_publish_path_overlay.py \
    --vins-csv "$VINS_CSV" \
    --dvl-csv "$DVL_CSV" \
    --vins-topic /vins_path_overlay \
    --dvl-topic /dvl_reference_path \
    --vins-coordinate-frame "${RVIZ_VINS_COORDINATE_FRAME:-ros}" \
    --dvl-coordinate-frame "${RVIZ_DVL_COORDINATE_FRAME:-flip-y}" \
    --vins-orientation-source "${RVIZ_VINS_ORIENTATION_SOURCE:-tangent}" \
    --dvl-orientation-source "${RVIZ_DVL_ORIENTATION_SOURCE:-tangent}" \
    --success-only-vins \
    --crop-reference-to-vins-time \
    --sample-reference-at-vins-times \
    > outputs/logs/path_overlay_publisher.log 2>&1 &
fi
PATH_PID=$!

cleanup() {
  if [[ -n "${FEATURE_PID:-}" ]]; then
    kill "$FEATURE_PID" >/dev/null 2>&1 || true
  fi
  kill "$PATH_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

rviz2 -d rviz/vins_dvl_overlay.rviz &
RVIZ_PID=$!
wait "$RVIZ_PID"
