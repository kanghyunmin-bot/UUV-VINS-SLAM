#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET_DIR="$PROJECT_ROOT/outputs/vins_fusion/underwater_stereo_imu_rosbag"
CONFIG_DIR="$PROJECT_ROOT/outputs/vins_fusion/config_stereo_imu"
OUTPUT_DIR="$PROJECT_ROOT/outputs/vins_fusion/vins_output_stereo_imu"
ODOM_CSV="$PROJECT_ROOT/outputs/vins_fusion/vins_stereo_imu_odometry.csv"
MAX_FRAMES="${1:-120}"

if ! docker info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Docker daemon is not running.

Run this first:

  open -a Docker

Then wait until Docker Desktop says it is running and retry.
EOF
  exit 2
fi

bash -lc "set +u; source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311; python '$PROJECT_ROOT/scripts/export_vins_stereo_imu_rosbag.py' --output-dir '$DATASET_DIR' --max-frames '$MAX_FRAMES' --force"
.venv/bin/python scripts/vins_make_stereo_imu_config.py \
  --metadata "$DATASET_DIR/metadata.json" \
  --config-dir "$CONFIG_DIR" \
  --output-dir "$OUTPUT_DIR"
scripts/vins_docker_build.sh
scripts/vins_docker_run_stereo_imu_rosbag.sh "$DATASET_DIR" "$CONFIG_DIR/underwater_stereo_imu_config.yaml" "$OUTPUT_DIR"
.venv/bin/python scripts/convert_vins_vio_to_csv.py \
  --vio-txt "$OUTPUT_DIR/vio.csv" \
  --times-txt "$DATASET_DIR/times.txt" \
  --output-csv "$ODOM_CSV" \
  --scale-mode vins_fusion_stereo_imu_rosbag_play
mkdir -p outputs/live
cp "$ODOM_CSV" outputs/live/latest_odometry.csv
scripts/start_vins_result_rviz.sh "$ODOM_CSV"
