#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! docker info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Docker daemon is not running.

Run this first:

  open -a Docker

Then wait until Docker Desktop says it is running and retry:

  scripts/run_vins_docker_pipeline.sh
EOF
  exit 2
fi

.venv/bin/python scripts/export_vins_kitti_stereo.py --max-frames 0 --force
.venv/bin/python scripts/vins_make_stereo_config.py
scripts/vins_docker_build.sh
scripts/vins_docker_run_stereo_kitti.sh
.venv/bin/python scripts/convert_vins_vio_to_csv.py
scripts/start_vins_result_rviz.sh
