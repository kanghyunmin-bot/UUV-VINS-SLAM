#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! docker info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Docker daemon is not running.
Open Docker Desktop first, wait until it says Docker is running, then retry.
EOF
  exit 2
fi

docker build \
  --platform linux/amd64 \
  --build-arg VINS_BUILD_JOBS="${VINS_BUILD_JOBS:-1}" \
  -t underwater-vins-fusion:kinetic \
  -f vins_fusion/docker/Dockerfile.kinetic-fixed \
  third_party/VINS-Fusion
