#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-uuv-vins-ros2:humble}"

if ! docker info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Docker daemon is not running.
Open Docker Desktop first, wait until Docker is running, then retry.
EOF
  exit 2
fi

platform_args=()
if [[ -n "${DOCKER_PLATFORM:-}" ]]; then
  platform_args+=(--platform "${DOCKER_PLATFORM}")
fi

docker build \
  "${platform_args[@]+"${platform_args[@]}"}" \
  -t "${IMAGE}" \
  -f "${ROOT}/docker/Dockerfile.ros2-humble" \
  "${ROOT}"
