#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMPORT_DIR="${IMPORT_DIR:-${ROOT}/data/rosbag_imports/bag_2026-04-02_21-46-20}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/data/rosbag_active/localization bag}"
START_OFFSET_SEC="${START_OFFSET_SEC:-34}"
DURATION_SEC="${DURATION_SEC:-51}"
OUTPUT_TAG="${OUTPUT_TAG:-34_85s}"
TIME_SOURCE="${TIME_SOURCE:-header}"
SYNC_TOLERANCE_MS="${SYNC_TOLERANCE_MS:-3}"

python "${ROOT}/scripts/extract_rosbag_active_bundle.py" \
  --import-dir "${IMPORT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --time-source "${TIME_SOURCE}" \
  --sync-tolerance-ms "${SYNC_TOLERANCE_MS}" \
  --start-offset-sec "${START_OFFSET_SEC}" \
  --duration-sec "${DURATION_SEC}" \
  --output-tag "${OUTPUT_TAG}" \
  --force-output
