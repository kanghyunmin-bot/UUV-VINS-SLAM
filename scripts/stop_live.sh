#!/usr/bin/env bash
set -euo pipefail

pkill -f "web_gui.py" || true
pkill -f "ros2_publish_odometry.py" || true
