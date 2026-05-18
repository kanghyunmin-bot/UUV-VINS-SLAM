# VINS-Fusion Integration

This folder contains the local adapter for running the upstream
HKUST-Aerial-Robotics VINS-Fusion code on the underwater MP4 data in this
project.

VINS-Fusion upstream:

- repository: `https://github.com/HKUST-Aerial-Robotics/VINS-Fusion`
- local clone: `third_party/VINS-Fusion`
- pinned clone currently checked out at: `be55a937a57436548ddfb1bd324bc1e9a9e828e0`

## Why Stereo-Only First

The available files are MP4 streams:

- `camera_camera_infra1_image_rect_raw.mp4`
- `camera_camera_infra2_image_rect_raw.mp4`
- `camera_camera_color_image_raw.mp4`
- `camera_camera_depth_image_rect_raw.mp4`

There is no timestamped IMU CSV in this project yet. Because VINS-Fusion
supports stereo-only mode, the first practical integration path is:

```text
infra1/infra2 MP4
  -> KITTI-style stereo image sequence
  -> VINS-Fusion kitti_odom_test
  -> vio.txt
  -> this project's odometry CSV
  -> ROS2 RViz publisher
```

## Prepare Dataset

```bash
cd /Users/kanghyunmin/Desktop/under_water_image_match
.venv/bin/python scripts/export_vins_kitti_stereo.py --force
```

By default this exports the first 300 stereo frames. Use `--max-frames 0` for
the full video.

```bash
.venv/bin/python scripts/export_vins_kitti_stereo.py --max-frames 0 --force
```

## Generate Initial Config

```bash
.venv/bin/python scripts/vins_make_stereo_config.py
```

The generated config is:

```text
outputs/vins_fusion/config/underwater_stereo_config.yaml
```

Important: the default intrinsics and baseline are only rough guesses. Replace
them with real underwater stereo calibration before judging metric accuracy.

## Build VINS-Fusion

VINS-Fusion is ROS1/catkin C++. Build it on Ubuntu with ROS1, not in the current
macOS ROS2 conda environment.

```bash
cd /Users/kanghyunmin/Desktop/under_water_image_match
scripts/vins_prepare_workspace.sh
```

The script uses:

```text
VINS_WS=$HOME/vins_fusion_ws
```

## Mac Docker Option

On macOS, open Docker Desktop first. Then:

```bash
scripts/vins_docker_build.sh
scripts/vins_docker_run_stereo_kitti.sh
```

This uses `--platform linux/amd64` because the upstream VINS-Fusion Dockerfile is
ROS Kinetic based. On Apple Silicon this runs through emulation and can be slow.

The local Dockerfile uses `VINS_BUILD_JOBS=1` by default to avoid Ceres running
out of memory under Docker Desktop emulation. If Docker has more memory assigned,
you can raise it:

```bash
VINS_BUILD_JOBS=2 scripts/vins_docker_build.sh
```

## Run VINS-Fusion Stereo

On Ubuntu ROS1 after building:

```bash
cd /Users/kanghyunmin/Desktop/under_water_image_match
scripts/vins_run_stereo_kitti.sh
```

VINS-Fusion writes:

```text
outputs/vins_fusion/vins_output/vio.txt
```

## Convert Result Back to This Project

```bash
.venv/bin/python scripts/convert_vins_vio_to_csv.py
```

Then publish the converted CSV in ROS2 RViz:

```bash
source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311
python scripts/ros2_publish_odometry.py \
  --csv outputs/vins_fusion/vins_odometry.csv \
  --rate 60 \
  --loop \
  --coordinate-frame ros
```

Or use the helper:

```bash
scripts/start_vins_result_rviz.sh
```

## Next Required Work

For real UUV VIO, this scaffold still needs:

- underwater stereo camera calibration
- real baseline between infra1 and infra2
- timestamped IMU data
- camera-IMU extrinsic
- time offset calibration
- sonar/global correction outside VINS-Fusion
