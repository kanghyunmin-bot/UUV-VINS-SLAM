# Standard ROS2 VINS Pipeline

This project keeps VINS-Fusion on the normal ROS2 topic contract.

## Contract

VINS-Fusion subscribes only to standard sensor topics:

```yaml
image0_topic: "/camera/camera/infra1/image_rect_raw"
image1_topic: "/camera/camera/infra2/image_rect_raw"
imu_topic: "/vins/imu/data"
```

The estimator must not read MP4 files, DVL CSV files, localization CSV files,
or runtime yaw/sign correction variables.

## Data Flow

```text
infra1_image_rect_raw.mp4
infra2_image_rect_raw.mp4
IMU source from bag
        |
        v
scripts/ros2_publish_stereo_mp4.py
        |
        |  sensor_msgs/msg/Image
        |  sensor_msgs/msg/Image
        |  sensor_msgs/msg/Imu
        v
VINS-Fusion ROS2 vins_node
        |
        |  /odometry
        |  /path
        |  /point_cloud
        |  /scan
        v
RViz and offline overlay/scoring
```

## Boundary

- VINS estimator input: stereo image topics and IMU topic only.
- DVL reference: external RViz path and offline scoring only.
- Localization reference: external RViz path and offline scoring only.
- MP4 replay: external dataset adapter only, not a VINS package executable.
- Axis/yaw/sign comparison: external overlay tools only.

## Runtime

Use the live script as a ROS2 replay wrapper:

```bash
./scripts/docker_run_ros2_native_rviz.sh
```

The wrapper publishes MP4 frames and IMU to the standard topics above, then
starts `vins_node` with the selected YAML. It does not rewrite the YAML at
runtime and does not inject DVL/reference yaw alignment into VINS.

## Evaluation

After a run, compare raw VINS output against references outside the estimator:

```bash
python scripts/overlay_vins_dvl_from_logs.py \
  --vins-log outputs/ros2_vins_fusion_live/last_completed_vio.csv \
  --dvl-csv "data/rosbag_active/localization bag/dvl_reference_34_85s.csv" \
  --dvl-coordinate-frame flip-y \
  --summary-json outputs/evaluation/current_summary.json \
  --overlay-csv outputs/evaluation/current_overlay.csv \
  --plot-png outputs/evaluation/current_overlay.png \
  --single-xy-plot-png outputs/evaluation/current_overlay_xy.png
```

Yaw sweeps may be used only as diagnostics. They are not estimator output.
