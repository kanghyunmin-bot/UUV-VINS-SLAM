# Underwater VINS-Fusion ROS2 Workspace

This root workspace is now reserved for the active pipeline:

1. Read the active stereo/DVL/IMU input bundle from `data/rosbag_active/`.
2. Run the modified ROS2 VINS-Fusion stack in `VINS-Fusion-ROS2/`.
3. Publish native Mac RViz topics through the Docker bridge.
4. Score VINS output against DVL as an offline reference only.

Older work is separated:

- `archive/gui_discarded/`: abandoned GUI feature-test phase.
- `my-vins/`: custom VIO/VINS-reference implementation kept for reference.
- `archive/`: old packages, runtime caches, stray files, and legacy repos.

See `WORKSPACE_LAYOUT.md` for the detailed folder contract.

## Main Commands

Prepare or refresh the active input bundle:

```bash
scripts/prepare_rosbag_active_inputs.sh
```

Start the current Docker + native RViz pipeline:

```bash
ROS_DOMAIN_ID=42 PLAY_SECONDS=30 ./scripts/docker_run_ros2_native_rviz.sh
```

The active input folder is:

```text
data/rosbag_active/localization bag/
```

It contains:

- `stamp_bag.db3`
- `metadata.yaml`
- `stereo_schedule_34_85s.csv`
- `dvl_reference_34_85s.csv`
- `localization_odometry_34_85s.csv`
- `imu_34_85s.csv`
- `infra1_image_rect_raw.mp4`
- `infra2_image_rect_raw.mp4`

Do not point live scripts directly at scattered `outputs/*` or `video_src/*` files. Refresh `data/rosbag_active/` instead.
