# Workspace Layout

This workspace is organized around three project phases.

1. `archive/gui_discarded/`: GUI feature-test phase. Preserved, but not active.
2. `my-vins/`: custom VIO/VINS-reference implementation. Preserved as a separate reference project.
3. Root workspace: active ROS2 VINS-Fusion modification and real-time RViz pipeline.

## Active Runtime Paths

- `VINS-Fusion-ROS2/`: active ROS2 VINS-Fusion source, configs, build, install, and logs.
- `scripts/`: only current launch, publisher, bridge, evaluation, and input-bundle scripts for the active path.
- `config/`: EKF and runtime YAML outside the VINS tree.
- `rviz/`: RViz display presets.
- `docker/`: Docker build context.
- `data/rosbag_active/localization bag/`: single input bundle used by the current pipeline.
  - `stamp_bag.db3`
  - `metadata.yaml`
  - `stereo_schedule_34_85s.csv`
  - `dvl_reference_34_85s.csv`
  - `localization_odometry_34_85s.csv`
  - `imu_34_85s.csv`
  - `infra1_image_rect_raw.mp4`
  - `infra2_image_rect_raw.mp4`
- `data/rosbag_imports/`: imported source bag database cache used by the extractor.
- `outputs/evaluation/`: scoring and overlay results.
- `outputs/docker_native_rviz/`: native Mac RViz bridge/run logs.
- `outputs/ros2_vins_fusion_live/`: current VINS live CSV/log products.

## Preserved But Not Active

- `archive/gui_discarded/`: old GUI source, GUI scripts, and GUI/Tk/web outputs from the first phase.
- `archive/legacy_repos/`: old root-level repository clones that are not part of the active ROS2 run.
- `archive/packages/`: old zip/package snapshots.
- `archive/stray_files/`: accidental root-level scratch files preserved for later cleanup.
- `archive/runtime_cache/`: moved Python cache folders.
- `archive/macos_metadata/`: moved macOS metadata files.
- `outputs/archive/legacy_runs/`: older non-GUI smoke/docker correction runs.
- `my-vins/`: second-phase custom VIO/VINS-reference work.
  - `my-vins/source/`
  - `my-vins/scripts/`
  - `my-vins/configs/`
  - `my-vins/outputs/`
  - `my-vins/reference/`

## Documentation And References

- `docs/reports/`: generated reports and review PDFs.
- `docs/reference_papers/`: paper PDFs used as references.
- `docs/course_materials/`: course/reference lecture PDFs.
- `my-vins/reference/third_party/VINS-Fusion/`: upstream ROS1 VINS-Fusion reference retained for the custom/reference phase.
- `my-vins/reference/vins_fusion_adapter/`: adapter docs/templates for ROS1 VINS-Fusion workflows.

## Rule

Do not point live scripts directly at scattered `outputs/*` or `video_src/*` paths. Put the intended input set into `data/rosbag_active/localization bag/` with `scripts/prepare_rosbag_active_inputs.sh`, then run the Docker/RViz scripts from that active bundle.
