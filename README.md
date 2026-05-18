# Underwater VIO-SLAM Frontend

이 프로젝트는 MP4 영상에서 VIO-SLAM backend에 넘길 수 있는 visual measurement를 만드는 frontend입니다. 현재 입력이 MP4뿐이면 진짜 VIO 전체 파이프라인은 아니지만, 정석적인 VIO-SLAM의 앞단 구조에 맞춰 `frame source -> feature detection -> KLT tracking -> geometric verification -> track log`로 분리되어 있습니다.

## 구조

```text
frontend_gui.py                # parameter GUI
track_features.py              # CLI entrypoint
configs/mp4_frontend.json      # MP4-only 실행 설정
scripts/ros2_publish_odometry.py
scripts/export_vins_kitti_stereo.py
scripts/vins_make_stereo_config.py
scripts/convert_vins_vio_to_csv.py
vins_fusion/                  # VINS-Fusion adapter docs/templates
third_party/VINS-Fusion/      # upstream VINS-Fusion clone
rviz/vio_frontend.rviz
vio_frontend/
  config.py                    # dataset, camera, IMU, tracker, RANSAC 설정
  dataset.py                   # video frame/timestamp source
  feature_tracker.py           # Shi-Tomasi feature + KLT optical flow
  geometry.py                  # Essential/Fundamental matrix RANSAC
  odometry.py                  # pseudo local visual odometry for RViz
  imu.py                       # future IMU CSV loader
  outputs.py                   # CSV, summary JSON, debug video
  pipeline.py                  # frontend orchestration
```

## 설치

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

시스템 Python에 OpenCV가 없어도 괜찮습니다. 위처럼 만든 `.venv` 안에는 `opencv-python`이 설치됩니다.

## 실행

설정 파일 기준:

```bash
.venv/bin/python track_features.py --config configs/mp4_frontend.json
```

기존처럼 MP4를 직접 넘겨도 됩니다.

```bash
.venv/bin/python track_features.py video_src/camera_camera_color_image_raw.mp4 --clahe --draw-rejected
```

빠른 테스트:

```bash
.venv/bin/python track_features.py --config configs/mp4_frontend.json --max-frames 200 --draw-rejected
```

## GUI로 파라미터 실험

브라우저 GUI에서 주요 파라미터와 feature 방식을 바꿔가며 단일 실행 또는 sweep을 돌릴 수 있습니다.

```bash
./scripts/start_web_gui.sh
```

실행 후 브라우저에서 표시된 주소를 엽니다. 기본 주소:

```text
http://127.0.0.1:8765
```

지원하는 feature 방식:

- `Shi-Tomasi + KLT`: 가까운 연속 프레임 tracking용 기본 방식
- `ORB descriptor`: 빠른 binary descriptor matching
- `SIFT descriptor`: scale/rotation에 강한 descriptor matching
- `AKAZE descriptor`: binary descriptor 기반 matching

GUI에는 프리셋도 포함되어 있습니다.

- `Pool balanced KLT`
- `Pool strict KLT`
- `Fast debug KLT`
- `ORB loose matching`
- `SIFT loose matching`
- `AKAZE loose matching`

`Run Selected`가 끝나면 오른쪽 패널에서 결과 MP4가 바로 재생됩니다. 브라우저 재생을 위해 결과 영상은 H.264 웹용 MP4로 한 번 더 변환됩니다.

수동 실행도 가능합니다: `.venv/bin/python web_gui.py`

GUI에서 조정할 수 있는 값:

- feature 수: `max_features`, `min_features`
- feature 간격: `min_distance`
- KLT 검증: `forward_backward_threshold`
- RANSAC 검증: `ransac_threshold`
- visual odometry: `assumed_focal_scale`, `motion_scale`, `pose_min_inliers`
- 출력: debug video, rejected track 표시, odometry CSV

Sweep은 `max_features`, `min_distance`, `ransac_threshold` 조합을 여러 개 실행하고 `sweep_results.csv`로 비교합니다.

## 출력

결과는 기본적으로 `outputs/`에 저장됩니다.

- `*_tracked.mp4`: feature tracking debug video
- `*_tracks.csv`: frame별 track 좌표, forward-backward error, geometry inlier 여부
- `*_odometry.csv`: RViz에서 볼 pseudo visual odometry pose
- `*_summary.json`: 평균 active track 수, RANSAC inlier ratio, 사용된 geometry model

영상 색상:

- 초록색: geometric verification을 통과한 inlier track
- 빨간색: RANSAC에서 제거된 track (`--draw-rejected` 사용 시)
- 파란색: 새로 검출된 feature

## VIO-SLAM 관점

현재 MP4-only 설정에서는 카메라 intrinsic이 없으므로 `Fundamental matrix + RANSAC`을 사용합니다. `configs/mp4_frontend.json`의 `camera` 값에 `fx`, `fy`, `cx`, `cy`, `distortion`을 넣으면 `geometry.model: "auto"`가 `Essential matrix + RANSAC`으로 전환됩니다.

기본값은 VIO frontend에 맞춰 다소 엄격하게 잡았습니다.

- `max_features: 200`
- `min_distance: 30`
- `forward_backward_threshold: 0.8`
- `ransac_threshold: 0.75`

수영장 타일처럼 반복 패턴이 강한 영상에서는 feature를 많이 잡는 것보다 적게 잡더라도 넓게 분포시키고, RANSAC threshold를 낮게 유지하는 편이 낫습니다.

## RViz에서 Odom / Local Path 확인

MP4-only visual odometry는 절대 scale이 없습니다. 현재 `*_odometry.csv`는 Essential matrix `recoverPose` 기반의 **임의 scale local path**입니다. 실제 거리/속도 검증용이 아니라, RANSAC/feature 설정에 따라 local motion이 얼마나 안정적으로 보이는지 RViz에서 확인하기 위한 출력입니다.

좌표계 기준:

- `*_odometry.csv`: OpenCV optical frame 기준입니다. `x=right`, `y=down`, `z=forward`
- RViz publish: ROS body/map 기준으로 변환합니다. `x=forward`, `y=left`, `z=up`
- 변환식: `x_ros=z_cv`, `y_ros=-x_cv`, `z_ros=-y_cv`

ROS2는 conda 환경에 설치되어 있습니다. 이 환경을 먼저 활성화합니다.

```bash
source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311
```

RViz와 live odometry/path publisher를 같이 실행:

```bash
./scripts/start_rviz_live.sh
```

기본값은 60Hz frame-by-frame 재생입니다. 현재 3100프레임 VINS CSV 기준으로 약 52초에 한 바퀴 돕니다. 더 빠르게 보려면:

```bash
RVIZ_ODOM_RATE=120 ./scripts/start_rviz_live.sh
```

이 상태에서 웹 GUI의 `Run Selected`를 누르면 먼저 RViz path가 원점으로 초기화되고, 처리가 끝난 뒤 `outputs/live/latest_odometry.csv`가 최종 결과로 교체됩니다. publisher는 이 CSV 변경을 감지해서 새 path를 RViz에 반영합니다.

수동으로 나눠서 실행하려면 1번 터미널:

```bash
source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311
cd /Users/kanghyunmin/Desktop/under_water_image_match
python scripts/ros2_publish_odometry.py --csv outputs/live/latest_odometry.csv --rate 60 --loop --watch
```

원본 OpenCV 좌표계를 그대로 보고 싶으면 publisher에 `--coordinate-frame opencv`를 추가합니다.

AQUA-SLAM처럼 acoustic/velocity 계열 정보를 참고해 VINS 경로를 더 현실적인 metric path로 보정하려면 웹 GUI의 `AQUA-style Metric Correction`을 사용합니다. 기본값은 원본 VINS stereo CSV를 `/odometry/filtered` 기준 거리로 맞추고 큰 frame jump를 잘라낸 뒤 `outputs/live/latest_odometry.csv`를 교체합니다.

수동 실행 예:

```bash
python scripts/aqua_metric_correct_odometry.py \
  --input-csv outputs/vins_fusion/gui_runs/run_20260507_205427_901/vins_odometry.csv \
  --output-csv outputs/live/latest_odometry_aqua_corrected.csv \
  --source reference \
  --max-step-m 0.3
cp outputs/live/latest_odometry_aqua_corrected.csv outputs/live/latest_odometry.csv
```

DVL만 적분해서 길이를 맞추려면 ROS2 환경에서 `--source dvl`을 쓸 수 있지만, 현재 bag에서는 DVL 단독 적분 거리와 EKF reference 거리가 크게 달라 기본값으로 쓰지 않습니다.

2번 터미널:

```bash
source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311
cd /Users/kanghyunmin/Desktop/under_water_image_match
rviz2 -d rviz/vio_frontend.rviz
```

프로세스 정리:

```bash
./scripts/stop_live.sh
```

publish되는 topic:

- `/visual_odom`: `nav_msgs/Odometry`
- `/local_path`: `nav_msgs/Path`
- `/tf`: `map -> camera_link`
- `/vins_estimator/odometry`: `nav_msgs/Odometry`, VINS-Fusion 호환 odometry alias
- `/vins_estimator/path`: `nav_msgs/Path`, VINS-Fusion 호환 path alias
- `/vins_estimator/point_cloud`: `sensor_msgs/PointCloud2`, stereo triangulation으로 얻은 sparse landmark cloud

토픽 확인:

```bash
ros2 topic list
ros2 topic echo /visual_odom --once
ros2 topic echo /vins_estimator/point_cloud --once
```

`scripts/ros2_stereo_imu_vio.py`는 odometry CSV와 함께 `outputs/live/latest_point_cloud.csv`를 생성합니다. 이 파일은 frame별 active track의 world 좌표를 담고, `scripts/ros2_publish_odometry.py`가 같은 frame index에 맞춰 `PointCloud2`로 publish합니다. 좌표 변환은 odometry와 동일하게 기본값에서 OpenCV optical frame을 RViz용 ROS frame으로 바꿉니다.

완전한 VIO-SLAM을 하려면 MP4 외에 다음이 필요합니다.

- 카메라 intrinsic/distortion
- IMU gyro/accel timestamped data
- camera-IMU extrinsic
- camera-IMU time offset
- IMU preintegration
- sliding-window backend optimizer

`vio_frontend/imu.py`는 IMU CSV 입력을 받을 수 있는 자리만 만들어둔 상태입니다. backend estimator는 아직 구현하지 않았습니다.

## VINS-Fusion으로 다시 실행

VINS-Fusion 원본은 `third_party/VINS-Fusion`에 받아두었습니다. 원본은 ROS1/catkin 기반 C++ 프로젝트라 현재 macOS ROS2 conda 환경에서 바로 빌드되지는 않습니다. 이 프로젝트에서는 먼저 infra stereo MP4를 VINS-Fusion의 KITTI stereo runner가 읽을 수 있게 변환합니다.

1번: infra1/infra2 MP4를 KITTI-style stereo folder로 변환:

```bash
.venv/bin/python scripts/export_vins_kitti_stereo.py --force
```

전체 영상으로 만들려면:

```bash
.venv/bin/python scripts/export_vins_kitti_stereo.py --max-frames 0 --force
```

2번: VINS-Fusion stereo-only config 생성:

```bash
.venv/bin/python scripts/vins_make_stereo_config.py
```

3번: Ubuntu ROS1 환경에서 VINS-Fusion 빌드:

```bash
scripts/vins_prepare_workspace.sh
```

Mac에서 Docker로 시도하려면 Docker Desktop을 켠 뒤:

```bash
scripts/vins_docker_build.sh
scripts/vins_docker_run_stereo_kitti.sh
```

Apple Silicon에서는 `linux/amd64` emulation이라 느릴 수 있습니다.

4번: VINS-Fusion stereo odometry 실행:

```bash
scripts/vins_run_stereo_kitti.sh
```

5번: VINS-Fusion `vio.txt`를 이 프로젝트 RViz publisher용 CSV로 변환:

```bash
.venv/bin/python scripts/convert_vins_vio_to_csv.py
```

변환된 결과를 RViz에서 보려면:

```bash
scripts/start_vins_result_rviz.sh
```

자세한 내용은 `vins_fusion/README.md`를 봅니다. 현재 config의 카메라 intrinsic과 stereo baseline은 rough guess입니다. 실제 대회용으로 쓰려면 underwater stereo calibration 값으로 반드시 교체해야 합니다.
