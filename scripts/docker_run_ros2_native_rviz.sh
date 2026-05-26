#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
PLAY_SECONDS="${PLAY_SECONDS:-30}"
CONTAINER="${CONTAINER:-uuv-vins-ros2-live}"
LOG_DIR_HOST="${LOG_DIR_HOST:-${ROOT}/outputs/docker_native_rviz}"
RVIZ_CONFIG="${RVIZ_CONFIG:-${ROOT}/rviz/uuv_vins_minimal.rviz}"
ROS_CONDA_ENV="${ROS_CONDA_ENV:-ros2_h311}"
ROS_CONDA_SH="${ROS_CONDA_SH:-/Users/kanghyunmin/miniconda3/bin/activate}"
DISCOVERY_PORT="${DISCOVERY_PORT:-11811}"
HOST_DISCOVERY_SERVER="${HOST_DISCOVERY_SERVER:-127.0.0.1:${DISCOVERY_PORT}}"
CONTAINER_DISCOVERY_SERVER="${CONTAINER_DISCOVERY_SERVER:-host.docker.internal:${DISCOVERY_PORT}}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
NATIVE_RVIZ_TRANSPORT="${NATIVE_RVIZ_TRANSPORT:-tcp_bridge}"
ROS_TCP_BRIDGE_PORT="${ROS_TCP_BRIDGE_PORT:-18765}"
MAC_DDS_INTERFACE="${MAC_DDS_INTERFACE:-en0}"
MAC_DDS_ADDRESS="${MAC_DDS_ADDRESS:-$(ipconfig getifaddr "${MAC_DDS_INTERFACE}" 2>/dev/null || true)}"
if [[ -z "${MAC_DDS_ADDRESS}" ]]; then
  MAC_DDS_ADDRESS="${MAC_DDS_ADDRESS:-127.0.0.1}"
fi
HOST_CYCLONEDDS_URI="file://${LOG_DIR_HOST}/cyclonedds_host.xml"
CONTAINER_CYCLONEDDS_URI="file:///workspace/under_water_image_match/outputs/docker_native_rviz/cyclonedds_container.xml"

mkdir -p "${LOG_DIR_HOST}"

if command -v screen >/dev/null 2>&1; then
  screen -S uuv_fastdds_discovery -X quit >/dev/null 2>&1 || true
  screen -S uuv_native_rviz -X quit >/dev/null 2>&1 || true
  screen -S uuv_tcp_bridge_client -X quit >/dev/null 2>&1 || true
  screen -S uuv_vins_loop -X quit >/dev/null 2>&1 || true
fi

pkill -f "fastdds discovery -i 0.*${DISCOVERY_PORT}" >/dev/null 2>&1 || true
pkill -f "ros2_tcp_topic_bridge_client.py --host 127.0.0.1 --port ${ROS_TCP_BRIDGE_PORT}" >/dev/null 2>&1 || true
pkill -9 -f "ros2_tcp_topic_bridge_client.py" >/dev/null 2>&1 || true
pkill -9 -f "rviz2 -d ${RVIZ_CONFIG}" >/dev/null 2>&1 || true
pkill -9 -f "${ROOT}/scripts/run_rviz_vins_loop.zsh" >/dev/null 2>&1 || true
pkill -9 -f "${ROOT}/VINS-Fusion-ROS2/install/vins/lib/vins/vins_node" >/dev/null 2>&1 || true
pkill -9 -f "${ROOT}/scripts/ros2_publish_stereo_mp4.py" >/dev/null 2>&1 || true
pkill -9 -f "${ROOT}/scripts/ros2_publish_static_path.py" >/dev/null 2>&1 || true

cat > "${LOG_DIR_HOST}/cyclonedds_host.xml" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain id="any">
    <General>
      <NetworkInterfaceAddress>${MAC_DDS_INTERFACE}</NetworkInterfaceAddress>
      <AllowMulticast>false</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
EOF

cat > "${LOG_DIR_HOST}/cyclonedds_container.xml" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain id="any">
    <General>
      <NetworkInterfaceAddress>eth0</NetworkInterfaceAddress>
      <AllowMulticast>true</AllowMulticast>
    </General>
    <Discovery>
      <Peers>
        <Peer Address="localhost"/>
        <Peer Address="192.168.65.3"/>
        <Peer Address="${MAC_DDS_ADDRESS}"/>
        <Peer Address="host.docker.internal"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
EOF

if [[ "${RMW_IMPLEMENTATION}" == "rmw_fastrtps_cpp" ]]; then
  cat > "${LOG_DIR_HOST}/start_fastdds_discovery.zsh" <<EOF
#!/usr/bin/env zsh
set -e
set +u
source "${ROS_CONDA_SH}" "${ROS_CONDA_ENV}"
set -u
exec fastdds discovery -i 0 -l 0.0.0.0 -p "${DISCOVERY_PORT}"
EOF
  chmod +x "${LOG_DIR_HOST}/start_fastdds_discovery.zsh"

  if command -v screen >/dev/null 2>&1; then
    screen -dmS uuv_fastdds_discovery zsh -lc "'${LOG_DIR_HOST}/start_fastdds_discovery.zsh' > '${LOG_DIR_HOST}/fastdds_discovery.log' 2>&1"
  else
    zsh -lc "'${LOG_DIR_HOST}/start_fastdds_discovery.zsh' > '${LOG_DIR_HOST}/fastdds_discovery.log' 2>&1" &
  fi

  sleep 1
fi

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

docker_env=(
  DETACH=1
  RUN_RVIZ=0
  RVIZ_BACKEND=none
  CONFIG="${CONFIG:-}"
  BUILD_ON_START="${BUILD_ON_START:-1}"
  RUN_ONCE="${RUN_ONCE:-0}"
  RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}"
  ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"
  PLAY_SECONDS="${PLAY_SECONDS}"
  PLAY_RATE="${PLAY_RATE:-1.0}"
  ROSBAG_INPUT_DIR="${ROSBAG_INPUT_DIR:-/workspace/under_water_image_match/data/rosbag_active/localization bag}"
  IMAGE_SOURCE_TYPE="${IMAGE_SOURCE_TYPE:-mp4}"
  STAMP_TIME_SOURCE="${STAMP_TIME_SOURCE:-}"
  STEREO_SYNC_TOLERANCE_MS="${STEREO_SYNC_TOLERANCE_MS:-3}"
  STEREO_SCHEDULE_SOURCE_CSV="${STEREO_SCHEDULE_SOURCE_CSV:-}"
  SWAP_STEREO_MP4="${SWAP_STEREO_MP4:-0}"
  RESTAMP_NOW="${RESTAMP_NOW:-1}"
  IMU_RESAMPLE_HZ="${IMU_RESAMPLE_HZ:-100}"
  IMU_SOURCE_TOPIC="${IMU_SOURCE_TOPIC-/mavros/imu/data}"
  IMU_SOURCE_TYPE="${IMU_SOURCE_TYPE:-sensor_msgs/msg/Imu}"
  IMU_OUTPUT_TOPIC="${IMU_OUTPUT_TOPIC:-/vins/imu/data}"
  IMU_FRAME_ID="${IMU_FRAME_ID:-fcu_link}"
  IMU_FRAME_TRANSFORM="${IMU_FRAME_TRANSFORM:-identity}"
  UNIFORM_STEREO_TIMESTAMPS="${UNIFORM_STEREO_TIMESTAMPS:-0}"
  PUBLISH_PIXHAWK_CAMERA_TF="${PUBLISH_PIXHAWK_CAMERA_TF:-1}"
  RUN_EKF_LOCALIZATION="${RUN_EKF_LOCALIZATION:-0}"
  RUN_BAG_LOCALIZATION_PATH="${RUN_BAG_LOCALIZATION_PATH:-1}"
  RUN_TF_LOCALIZED_PATH="${RUN_TF_LOCALIZED_PATH:-0}"
  SYNC_REPLAY_PATHS="${SYNC_REPLAY_PATHS:-1}"
  DVL_REPLAY_HZ="${DVL_REPLAY_HZ:-10.0}"
  LOCALIZATION_ALIGN_MODE="${LOCALIZATION_ALIGN_MODE:-none}"
  LOCALIZATION_ALIGN_YAW_DEG="${LOCALIZATION_ALIGN_YAW_DEG:-0.0}"
  DVL_COORDINATE_FRAME="${DVL_COORDINATE_FRAME:-flip-y}"
  DVL_REFERENCE_ALIGN_MODE="${DVL_REFERENCE_ALIGN_MODE:-none}"
  LOCALIZATION_REFERENCE_CSV="${LOCALIZATION_REFERENCE_CSV:-}"
  VINS_DRAIN_TIMEOUT_SEC="${VINS_DRAIN_TIMEOUT_SEC:-25}"
  VINS_DRAIN_IDLE_SEC="${VINS_DRAIN_IDLE_SEC:-5}"
  LOG_DIR="/workspace/under_water_image_match/outputs/docker_native_rviz/container"
)
if [[ "${NATIVE_RVIZ_TRANSPORT}" == "direct_dds" ]]; then
  docker_env+=(DOCKER_NETWORK_MODE=host)
fi
if [[ "${NATIVE_RVIZ_TRANSPORT}" == "tcp_bridge" ]]; then
  docker_env+=(ROS_TCP_BRIDGE_PORT="${ROS_TCP_BRIDGE_PORT}")
elif [[ "${RMW_IMPLEMENTATION}" == "rmw_cyclonedds_cpp" ]]; then
  docker_env+=(CYCLONEDDS_URI="${CONTAINER_CYCLONEDDS_URI}")
elif [[ "${RMW_IMPLEMENTATION}" == "rmw_fastrtps_cpp" ]]; then
  docker_env+=(ROS_DISCOVERY_SERVER="${CONTAINER_DISCOVERY_SERVER}")
fi

env "${docker_env[@]}" "${ROOT}/scripts/docker_run_ros2_rviz_live.sh"

if [[ "${NATIVE_RVIZ_TRANSPORT}" == "tcp_bridge" ]]; then
  docker exec -d "${CONTAINER}" bash -lc "
    set +u
    source /opt/ros/humble/setup.bash
    source /workspace/under_water_image_match/VINS-Fusion-ROS2/install_docker/setup.bash
    set -u
    export ROS_DOMAIN_ID='${ROS_DOMAIN_ID}'
    export ROS_LOCALHOST_ONLY=0
    export RMW_IMPLEMENTATION='${RMW_IMPLEMENTATION}'
    python3 /workspace/under_water_image_match/scripts/ros2_tcp_topic_bridge_server.py \
      --port '${ROS_TCP_BRIDGE_PORT}' \
      --topic /dvl_reference_path:nav_msgs/msg/Path \
      --topic /path:nav_msgs/msg/Path \
      --topic /odometry:nav_msgs/msg/Odometry \
      --topic /odometry/filtered:nav_msgs/msg/Odometry \
      --topic /localization/path:nav_msgs/msg/Path \
      --topic /point_cloud:sensor_msgs/msg/PointCloud \
      --topic /tf_static:tf2_msgs/msg/TFMessage \
      --topic /clock:rosgraph_msgs/msg/Clock \
      --topic /vins_estimator/image_track:sensor_msgs/msg/Image \
      > /workspace/under_water_image_match/outputs/docker_native_rviz/container/tcp_bridge_server.log 2>&1
  "
fi

cat > "${LOG_DIR_HOST}/start_native_rviz.zsh" <<EOF
#!/usr/bin/env zsh
set -e
cd "${ROOT}"
set +u
source "${ROS_CONDA_SH}" "${ROS_CONDA_ENV}"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"
export ROS_LOCALHOST_ONLY=$([[ "${NATIVE_RVIZ_TRANSPORT}" == "tcp_bridge" ]] && echo 1 || echo 0)
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}"
if [[ "${NATIVE_RVIZ_TRANSPORT}" != "tcp_bridge" ]]; then
  export CYCLONEDDS_URI="${HOST_CYCLONEDDS_URI}"
  if [[ "${RMW_IMPLEMENTATION}" == "rmw_fastrtps_cpp" ]]; then
    export ROS_DISCOVERY_SERVER="${HOST_DISCOVERY_SERVER}"
  fi
fi
export DYLD_LIBRARY_PATH="/Users/kanghyunmin/miniconda3/envs/${ROS_CONDA_ENV}/lib:/Users/kanghyunmin/miniconda3/envs/${ROS_CONDA_ENV}/opt/rviz_ogre_vendor/lib:\${DYLD_LIBRARY_PATH:-}"
exec rviz2 -d "${RVIZ_CONFIG}"
EOF
chmod +x "${LOG_DIR_HOST}/start_native_rviz.zsh"

if [[ "${NATIVE_RVIZ_TRANSPORT}" == "tcp_bridge" ]]; then
  cat > "${LOG_DIR_HOST}/start_tcp_bridge_client.zsh" <<EOF
#!/usr/bin/env zsh
set -e
cd "${ROOT}"
set +u
source "${ROS_CONDA_SH}" "${ROS_CONDA_ENV}"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}"
python "${ROOT}/scripts/ros2_tcp_topic_bridge_client.py" --host 127.0.0.1 --port "${ROS_TCP_BRIDGE_PORT}"
EOF
  chmod +x "${LOG_DIR_HOST}/start_tcp_bridge_client.zsh"
  if command -v screen >/dev/null 2>&1; then
    screen -dmS uuv_tcp_bridge_client zsh -lc "'${LOG_DIR_HOST}/start_tcp_bridge_client.zsh' > '${LOG_DIR_HOST}/tcp_bridge_client.log' 2>&1"
  else
    zsh -lc "'${LOG_DIR_HOST}/start_tcp_bridge_client.zsh' > '${LOG_DIR_HOST}/tcp_bridge_client.log' 2>&1" &
  fi
fi

if command -v screen >/dev/null 2>&1; then
  screen -dmS uuv_native_rviz zsh -lc "'${LOG_DIR_HOST}/start_native_rviz.zsh' > '${LOG_DIR_HOST}/rviz.log' 2>&1"
  echo "Started Mac native RViz in screen session: uuv_native_rviz"
else
  zsh -lc "'${LOG_DIR_HOST}/start_native_rviz.zsh' > '${LOG_DIR_HOST}/rviz.log' 2>&1" &
  echo "Started Mac native RViz in background pid=$!"
fi

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "PLAY_RATE=${PLAY_RATE:-1.0}"
echo "IMAGE_SOURCE_TYPE=${IMAGE_SOURCE_TYPE:-mp4}"
echo "STAMP_TIME_SOURCE=${STAMP_TIME_SOURCE:-manifest/default}"
echo "SWAP_STEREO_MP4=${SWAP_STEREO_MP4:-0}"
echo "IMU_RESAMPLE_HZ=${IMU_RESAMPLE_HZ:-100}"
echo "IMU_SOURCE_TOPIC=${IMU_SOURCE_TOPIC-/mavros/imu/data}"
echo "IMU_SOURCE_TYPE=${IMU_SOURCE_TYPE:-sensor_msgs/msg/Imu}"
echo "IMU_OUTPUT_TOPIC=${IMU_OUTPUT_TOPIC:-/vins/imu/data}"
echo "ROSBAG_INPUT_DIR=${ROSBAG_INPUT_DIR:-/workspace/under_water_image_match/data/rosbag_active/localization bag}"
echo "DVL_COORDINATE_FRAME=${DVL_COORDINATE_FRAME:-flip-y}"
echo "DVL_REFERENCE_ALIGN_MODE=${DVL_REFERENCE_ALIGN_MODE:-none}"
echo "SYNC_REPLAY_PATHS=${SYNC_REPLAY_PATHS:-1}"
echo "RUN_ONCE=${RUN_ONCE:-0}"
echo "Native RViz transport=${NATIVE_RVIZ_TRANSPORT}"
if [[ "${NATIVE_RVIZ_TRANSPORT}" == "tcp_bridge" ]]; then
  echo "TCP bridge=127.0.0.1:${ROS_TCP_BRIDGE_PORT}"
fi
echo "Mac DDS interface=${MAC_DDS_INTERFACE} ${MAC_DDS_ADDRESS}"
echo "Host CYCLONEDDS_URI=${HOST_CYCLONEDDS_URI}"
echo "Container CYCLONEDDS_URI=${CONTAINER_CYCLONEDDS_URI}"
if [[ "${RMW_IMPLEMENTATION}" == "rmw_fastrtps_cpp" ]]; then
  echo "Host ROS_DISCOVERY_SERVER=${HOST_DISCOVERY_SERVER}"
  echo "Container ROS_DISCOVERY_SERVER=${CONTAINER_DISCOVERY_SERVER}"
  echo "Fast DDS discovery log=${LOG_DIR_HOST}/fastdds_discovery.log"
fi
echo "Container=${CONTAINER}"
echo "Container logs=${LOG_DIR_HOST}/container"
echo "RViz log=${LOG_DIR_HOST}/rviz.log"

sleep 3
check_topic="/path"
if zsh -lc "set +u; source '${ROS_CONDA_SH}' '${ROS_CONDA_ENV}'; set -u; export ROS_DOMAIN_ID='${ROS_DOMAIN_ID}'; export ROS_LOCALHOST_ONLY=$([[ "${NATIVE_RVIZ_TRANSPORT}" == "tcp_bridge" ]] && echo 1 || echo 0); export RMW_IMPLEMENTATION='${RMW_IMPLEMENTATION}'; if [[ '${NATIVE_RVIZ_TRANSPORT}' != 'tcp_bridge' ]]; then export CYCLONEDDS_URI='${HOST_CYCLONEDDS_URI}'; fi; timeout 5 ros2 topic list --no-daemon" \
  2>"${LOG_DIR_HOST}/native_topic_check.err" | tee "${LOG_DIR_HOST}/native_topic_check.log" | grep -q "^${check_topic}$"; then
  echo "Native RViz DDS check: OK, ${check_topic} is visible from macOS."
else
  cat >&2 <<EOF
Native RViz DDS check: ${check_topic} is not visible from macOS.
Check ${LOG_DIR_HOST}/tcp_bridge_client.log and
${LOG_DIR_HOST}/container/tcp_bridge_server.log.
EOF
fi
