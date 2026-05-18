%YAML:1.0

# VINS-Fusion stereo+IMU config generated from the original ROS2 bag.
# This is the correct VIO path for the underwater UUV experiment:
# synchronized stereo images + IMU measurements are replayed into vins_node.

# common parameters
imu: 1
num_of_cam: 2

imu_topic: "/imu0"
image0_topic: "/camera/infra1/image_rect_raw"
image1_topic: "/camera/infra2/image_rect_raw"
output_path: "__OUTPUT_PATH__"

cam0_calib: "__CAM0_CALIB__"
cam1_calib: "__CAM1_CALIB__"
image_width: __WIDTH__
image_height: __HEIGHT__

# IMU-to-camera extrinsics.
# estimate_extrinsic=1 lets VINS refine this RealSense D435i initial guess online.
estimate_extrinsic: __ESTIMATE_EXTRINSIC__

body_T_cam0: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ __BODY_T_CAM0_DATA__ ]

body_T_cam1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [ __BODY_T_CAM1_DATA__ ]

multiple_thread: 1

# feature tracker parameters
max_cnt: __MAX_CNT__
min_dist: __MIN_DIST__
freq: __FREQ__
F_threshold: __F_THRESHOLD__
show_track: 1
flow_back: 1

# optimization parameters
max_solver_time: __MAX_SOLVER_TIME__
max_num_iterations: __MAX_NUM_ITERATIONS__
keyframe_parallax: __KEYFRAME_PARALLAX__

# IMU noise parameters
acc_n: __ACC_N__
gyr_n: __GYR_N__
acc_w: __ACC_W__
gyr_w: __GYR_W__
g_norm: __G_NORM__

# camera/IMU time offset
estimate_td: __ESTIMATE_TD__
td: __TD__

load_previous_pose_graph: 0
pose_graph_save_path: "__OUTPUT_PATH__/pose_graph/"
save_image: 0
