%YAML:1.0

# VINS-Fusion stereo-only config generated for underwater MP4 experiments.
# This is an initial scaffold. Replace fx/fy/cx/cy/baseline with calibrated
# underwater stereo values before treating the trajectory as metric.

# common parameters
imu: 0
num_of_cam: 2

imu_topic: ""
image0_topic: "/leftImage"
image1_topic: "/rightImage"
output_path: "__OUTPUT_PATH__"

cam0_calib: "__CAM0_CALIB__"
cam1_calib: "__CAM1_CALIB__"
image_width: __WIDTH__
image_height: __HEIGHT__

# Extrinsic parameter between body and cameras. For stereo-only KITTI runner,
# body is kept equal to cam0 and cam1 is shifted by the stereo baseline.
estimate_extrinsic: 0

body_T_cam0: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1]

body_T_cam1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [1, 0, 0, __BASELINE__,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1]

multiple_thread: 0

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

# inertial parameters are unused when imu: 0, but VINS-Fusion config parser
# expects the fields to exist in some paths.
acc_n: 0.1
gyr_n: 0.01
acc_w: 0.001
gyr_w: 1.0e-4
g_norm: 9.81007

estimate_td: 0
td: 0.0
