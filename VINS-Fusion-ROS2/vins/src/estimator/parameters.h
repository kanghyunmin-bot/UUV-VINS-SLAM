/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_ERROR RCUTILS_LOG_ERROR

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern int USE_GPU;
extern int USE_GPU_ACC_FLOW;
extern int USE_GPU_CERES;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern double IMAGE_FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;
extern std::string WORLD_FRAME_ID;
extern std::string BODY_FRAME_ID;
extern std::string CAMERA_FRAME_ID;
extern int PUBLISH_VINS_TF;

extern int UNDERWATER_MODE;
extern int UNDERWATER_TEMPORAL_RANSAC;
extern int UNDERWATER_QUAD_ENABLE;
extern int UNDERWATER_GRID_ROWS;
extern int UNDERWATER_GRID_COLS;
extern int UNDERWATER_CELL_MAX_FEATURES;
extern double UNDERWATER_QUAD_RIGHT_CLOSURE_THRESHOLD_PX;
extern double UNDERWATER_QUAD_RIGHT_FB_THRESHOLD_PX;
extern double UNDERWATER_QUAD_MOTION_CONSISTENCY_THRESHOLD_PX;
extern double UNDERWATER_FLOW_BACK_THRESHOLD_PX;
extern int UNDERWATER_LK_WIN_SIZE;
extern int UNDERWATER_LK_MAX_LEVEL;
extern int UNDERWATER_QUAD_MIN_CANDIDATES;
extern int UNDERWATER_QUAD_MIN_TRACKS_AFTER_REJECTION;
extern double UNDERWATER_STEREO_MAX_Y_DIFF_PX;
extern int UNDERWATER_STEREO_NCC_FALLBACK;
extern int UNDERWATER_STEREO_NCC_PATCH_SIZE;
extern double UNDERWATER_STEREO_NCC_MIN_SCORE;
extern double UNDERWATER_STEREO_NCC_MIN_MARGIN;
extern double UNDERWATER_STEREO_MIN_DISPARITY_PX;
extern double UNDERWATER_STEREO_MAX_DISPARITY_PX;
extern int UNDERWATER_STEREO_NCC_Y_RADIUS_PX;
extern int UNDERWATER_STEREO_NCC_FORWARD_BACK;
extern int UNDERWATER_STEREO_NCC_MIN_TRACK_CNT;
extern int UNDERWATER_STEREO_NCC_REQUIRE_TEMPORAL;
extern int UNDERWATER_STEREO_NCC_MAX_CANDIDATES;
extern int UNDERWATER_REFRACTION_ENABLE;
extern double UNDERWATER_REFRACTION_INDEX;
extern double UNDERWATER_REFRACTION_RAY_BLEND;
extern double UNDERWATER_REFRACTION_DEPTH_SCALE;
extern double UNDERWATER_FEATURE_DEPTH_SIGN;
extern int UNDERWATER_RECTIFIED_STEREO_DEPTH_INIT;
extern double UNDERWATER_EXTRINSIC_PITCH_RAD;
extern int UNDERWATER_GRAVITY_PNP_ENABLE;
extern double UNDERWATER_GRAVITY_PNP_YAW_BLEND;
extern double UNDERWATER_GRAVITY_PNP_MAX_YAW_STEP_RAD;
extern int UNDERWATER_PNP_RANSAC;
extern int UNDERWATER_MIN_PNP_INLIERS;
extern int UNDERWATER_SOFT_MIN_PNP_INLIERS;
extern double UNDERWATER_MIN_PNP_INLIER_RATIO;
extern double UNDERWATER_PNP_REPROJ_THRESHOLD_NORM;
extern double UNDERWATER_MAX_REPROJECTION_RMSE_PX;
extern double UNDERWATER_PROJECTION_NOISE_PX;
extern double UNDERWATER_PNP_MAX_STEP_M;
extern double UNDERWATER_PNP_MAX_ROTATION_RAD;
extern double UNDERWATER_MIN_LANDMARK_DEPTH;
extern double UNDERWATER_MAX_LANDMARK_DEPTH;
extern double UNDERWATER_MAX_LANDMARK_RANGE;
extern int UNDERWATER_VERTICAL_AXIS;
extern int UNDERWATER_STABLE_POINT_CLOUD;
extern int UNDERWATER_STABLE_POINT_CLOUD_MIN_TRACK_CNT;
extern double UNDERWATER_STABLE_POINT_CLOUD_MAX_DEPTH;
extern double UNDERWATER_STABLE_POINT_CLOUD_MAX_RANGE;
extern double UNDERWATER_STEREO_SYNC_TOLERANCE;
extern int UNDERWATER_DROP_OLD_FRAMES;
extern int UNDERWATER_VISUAL_DOMINANT_VIO;
extern int UNDERWATER_PNP_PRIOR_IN_IMU_MODE;
extern int UNDERWATER_FRONTEND_STATS_LOG;
extern int UNDERWATER_PNP_MATURE_FEATURE_SELECTION;
extern int UNDERWATER_PNP_MIN_TRACK_CNT;
extern int UNDERWATER_PNP_REQUIRE_STEREO;
extern int UNDERWATER_PNP_REQUIRE_CURRENT_STEREO;
extern int UNDERWATER_PNP_MAX_CANDIDATES;
extern int UNDERWATER_PNP_GRID_ROWS;
extern int UNDERWATER_PNP_GRID_COLS;
extern int UNDERWATER_PNP_CELL_MAX_FEATURES;
extern int UNDERWATER_CLAHE_ENABLE;
extern double UNDERWATER_CLAHE_CLIP_LIMIT;
extern int UNDERWATER_CLAHE_TILE_SIZE;
extern int UNDERWATER_FEATURE_MIN_INTENSITY;
extern int UNDERWATER_FEATURE_MAX_INTENSITY;
extern int UNDERWATER_FEATURE_STRUCTURE_MASK;
extern double UNDERWATER_FEATURE_MIN_GRADIENT;
extern int UNDERWATER_FEATURE_SUPPORT_WINDOW;
extern double UNDERWATER_FEATURE_MIN_SUPPORT_PIXELS;
extern int UNDERWATER_FEATURE_BRIGHT_BLOB_THRESHOLD;
extern int UNDERWATER_FEATURE_MAX_BLOB_AREA;
extern int UNDERWATER_FEATURE_BLOB_DILATE_PX;

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
