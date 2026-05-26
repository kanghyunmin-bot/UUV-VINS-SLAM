/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"
#include <cstdlib>

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

int USE_GPU;
int USE_GPU_ACC_FLOW;
int USE_GPU_CERES;

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double IMAGE_FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

std::string WORLD_FRAME_ID;
std::string BODY_FRAME_ID;
std::string CAMERA_FRAME_ID;
int PUBLISH_VINS_TF = 1;

int UNDERWATER_MODE;
int UNDERWATER_TEMPORAL_RANSAC;
int UNDERWATER_QUAD_ENABLE;
int UNDERWATER_GRID_ROWS;
int UNDERWATER_GRID_COLS;
int UNDERWATER_CELL_MAX_FEATURES;
double UNDERWATER_QUAD_RIGHT_CLOSURE_THRESHOLD_PX;
double UNDERWATER_QUAD_RIGHT_FB_THRESHOLD_PX;
double UNDERWATER_QUAD_MOTION_CONSISTENCY_THRESHOLD_PX;
double UNDERWATER_FLOW_BACK_THRESHOLD_PX;
int UNDERWATER_LK_WIN_SIZE;
int UNDERWATER_LK_MAX_LEVEL;
int UNDERWATER_QUAD_MIN_CANDIDATES;
int UNDERWATER_QUAD_MIN_TRACKS_AFTER_REJECTION;
double UNDERWATER_STEREO_MAX_Y_DIFF_PX;
int UNDERWATER_STEREO_NCC_FALLBACK;
int UNDERWATER_STEREO_NCC_PATCH_SIZE;
double UNDERWATER_STEREO_NCC_MIN_SCORE;
double UNDERWATER_STEREO_NCC_MIN_MARGIN;
double UNDERWATER_STEREO_MIN_DISPARITY_PX;
double UNDERWATER_STEREO_MAX_DISPARITY_PX;
int UNDERWATER_STEREO_NCC_Y_RADIUS_PX;
int UNDERWATER_STEREO_NCC_FORWARD_BACK;
int UNDERWATER_STEREO_NCC_MIN_TRACK_CNT;
int UNDERWATER_STEREO_NCC_REQUIRE_TEMPORAL;
int UNDERWATER_STEREO_NCC_MAX_CANDIDATES;
int UNDERWATER_REFRACTION_ENABLE;
double UNDERWATER_REFRACTION_INDEX;
double UNDERWATER_REFRACTION_RAY_BLEND;
double UNDERWATER_REFRACTION_DEPTH_SCALE;
double UNDERWATER_FEATURE_DEPTH_SIGN;
int UNDERWATER_RECTIFIED_STEREO_DEPTH_INIT;
double UNDERWATER_EXTRINSIC_PITCH_RAD;
int UNDERWATER_GRAVITY_PNP_ENABLE;
double UNDERWATER_GRAVITY_PNP_YAW_BLEND;
double UNDERWATER_GRAVITY_PNP_MAX_YAW_STEP_RAD;
int UNDERWATER_PNP_RANSAC;
int UNDERWATER_MIN_PNP_INLIERS;
int UNDERWATER_SOFT_MIN_PNP_INLIERS;
double UNDERWATER_MIN_PNP_INLIER_RATIO;
double UNDERWATER_PNP_REPROJ_THRESHOLD_NORM;
double UNDERWATER_MAX_REPROJECTION_RMSE_PX;
double UNDERWATER_PROJECTION_NOISE_PX;
double UNDERWATER_PNP_MAX_STEP_M;
double UNDERWATER_PNP_MAX_ROTATION_RAD;
double UNDERWATER_MIN_LANDMARK_DEPTH;
double UNDERWATER_MAX_LANDMARK_DEPTH;
double UNDERWATER_MAX_LANDMARK_RANGE;
int UNDERWATER_VERTICAL_AXIS;
int UNDERWATER_STABLE_POINT_CLOUD;
int UNDERWATER_STABLE_POINT_CLOUD_MIN_TRACK_CNT;
double UNDERWATER_STABLE_POINT_CLOUD_MAX_DEPTH;
double UNDERWATER_STABLE_POINT_CLOUD_MAX_RANGE;
double UNDERWATER_STEREO_SYNC_TOLERANCE;
int UNDERWATER_DROP_OLD_FRAMES;
int UNDERWATER_VISUAL_DOMINANT_VIO;
int UNDERWATER_PNP_PRIOR_IN_IMU_MODE;
int UNDERWATER_FRONTEND_STATS_LOG;
int UNDERWATER_PNP_MATURE_FEATURE_SELECTION;
int UNDERWATER_PNP_MIN_TRACK_CNT;
int UNDERWATER_PNP_REQUIRE_STEREO;
int UNDERWATER_PNP_REQUIRE_CURRENT_STEREO;
int UNDERWATER_PNP_MAX_CANDIDATES;
int UNDERWATER_PNP_GRID_ROWS;
int UNDERWATER_PNP_GRID_COLS;
int UNDERWATER_PNP_CELL_MAX_FEATURES;
int UNDERWATER_CLAHE_ENABLE;
double UNDERWATER_CLAHE_CLIP_LIMIT;
int UNDERWATER_CLAHE_TILE_SIZE;
int UNDERWATER_FEATURE_MIN_INTENSITY;
int UNDERWATER_FEATURE_MAX_INTENSITY;
int UNDERWATER_FEATURE_STRUCTURE_MASK;
double UNDERWATER_FEATURE_MIN_GRADIENT;
int UNDERWATER_FEATURE_SUPPORT_WINDOW;
double UNDERWATER_FEATURE_MIN_SUPPORT_PIXELS;
int UNDERWATER_FEATURE_BRIGHT_BLOB_THRESHOLD;
int UNDERWATER_FEATURE_MAX_BLOB_AREA;
int UNDERWATER_FEATURE_BLOB_DILATE_PX;

template <typename T>
T readParam(rclcpp::Node::SharedPtr n, std::string name)
{
    T ans;
    if (n->get_parameter(name, ans))
    {
        ROS_INFO("Loaded %s: ", name);
        std::cout << ans << std::endl;
    }
    else
    {
        ROS_ERROR("Failed to load %s", name);
        rclcpp::shutdown();
    }
    return ans;
}

template <typename T>
void readOptionalCvParam(cv::FileStorage &fsSettings, const std::string &name, T &value)
{
    cv::FileNode node = fsSettings[name];
    if (!node.empty())
    {
        node >> value;
        std::cout << "Loaded optional " << name << ": " << value << std::endl;
    }
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        // ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    IMAGE_FREQ = fsSettings["freq"];
    if (!std::isfinite(IMAGE_FREQ) || IMAGE_FREQ < 0.0)
        IMAGE_FREQ = 0.0;
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    UNDERWATER_MODE = 0;
    UNDERWATER_TEMPORAL_RANSAC = 0;
    UNDERWATER_QUAD_ENABLE = 0;
    UNDERWATER_GRID_ROWS = 1;
    UNDERWATER_GRID_COLS = 1;
    UNDERWATER_CELL_MAX_FEATURES = 0;
    UNDERWATER_QUAD_RIGHT_CLOSURE_THRESHOLD_PX = 5.0;
    UNDERWATER_QUAD_RIGHT_FB_THRESHOLD_PX = 0.0;
    UNDERWATER_QUAD_MOTION_CONSISTENCY_THRESHOLD_PX = 0.0;
    UNDERWATER_FLOW_BACK_THRESHOLD_PX = 0.5;
    UNDERWATER_LK_WIN_SIZE = 21;
    UNDERWATER_LK_MAX_LEVEL = 3;
    UNDERWATER_QUAD_MIN_CANDIDATES = 12;
    UNDERWATER_QUAD_MIN_TRACKS_AFTER_REJECTION = 40;
    UNDERWATER_STEREO_MAX_Y_DIFF_PX = 2.0;
    UNDERWATER_STEREO_NCC_FALLBACK = 0;
    UNDERWATER_STEREO_NCC_PATCH_SIZE = 13;
    UNDERWATER_STEREO_NCC_MIN_SCORE = 0.86;
    UNDERWATER_STEREO_NCC_MIN_MARGIN = 0.02;
    UNDERWATER_STEREO_MIN_DISPARITY_PX = 4.0;
    UNDERWATER_STEREO_MAX_DISPARITY_PX = 96.0;
    UNDERWATER_STEREO_NCC_Y_RADIUS_PX = 2;
    UNDERWATER_STEREO_NCC_FORWARD_BACK = 1;
    UNDERWATER_STEREO_NCC_MIN_TRACK_CNT = 2;
    UNDERWATER_STEREO_NCC_REQUIRE_TEMPORAL = 0;
    UNDERWATER_STEREO_NCC_MAX_CANDIDATES = 60;
    UNDERWATER_REFRACTION_ENABLE = 0;
    UNDERWATER_REFRACTION_INDEX = 1.333;
    UNDERWATER_REFRACTION_RAY_BLEND = 0.0;
    UNDERWATER_REFRACTION_DEPTH_SCALE = 1.0;
    UNDERWATER_FEATURE_DEPTH_SIGN = 1.0;
    UNDERWATER_RECTIFIED_STEREO_DEPTH_INIT = 0;
    UNDERWATER_EXTRINSIC_PITCH_RAD = 0.0;
    UNDERWATER_GRAVITY_PNP_ENABLE = 0;
    UNDERWATER_GRAVITY_PNP_YAW_BLEND = 1.0;
    UNDERWATER_GRAVITY_PNP_MAX_YAW_STEP_RAD = 0.35;
    UNDERWATER_PNP_RANSAC = 0;
    UNDERWATER_MIN_PNP_INLIERS = 24;
    UNDERWATER_SOFT_MIN_PNP_INLIERS = 8;
    UNDERWATER_MIN_PNP_INLIER_RATIO = 0.75;
    double underwater_pnp_reproj_threshold_px = 3.0;
    UNDERWATER_MAX_REPROJECTION_RMSE_PX = 3.0;
    UNDERWATER_PROJECTION_NOISE_PX = 1.5;
    UNDERWATER_PNP_MAX_STEP_M = 0.8;
    UNDERWATER_PNP_MAX_ROTATION_RAD = 0.20;
    UNDERWATER_MIN_LANDMARK_DEPTH = 0.12;
    UNDERWATER_MAX_LANDMARK_DEPTH = 3.0;
    UNDERWATER_MAX_LANDMARK_RANGE = 3.5;
    UNDERWATER_VERTICAL_AXIS = 2;
    UNDERWATER_STABLE_POINT_CLOUD = 0;
    UNDERWATER_STABLE_POINT_CLOUD_MIN_TRACK_CNT = 6;
    UNDERWATER_STABLE_POINT_CLOUD_MAX_DEPTH = 1.5;
    UNDERWATER_STABLE_POINT_CLOUD_MAX_RANGE = 1.5;
    UNDERWATER_STEREO_SYNC_TOLERANCE = 0.003;
    UNDERWATER_DROP_OLD_FRAMES = 1;
    UNDERWATER_VISUAL_DOMINANT_VIO = 0;
    UNDERWATER_PNP_PRIOR_IN_IMU_MODE = 0;
    UNDERWATER_FRONTEND_STATS_LOG = 1;
    UNDERWATER_PNP_MATURE_FEATURE_SELECTION = 1;
    UNDERWATER_PNP_MIN_TRACK_CNT = 3;
    UNDERWATER_PNP_REQUIRE_STEREO = 1;
    UNDERWATER_PNP_REQUIRE_CURRENT_STEREO = 0;
    UNDERWATER_PNP_MAX_CANDIDATES = 140;
    UNDERWATER_PNP_GRID_ROWS = 3;
    UNDERWATER_PNP_GRID_COLS = 4;
    UNDERWATER_PNP_CELL_MAX_FEATURES = 18;
    UNDERWATER_CLAHE_ENABLE = 0;
    UNDERWATER_CLAHE_CLIP_LIMIT = 2.0;
    UNDERWATER_CLAHE_TILE_SIZE = 8;
    UNDERWATER_FEATURE_MIN_INTENSITY = 0;
    UNDERWATER_FEATURE_MAX_INTENSITY = 255;
    UNDERWATER_FEATURE_STRUCTURE_MASK = 0;
    UNDERWATER_FEATURE_MIN_GRADIENT = 8.0;
    UNDERWATER_FEATURE_SUPPORT_WINDOW = 21;
    UNDERWATER_FEATURE_MIN_SUPPORT_PIXELS = 45.0;
    UNDERWATER_FEATURE_BRIGHT_BLOB_THRESHOLD = 235;
    UNDERWATER_FEATURE_MAX_BLOB_AREA = 80;
    UNDERWATER_FEATURE_BLOB_DILATE_PX = 3;

    readOptionalCvParam(fsSettings, "underwater_mode", UNDERWATER_MODE);
    readOptionalCvParam(fsSettings, "underwater_temporal_ransac", UNDERWATER_TEMPORAL_RANSAC);
    readOptionalCvParam(fsSettings, "underwater_quad_enable", UNDERWATER_QUAD_ENABLE);
    readOptionalCvParam(fsSettings, "underwater_grid_rows", UNDERWATER_GRID_ROWS);
    readOptionalCvParam(fsSettings, "underwater_grid_cols", UNDERWATER_GRID_COLS);
    readOptionalCvParam(fsSettings, "underwater_cell_max_features", UNDERWATER_CELL_MAX_FEATURES);
    readOptionalCvParam(fsSettings, "underwater_quad_right_closure_threshold_px", UNDERWATER_QUAD_RIGHT_CLOSURE_THRESHOLD_PX);
    readOptionalCvParam(fsSettings, "underwater_quad_right_fb_threshold_px", UNDERWATER_QUAD_RIGHT_FB_THRESHOLD_PX);
    readOptionalCvParam(fsSettings, "underwater_quad_motion_consistency_threshold_px", UNDERWATER_QUAD_MOTION_CONSISTENCY_THRESHOLD_PX);
    readOptionalCvParam(fsSettings, "underwater_flow_back_threshold_px", UNDERWATER_FLOW_BACK_THRESHOLD_PX);
    readOptionalCvParam(fsSettings, "underwater_lk_win_size", UNDERWATER_LK_WIN_SIZE);
    readOptionalCvParam(fsSettings, "underwater_lk_max_level", UNDERWATER_LK_MAX_LEVEL);
    readOptionalCvParam(fsSettings, "underwater_quad_min_candidates", UNDERWATER_QUAD_MIN_CANDIDATES);
    readOptionalCvParam(fsSettings, "underwater_quad_min_tracks_after_rejection", UNDERWATER_QUAD_MIN_TRACKS_AFTER_REJECTION);
    readOptionalCvParam(fsSettings, "underwater_stereo_max_y_diff_px", UNDERWATER_STEREO_MAX_Y_DIFF_PX);
    readOptionalCvParam(fsSettings, "underwater_stereo_ncc_fallback", UNDERWATER_STEREO_NCC_FALLBACK);
    readOptionalCvParam(fsSettings, "underwater_stereo_ncc_patch_size", UNDERWATER_STEREO_NCC_PATCH_SIZE);
    readOptionalCvParam(fsSettings, "underwater_stereo_ncc_min_score", UNDERWATER_STEREO_NCC_MIN_SCORE);
    readOptionalCvParam(fsSettings, "underwater_stereo_ncc_min_margin", UNDERWATER_STEREO_NCC_MIN_MARGIN);
    readOptionalCvParam(fsSettings, "underwater_stereo_min_disparity_px", UNDERWATER_STEREO_MIN_DISPARITY_PX);
    readOptionalCvParam(fsSettings, "underwater_stereo_max_disparity_px", UNDERWATER_STEREO_MAX_DISPARITY_PX);
    readOptionalCvParam(fsSettings, "underwater_stereo_ncc_y_radius_px", UNDERWATER_STEREO_NCC_Y_RADIUS_PX);
    readOptionalCvParam(fsSettings, "underwater_stereo_ncc_forward_back", UNDERWATER_STEREO_NCC_FORWARD_BACK);
    readOptionalCvParam(fsSettings, "underwater_stereo_ncc_min_track_cnt", UNDERWATER_STEREO_NCC_MIN_TRACK_CNT);
    readOptionalCvParam(fsSettings, "underwater_stereo_ncc_require_temporal", UNDERWATER_STEREO_NCC_REQUIRE_TEMPORAL);
    readOptionalCvParam(fsSettings, "underwater_stereo_ncc_max_candidates", UNDERWATER_STEREO_NCC_MAX_CANDIDATES);
    readOptionalCvParam(fsSettings, "underwater_refraction_enable", UNDERWATER_REFRACTION_ENABLE);
    readOptionalCvParam(fsSettings, "underwater_refraction_index", UNDERWATER_REFRACTION_INDEX);
    readOptionalCvParam(fsSettings, "underwater_refraction_ray_blend", UNDERWATER_REFRACTION_RAY_BLEND);
    readOptionalCvParam(fsSettings, "underwater_refraction_depth_scale", UNDERWATER_REFRACTION_DEPTH_SCALE);
    readOptionalCvParam(fsSettings, "underwater_feature_depth_sign", UNDERWATER_FEATURE_DEPTH_SIGN);
    UNDERWATER_FEATURE_DEPTH_SIGN = UNDERWATER_FEATURE_DEPTH_SIGN < 0.0 ? -1.0 : 1.0;
    readOptionalCvParam(fsSettings, "underwater_rectified_stereo_depth_init", UNDERWATER_RECTIFIED_STEREO_DEPTH_INIT);
    readOptionalCvParam(fsSettings, "underwater_extrinsic_pitch_rad", UNDERWATER_EXTRINSIC_PITCH_RAD);
    readOptionalCvParam(fsSettings, "underwater_gravity_pnp_enable", UNDERWATER_GRAVITY_PNP_ENABLE);
    readOptionalCvParam(fsSettings, "underwater_gravity_pnp_yaw_blend", UNDERWATER_GRAVITY_PNP_YAW_BLEND);
    readOptionalCvParam(fsSettings, "underwater_gravity_pnp_max_yaw_step_rad", UNDERWATER_GRAVITY_PNP_MAX_YAW_STEP_RAD);
    readOptionalCvParam(fsSettings, "underwater_pnp_ransac", UNDERWATER_PNP_RANSAC);
    readOptionalCvParam(fsSettings, "underwater_min_pnp_inliers", UNDERWATER_MIN_PNP_INLIERS);
    readOptionalCvParam(fsSettings, "underwater_soft_min_pnp_inliers", UNDERWATER_SOFT_MIN_PNP_INLIERS);
    readOptionalCvParam(fsSettings, "underwater_min_pnp_inlier_ratio", UNDERWATER_MIN_PNP_INLIER_RATIO);
    readOptionalCvParam(fsSettings, "underwater_pnp_reproj_threshold_px", underwater_pnp_reproj_threshold_px);
    readOptionalCvParam(fsSettings, "underwater_max_reprojection_rmse_px", UNDERWATER_MAX_REPROJECTION_RMSE_PX);
    readOptionalCvParam(fsSettings, "underwater_projection_noise_px", UNDERWATER_PROJECTION_NOISE_PX);
    readOptionalCvParam(fsSettings, "underwater_pnp_max_step_m", UNDERWATER_PNP_MAX_STEP_M);
    readOptionalCvParam(fsSettings, "underwater_pnp_max_rotation_rad", UNDERWATER_PNP_MAX_ROTATION_RAD);
    readOptionalCvParam(fsSettings, "underwater_min_landmark_depth", UNDERWATER_MIN_LANDMARK_DEPTH);
    readOptionalCvParam(fsSettings, "underwater_max_landmark_depth", UNDERWATER_MAX_LANDMARK_DEPTH);
    readOptionalCvParam(fsSettings, "underwater_max_landmark_range", UNDERWATER_MAX_LANDMARK_RANGE);
    readOptionalCvParam(fsSettings, "underwater_vertical_axis", UNDERWATER_VERTICAL_AXIS);
    readOptionalCvParam(fsSettings, "underwater_stable_point_cloud", UNDERWATER_STABLE_POINT_CLOUD);
    readOptionalCvParam(fsSettings, "underwater_stable_point_cloud_min_track_cnt", UNDERWATER_STABLE_POINT_CLOUD_MIN_TRACK_CNT);
    readOptionalCvParam(fsSettings, "underwater_stable_point_cloud_max_depth", UNDERWATER_STABLE_POINT_CLOUD_MAX_DEPTH);
    readOptionalCvParam(fsSettings, "underwater_stable_point_cloud_max_range", UNDERWATER_STABLE_POINT_CLOUD_MAX_RANGE);
    readOptionalCvParam(fsSettings, "underwater_stereo_sync_tolerance", UNDERWATER_STEREO_SYNC_TOLERANCE);
    readOptionalCvParam(fsSettings, "underwater_drop_old_frames", UNDERWATER_DROP_OLD_FRAMES);
    readOptionalCvParam(fsSettings, "underwater_visual_dominant_vio", UNDERWATER_VISUAL_DOMINANT_VIO);
    readOptionalCvParam(fsSettings, "underwater_pnp_prior_in_imu_mode", UNDERWATER_PNP_PRIOR_IN_IMU_MODE);
    readOptionalCvParam(fsSettings, "underwater_frontend_stats_log", UNDERWATER_FRONTEND_STATS_LOG);
    readOptionalCvParam(fsSettings, "underwater_pnp_mature_feature_selection", UNDERWATER_PNP_MATURE_FEATURE_SELECTION);
    readOptionalCvParam(fsSettings, "underwater_pnp_min_track_cnt", UNDERWATER_PNP_MIN_TRACK_CNT);
    readOptionalCvParam(fsSettings, "underwater_pnp_require_stereo", UNDERWATER_PNP_REQUIRE_STEREO);
    readOptionalCvParam(fsSettings, "underwater_pnp_require_current_stereo", UNDERWATER_PNP_REQUIRE_CURRENT_STEREO);
    readOptionalCvParam(fsSettings, "underwater_pnp_max_candidates", UNDERWATER_PNP_MAX_CANDIDATES);
    readOptionalCvParam(fsSettings, "underwater_pnp_grid_rows", UNDERWATER_PNP_GRID_ROWS);
    readOptionalCvParam(fsSettings, "underwater_pnp_grid_cols", UNDERWATER_PNP_GRID_COLS);
    readOptionalCvParam(fsSettings, "underwater_pnp_cell_max_features", UNDERWATER_PNP_CELL_MAX_FEATURES);
    readOptionalCvParam(fsSettings, "underwater_clahe_enable", UNDERWATER_CLAHE_ENABLE);
    readOptionalCvParam(fsSettings, "underwater_clahe_clip_limit", UNDERWATER_CLAHE_CLIP_LIMIT);
    readOptionalCvParam(fsSettings, "underwater_clahe_tile_size", UNDERWATER_CLAHE_TILE_SIZE);
    readOptionalCvParam(fsSettings, "underwater_feature_min_intensity", UNDERWATER_FEATURE_MIN_INTENSITY);
    readOptionalCvParam(fsSettings, "underwater_feature_max_intensity", UNDERWATER_FEATURE_MAX_INTENSITY);
    readOptionalCvParam(fsSettings, "underwater_feature_structure_mask", UNDERWATER_FEATURE_STRUCTURE_MASK);
    readOptionalCvParam(fsSettings, "underwater_feature_min_gradient", UNDERWATER_FEATURE_MIN_GRADIENT);
    readOptionalCvParam(fsSettings, "underwater_feature_support_window", UNDERWATER_FEATURE_SUPPORT_WINDOW);
    readOptionalCvParam(fsSettings, "underwater_feature_min_support_pixels", UNDERWATER_FEATURE_MIN_SUPPORT_PIXELS);
    readOptionalCvParam(fsSettings, "underwater_feature_bright_blob_threshold", UNDERWATER_FEATURE_BRIGHT_BLOB_THRESHOLD);
    readOptionalCvParam(fsSettings, "underwater_feature_max_blob_area", UNDERWATER_FEATURE_MAX_BLOB_AREA);
    readOptionalCvParam(fsSettings, "underwater_feature_blob_dilate_px", UNDERWATER_FEATURE_BLOB_DILATE_PX);
    UNDERWATER_PNP_REPROJ_THRESHOLD_NORM = underwater_pnp_reproj_threshold_px / FOCAL_LENGTH;

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_GPU = fsSettings["use_gpu"];
    USE_GPU_ACC_FLOW = fsSettings["use_gpu_acc_flow"];
    USE_GPU_CERES = fsSettings["use_gpu_ceres"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(!fsSettings["imu_topic"].empty())
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
    }
    if(USE_IMU)
    {
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    const char *output_path_env = std::getenv("VINS_OUTPUT_PATH");
    if (output_path_env != nullptr && output_path_env[0] != '\0')
        OUTPUT_FOLDER = output_path_env;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        if (UNDERWATER_MODE && std::abs(UNDERWATER_EXTRINSIC_PITCH_RAD) > 1e-12)
            T.block<3, 3>(0, 0) =
                T.block<3, 3>(0, 0) *
                Eigen::AngleAxisd(UNDERWATER_EXTRINSIC_PITCH_RAD, Eigen::Vector3d::UnitY()).toRotationMatrix();
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }


    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib; 
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);
        
        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        if (UNDERWATER_MODE && std::abs(UNDERWATER_EXTRINSIC_PITCH_RAD) > 1e-12)
            T.block<3, 3>(0, 0) =
                T.block<3, 3>(0, 0) *
                Eigen::AngleAxisd(UNDERWATER_EXTRINSIC_PITCH_RAD, Eigen::Vector3d::UnitY()).toRotationMatrix();
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO("Unsynchronized sensors, online estimate time offset, initial td: %f", TD);
    else
        ROS_INFO("Synchronized sensors, fix time offset: %f", TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings["world_frame_id"] >> WORLD_FRAME_ID;
    WORLD_FRAME_ID.empty()? WORLD_FRAME_ID = "world" : WORLD_FRAME_ID;
    fsSettings["body_frame_id"] >> BODY_FRAME_ID;   
    BODY_FRAME_ID.empty()? BODY_FRAME_ID = "body" : BODY_FRAME_ID;
    fsSettings["camera_frame_id"] >> CAMERA_FRAME_ID;
    CAMERA_FRAME_ID.empty()? CAMERA_FRAME_ID = "camera" : CAMERA_FRAME_ID;
    readOptionalCvParam(fsSettings, "publish_vins_tf", PUBLISH_VINS_TF);

    ROS_INFO("frame_ids: world=%s body=%s camera=%s publish_vins_tf=%d", WORLD_FRAME_ID.c_str(),
             BODY_FRAME_ID.c_str(), CAMERA_FRAME_ID.c_str(), PUBLISH_VINS_TF);

    fsSettings.release();
}
