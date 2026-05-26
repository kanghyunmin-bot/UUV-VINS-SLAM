/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace
{
struct PnPCandidate
{
    cv::Point2f point2d;
    cv::Point3f point3d;
    cv::Point2f uv;
    int track_count = 0;
    bool has_stereo = false;
    bool current_stereo = false;
    double stereo_y_error = std::numeric_limits<double>::infinity();
    double abs_disparity = std::numeric_limits<double>::infinity();
    double depth = -1.0;
    double range = -1.0;
    double score = 0.0;
};

bool featureHasStereoObservation(const FeaturePerId &feature)
{
    for (size_t i = 0; i < feature.feature_per_frame.size(); i++)
        if (feature.feature_per_frame[i].is_stereo)
            return true;
    return false;
}

double applyFeatureDepthSign(double depth)
{
    if (UNDERWATER_MODE && UNDERWATER_FEATURE_DEPTH_SIGN < 0.0)
        return -depth;
    return depth;
}

bool estimateRectifiedStereoDepth(const FeaturePerFrame &observation,
                                  const Vector3d tic[],
                                  const Matrix3d ric[],
                                  double *depth,
                                  Vector3d *local_point)
{
    if (!UNDERWATER_MODE || !UNDERWATER_RECTIFIED_STEREO_DEPTH_INIT ||
        !observation.is_stereo || depth == nullptr || local_point == nullptr)
        return false;

    Eigen::Vector3d cam1_in_cam0 = ric[0].transpose() * (tic[1] - tic[0]);
    const double baseline = cam1_in_cam0.x();
    const double disparity = observation.point.x() - observation.pointRight.x();
    if (!std::isfinite(baseline) || !std::isfinite(disparity) ||
        std::abs(baseline) < 1e-6 || std::abs(disparity) < 1e-9)
        return false;

    double raw_depth = baseline / disparity;
    raw_depth = applyFeatureDepthSign(raw_depth);
    double corrected_depth = raw_depth;
    if (UNDERWATER_REFRACTION_ENABLE)
        corrected_depth *= std::max(0.05, UNDERWATER_REFRACTION_DEPTH_SCALE);
    if (!std::isfinite(corrected_depth))
        return false;

    *depth = corrected_depth;
    *local_point = observation.point * corrected_depth;
    return true;
}

double scorePnPCandidate(const PnPCandidate &candidate)
{
    double score = 0.0;
    score += std::min(candidate.track_count, 12) * 2.0;
    if (candidate.has_stereo)
        score += 4.0;
    if (candidate.current_stereo)
        score += 5.0;
    if (std::isfinite(candidate.stereo_y_error))
        score += std::max(0.0, 4.0 - candidate.stereo_y_error);
    if (std::isfinite(candidate.abs_disparity))
    {
        const double min_disp = std::max(0.0, UNDERWATER_STEREO_MIN_DISPARITY_PX);
        const double max_disp = std::max(min_disp, UNDERWATER_STEREO_MAX_DISPARITY_PX);
        if (candidate.abs_disparity >= min_disp && candidate.abs_disparity <= max_disp)
            score += 2.0;
    }
    if (std::isfinite(candidate.depth) && candidate.depth > 0.0)
        score += std::max(0.0, 3.0 - std::abs(candidate.depth - 1.5));
    return score;
}

void selectPnPCandidateSubset(const vector<PnPCandidate> &candidates,
                              vector<cv::Point2f> &pts2D,
                              vector<cv::Point3f> &pts3D,
                              int grid_rows,
                              int grid_cols,
                              int cell_limit,
                              int max_candidates)
{
    pts2D.clear();
    pts3D.clear();
    const int rows = std::max(1, grid_rows);
    const int cols = std::max(1, grid_cols);
    const int per_cell_limit = std::max(0, cell_limit);
    const int total_limit = std::max(0, max_candidates);
    vector<int> cell_counts(rows * cols, 0);

    for (size_t i = 0; i < candidates.size(); i++)
    {
        const PnPCandidate &candidate = candidates[i];
        int gx = 0;
        int gy = 0;
        if (COL > 0 && ROW > 0)
        {
            gx = std::max(0, std::min(cols - 1,
                static_cast<int>(candidate.uv.x / std::max(1.0, static_cast<double>(COL)) * cols)));
            gy = std::max(0, std::min(rows - 1,
                static_cast<int>(candidate.uv.y / std::max(1.0, static_cast<double>(ROW)) * rows)));
        }
        const int cell_index = gy * cols + gx;
        if (per_cell_limit > 0 && cell_counts[cell_index] >= per_cell_limit)
            continue;
        pts3D.push_back(candidate.point3d);
        pts2D.push_back(candidate.point2d);
        cell_counts[cell_index]++;
        if (total_limit > 0 && static_cast<int>(pts2D.size()) >= total_limit)
            break;
    }
}

void appendUnderwaterPnPStats(int frame_cnt,
                              int raw_candidates,
                              int valid_depth,
                              int mature_candidates,
                              int selected_candidates,
                              int pnp_inliers,
                              double pnp_rmse_px,
                              bool success,
                              double pose_step_m,
                              double pose_rotation_rad,
                              const std::string &reject_reason,
                              int reject_young,
                              int reject_no_stereo,
                              int reject_current_no_stereo,
                              int reject_stereo_geometry)
{
    if (!UNDERWATER_MODE || !UNDERWATER_FRONTEND_STATS_LOG || OUTPUT_FOLDER.empty())
        return;

    static bool initialized = false;
    const std::string path = OUTPUT_FOLDER + "/underwater_pnp_stats.csv";
    std::ofstream fout;
    if (!initialized)
    {
        fout.open(path, std::ios::out);
        fout << "frame,raw_candidates,valid_depth,mature_candidates,selected_candidates,"
             << "pnp_inliers,pnp_rmse_px,success,pose_step_m,pose_rotation_rad,reject_reason,"
             << "reject_young,reject_no_stereo,reject_current_no_stereo,reject_stereo_geometry\n";
        initialized = true;
    }
    else
    {
        fout.open(path, std::ios::app);
    }
    if (!fout.is_open())
        return;
    fout << frame_cnt << ","
         << raw_candidates << ","
         << valid_depth << ","
         << mature_candidates << ","
         << selected_candidates << ","
         << pnp_inliers << ","
         << std::fixed << std::setprecision(3) << pnp_rmse_px << ","
         << (success ? 1 : 0) << ","
         << std::fixed << std::setprecision(4) << pose_step_m << ","
         << std::fixed << std::setprecision(4) << pose_rotation_rad << ","
         << reject_reason << ","
         << reject_young << ","
         << reject_no_stereo << ","
         << reject_current_no_stereo << ","
         << reject_stereo_geometry << "\n";
}
}

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

bool FeatureManager::isUnderwaterDepthValueValid(double depth, const Eigen::Vector3d *local_point) const
{
    if (!UNDERWATER_MODE)
        return depth > 0.0;
    if (!std::isfinite(depth) || depth < UNDERWATER_MIN_LANDMARK_DEPTH)
        return false;
    if (UNDERWATER_MAX_LANDMARK_DEPTH > 0.0 && depth > UNDERWATER_MAX_LANDMARK_DEPTH)
        return false;
    if (local_point != nullptr && UNDERWATER_MAX_LANDMARK_RANGE > 0.0 &&
        (!std::isfinite(local_point->norm()) || local_point->norm() > UNDERWATER_MAX_LANDMARK_RANGE))
        return false;
    return true;
}

bool FeatureManager::isDepthUsableForOptimization(const FeaturePerId &it_per_id) const
{
    const size_t min_observations = UNDERWATER_MODE ? 3 : 4;
    if (it_per_id.feature_per_frame.size() < min_observations)
        return false;
    if (UNDERWATER_MODE && STEREO && UNDERWATER_PNP_REQUIRE_STEREO &&
        !featureHasStereoObservation(it_per_id))
        return false;
    return isUnderwaterDepthValueValid(it_per_id.estimated_depth);
}

Eigen::Vector3d FeatureManager::correctUnderwaterRefractiveRay(const Eigen::Vector3d &point) const
{
    if (!UNDERWATER_MODE || !UNDERWATER_REFRACTION_ENABLE)
        return point;

    double blend = std::max(0.0, std::min(1.0, UNDERWATER_REFRACTION_RAY_BLEND));
    double refractive_index = std::max(1.0001, UNDERWATER_REFRACTION_INDEX);
    if (blend <= 0.0)
        return point;

    Eigen::Vector3d corrected = point;
    double z = std::abs(point.z()) > 1e-9 ? point.z() : 1.0;
    double x = point.x() / z;
    double y = point.y() / z;
    double r = std::sqrt(x * x + y * y);
    if (!std::isfinite(r) || r <= 1e-9)
        return point;

    // First-order flat-port Snell correction for a flat acrylic interface.
    // Keep the blend near zero unless the housing geometry is calibrated.
    double theta_air = std::atan(r);
    double sin_water = std::sin(theta_air) / refractive_index;
    sin_water = std::max(-0.999999, std::min(0.999999, sin_water));
    double theta_water = std::asin(sin_water);
    double refracted_r = std::tan(theta_water);
    if (!std::isfinite(refracted_r) || refracted_r <= 0.0)
        return point;

    double radial_scale = refracted_r / r;
    double mixed_scale = 1.0 + blend * (radial_scale - 1.0);
    corrected.x() = x * mixed_scale;
    corrected.y() = y * mixed_scale;
    corrected.z() = 1.0;
    return corrected;
}

double FeatureManager::correctUnderwaterRefractiveDepth(double depth) const
{
    if (!UNDERWATER_MODE || !UNDERWATER_REFRACTION_ENABLE)
        return depth;
    double depth_scale = std::max(0.05, UNDERWATER_REFRACTION_DEPTH_SCALE);
    return depth * depth_scale;
}

Eigen::Matrix3d FeatureManager::applyUnderwaterGravityPnPConstraint(
    const Eigen::Matrix3d &previous_body_R,
    const Eigen::Matrix3d &candidate_body_R
) const
{
    if (!UNDERWATER_MODE || !UNDERWATER_GRAVITY_PNP_ENABLE)
        return candidate_body_R;

    int vertical_axis = UNDERWATER_VERTICAL_AXIS;
    if (vertical_axis < 0 || vertical_axis > 2)
        vertical_axis = 2;

    Eigen::Vector3d axis = Eigen::Vector3d::Zero();
    axis(vertical_axis) = 1.0;

    Eigen::Vector3d prev_forward = previous_body_R * ric[0] * Eigen::Vector3d(0.0, 0.0, 1.0);
    Eigen::Vector3d cand_forward = candidate_body_R * ric[0] * Eigen::Vector3d(0.0, 0.0, 1.0);
    prev_forward(vertical_axis) = 0.0;
    cand_forward(vertical_axis) = 0.0;
    double prev_norm = prev_forward.norm();
    double cand_norm = cand_forward.norm();
    if (!std::isfinite(prev_norm) || !std::isfinite(cand_norm) || prev_norm <= 1e-9 || cand_norm <= 1e-9)
        return candidate_body_R;

    prev_forward /= prev_norm;
    cand_forward /= cand_norm;
    double dot = std::max(-1.0, std::min(1.0, prev_forward.dot(cand_forward)));
    double cross_axis = prev_forward.cross(cand_forward).dot(axis);
    double yaw_delta = std::atan2(cross_axis, dot);

    double max_yaw_step = std::max(0.0, UNDERWATER_GRAVITY_PNP_MAX_YAW_STEP_RAD);
    if (max_yaw_step > 0.0)
        yaw_delta = std::max(-max_yaw_step, std::min(max_yaw_step, yaw_delta));
    double yaw_blend = std::max(0.0, std::min(1.0, UNDERWATER_GRAVITY_PNP_YAW_BLEND));
    yaw_delta *= yaw_blend;

    return Eigen::AngleAxisd(yaw_delta, axis).toRotationMatrix() * previous_body_R;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();
        if (isDepthUsableForOptimization(it))
        {
            cnt++;
        }
    }
    return cnt;
}


bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    last_average_parallax = 0;
    new_feature_num = 0;
    long_track_num = 0;
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
        f_per_fra.point = correctUnderwaterRefractiveRay(f_per_fra.point);
        assert(id_pts.second[0].first == 0);
        if(id_pts.second.size() == 2)
        {
            f_per_fra.rightObservation(id_pts.second[1].second);
            f_per_fra.pointRight = correctUnderwaterRefractiveRay(f_per_fra.pointRight);
            assert(id_pts.second[1].first == 1);
        }

        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
            new_feature_num++;
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            if( it-> feature_per_frame.size() >= 4)
                long_track_num++;
        }
    }

    const int min_last_track = UNDERWATER_MODE ? 12 : 20;
    const int min_long_track = UNDERWATER_MODE ? 18 : 40;
    const double max_new_track_ratio = UNDERWATER_MODE ? 0.75 : 0.5;
    if (frame_count < 2 ||
        last_track_num < min_last_track ||
        long_track_num < min_long_track ||
        new_feature_num > max_new_track_ratio * std::max(1, last_track_num))
        return true;

    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!isDepthUsableForOptimization(it_per_id))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (!isUnderwaterDepthValueValid(it_per_id.estimated_depth))
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth()
{
    for (auto &it_per_id : feature)
        it_per_id.estimated_depth = -1;
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!isDepthUsableForOptimization(it_per_id))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}


void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D,
                                      int *inlier_count,
                                      double *rmse_px_out,
                                      std::string *reject_reason)
{
    if (inlier_count != nullptr)
        *inlier_count = 0;
    if (rmse_px_out != nullptr)
        *rmse_px_out = -1.0;
    if (reject_reason != nullptr)
        reject_reason->clear();
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        if (reject_reason != nullptr)
            *reject_reason = "too_few_points";
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
    bool pnp_succ;
    cv::Mat inliers;
    const bool use_underwater_pnp_ransac = UNDERWATER_MODE && UNDERWATER_PNP_RANSAC;
    if (use_underwater_pnp_ransac)
    {
        pnp_succ = cv::solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100,
                                      UNDERWATER_PNP_REPROJ_THRESHOLD_NORM, 0.99, inliers);
        const int candidate_count = static_cast<int>(pts2D.size());
        const int ratio_min_inliers = static_cast<int>(
            std::ceil(UNDERWATER_MIN_PNP_INLIER_RATIO * std::max(candidate_count, 1))
        );
        int configured_min_inliers = UNDERWATER_MIN_PNP_INLIERS;
        if (UNDERWATER_SOFT_MIN_PNP_INLIERS > 0 &&
            candidate_count < std::max(UNDERWATER_MIN_PNP_INLIERS * 2, UNDERWATER_MIN_PNP_INLIERS + 1))
        {
            configured_min_inliers = std::min(UNDERWATER_MIN_PNP_INLIERS, UNDERWATER_SOFT_MIN_PNP_INLIERS);
        }
        const int adaptive_min_inliers = std::min(
            candidate_count,
            std::max(configured_min_inliers, ratio_min_inliers)
        );
        if (!pnp_succ || inliers.rows < adaptive_min_inliers)
        {
            ROS_WARN("underwater pnp rejected: inliers %d / %d, need %d",
                     inliers.rows, candidate_count, adaptive_min_inliers);
            if (inlier_count != nullptr)
                *inlier_count = inliers.rows;
            if (reject_reason != nullptr)
                *reject_reason = "low_inliers";
            return false;
        }
        if (inlier_count != nullptr)
            *inlier_count = inliers.rows;
        if (inliers.rows >= 4)
        {
            vector<cv::Point2f> inlier_pts2D;
            vector<cv::Point3f> inlier_pts3D;
            inlier_pts2D.reserve(inliers.rows);
            inlier_pts3D.reserve(inliers.rows);
            for (int i = 0; i < inliers.rows; i++)
            {
                int idx = inliers.at<int>(i, 0);
                if (idx < 0 || idx >= candidate_count)
                    continue;
                inlier_pts2D.push_back(pts2D[idx]);
                inlier_pts3D.push_back(pts3D[idx]);
            }
            if (inlier_pts2D.size() >= 4)
                cv::solvePnP(inlier_pts3D, inlier_pts2D, K, D, rvec, t, true, cv::SOLVEPNP_ITERATIVE);
        }
    }
    else
    {
        pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    }

    if(!pnp_succ)
    {
        printf("pnp failed ! \n");
        if (reject_reason != nullptr)
            *reject_reason = "solve_failed";
        return false;
    }

    if (use_underwater_pnp_ransac)
    {
        vector<cv::Point2f> projected;
        cv::projectPoints(pts3D, rvec, t, K, D, projected);
        double sq_sum = 0.0;
        int count = 0;
        if (inliers.rows > 0)
        {
            for (int i = 0; i < inliers.rows; i++)
            {
                int idx = inliers.at<int>(i, 0);
                if (idx < 0 || idx >= static_cast<int>(pts2D.size()))
                    continue;
                double dx = static_cast<double>(projected[idx].x - pts2D[idx].x) * FOCAL_LENGTH;
                double dy = static_cast<double>(projected[idx].y - pts2D[idx].y) * FOCAL_LENGTH;
                sq_sum += dx * dx + dy * dy;
                count++;
            }
        }
        if (count == 0)
        {
            for (size_t i = 0; i < pts2D.size(); i++)
            {
                double dx = static_cast<double>(projected[i].x - pts2D[i].x) * FOCAL_LENGTH;
                double dy = static_cast<double>(projected[i].y - pts2D[i].y) * FOCAL_LENGTH;
                sq_sum += dx * dx + dy * dy;
                count++;
            }
        }
        double rmse_px = std::sqrt(sq_sum / std::max(count, 1));
        if (rmse_px_out != nullptr)
            *rmse_px_out = rmse_px;
        if (rmse_px > UNDERWATER_MAX_REPROJECTION_RMSE_PX)
        {
            ROS_WARN("underwater pnp rejected: reprojection rmse %.3f px", rmse_px);
            if (reject_reason != nullptr)
                *reject_reason = "high_rmse";
            return false;
        }
    }
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

bool FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[],
                                        double underwater_pnp_step_limit_m,
                                        double underwater_pnp_rotation_limit_rad)
{

    if(frameCnt > 0)
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        vector<PnPCandidate> all_valid_candidates;
        vector<PnPCandidate> fallback_candidates;
        vector<PnPCandidate> pnp_candidates;
        int raw_candidates = 0;
        int valid_depth_candidates = 0;
        int reject_young = 0;
        int reject_no_stereo = 0;
        int reject_current_no_stereo = 0;
        int reject_stereo_geometry = 0;
        for (auto &it_per_id : feature)
        {
            raw_candidates++;
            const int index = frameCnt - it_per_id.start_frame;
            if (index < 0 || static_cast<int>(it_per_id.feature_per_frame.size()) < index + 1)
                continue;

            Eigen::Vector3d local_point =
                it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            if (isUnderwaterDepthValueValid(it_per_id.estimated_depth, &local_point))
            {
                valid_depth_candidates++;
                const int track_count = static_cast<int>(it_per_id.feature_per_frame.size());
                const bool mature_selection =
                    UNDERWATER_MODE && UNDERWATER_PNP_MATURE_FEATURE_SELECTION;
                const int min_track_count = std::max(1, UNDERWATER_PNP_MIN_TRACK_CNT);
                const bool has_stereo = featureHasStereoObservation(it_per_id);
                const FeaturePerFrame &current_obs = it_per_id.feature_per_frame[index];

                PnPCandidate candidate;
                candidate.track_count = track_count;
                candidate.has_stereo = has_stereo;
                candidate.current_stereo = current_obs.is_stereo;
                candidate.depth = it_per_id.estimated_depth;
                candidate.range = local_point.norm();

                if (current_obs.is_stereo)
                {
                    candidate.stereo_y_error = std::abs(current_obs.uv.y() - current_obs.uvRight.y());
                    candidate.abs_disparity = std::abs(current_obs.uv.x() - current_obs.uvRight.x());
                    if (UNDERWATER_STEREO_MAX_Y_DIFF_PX > 0.0 &&
                        candidate.stereo_y_error > UNDERWATER_STEREO_MAX_Y_DIFF_PX)
                    {
                        reject_stereo_geometry++;
                        continue;
                    }
                    if (candidate.abs_disparity < UNDERWATER_STEREO_MIN_DISPARITY_PX ||
                        candidate.abs_disparity > UNDERWATER_STEREO_MAX_DISPARITY_PX)
                    {
                        reject_stereo_geometry++;
                        continue;
                    }
                }

                Vector3d ptsInCam = ric[0] * local_point + tic[0];
                Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame];
                candidate.point3d = cv::Point3f(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                candidate.point2d = cv::Point2f(current_obs.point.x(), current_obs.point.y());
                candidate.uv = cv::Point2f(current_obs.uv.x(), current_obs.uv.y());
                candidate.score = scorePnPCandidate(candidate);
                all_valid_candidates.push_back(candidate);

                if (mature_selection && track_count < min_track_count)
                {
                    reject_young++;
                    continue;
                }
                if (mature_selection && UNDERWATER_PNP_REQUIRE_STEREO && !has_stereo)
                {
                    reject_no_stereo++;
                    continue;
                }
                if (mature_selection && UNDERWATER_PNP_REQUIRE_CURRENT_STEREO && !current_obs.is_stereo)
                {
                    reject_current_no_stereo++;
                    continue;
                }
                bool stereo_requirements_ok = true;
                if (mature_selection && UNDERWATER_PNP_REQUIRE_STEREO && !has_stereo)
                    stereo_requirements_ok = false;
                if (mature_selection && UNDERWATER_PNP_REQUIRE_CURRENT_STEREO && !current_obs.is_stereo)
                    stereo_requirements_ok = false;
                if (stereo_requirements_ok)
                    fallback_candidates.push_back(candidate);
                pnp_candidates.push_back(candidate);
            }
        }
        if (UNDERWATER_MODE && UNDERWATER_PNP_MATURE_FEATURE_SELECTION)
        {
            std::sort(pnp_candidates.begin(), pnp_candidates.end(),
                      [](const PnPCandidate &a, const PnPCandidate &b)
                      {
                          return a.score > b.score;
                      });
            const int grid_rows = std::max(1, UNDERWATER_PNP_GRID_ROWS);
            const int grid_cols = std::max(1, UNDERWATER_PNP_GRID_COLS);
            const int cell_limit = std::max(0, UNDERWATER_PNP_CELL_MAX_FEATURES);
            const int max_candidates = std::max(0, UNDERWATER_PNP_MAX_CANDIDATES);
            selectPnPCandidateSubset(pnp_candidates, pts2D, pts3D,
                                     grid_rows, grid_cols, cell_limit, max_candidates);

            const int fallback_min_points = std::max(4, UNDERWATER_SOFT_MIN_PNP_INLIERS);
            if (static_cast<int>(pts2D.size()) < fallback_min_points &&
                static_cast<int>(fallback_candidates.size()) >= fallback_min_points)
            {
                std::sort(fallback_candidates.begin(), fallback_candidates.end(),
                          [](const PnPCandidate &a, const PnPCandidate &b)
                          {
                              return a.score > b.score;
                          });
                selectPnPCandidateSubset(fallback_candidates, pts2D, pts3D,
                                         grid_rows, grid_cols, cell_limit, max_candidates);
                ROS_INFO("underwater pnp fallback selected %zu stereo-contract candidates after strict selected %zu",
                         pts2D.size(), pnp_candidates.size());
            }
        }
        else
        {
            for (size_t i = 0; i < pnp_candidates.size(); i++)
            {
                pts3D.push_back(pnp_candidates[i].point3d);
                pts2D.push_back(pnp_candidates[i].point2d);
            }
        }
        Eigen::Matrix3d RCam;
        Eigen::Vector3d PCam;
        // trans to w_T_cam
        RCam = Rs[frameCnt - 1] * ric[0];
        PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];

        int pnp_inliers = 0;
        double pnp_rmse_px = -1.0;
        std::string reject_reason;
        if(solvePoseByPnP(RCam, PCam, pts2D, pts3D, &pnp_inliers, &pnp_rmse_px, &reject_reason))
        {
            Eigen::Matrix3d candidate_body_R = RCam * ric[0].transpose();
            if (UNDERWATER_MODE && UNDERWATER_GRAVITY_PNP_ENABLE)
            {
                candidate_body_R = applyUnderwaterGravityPnPConstraint(Rs[frameCnt - 1], candidate_body_R);
                RCam = candidate_body_R * ric[0];
            }
            Eigen::Vector3d candidate_P = -RCam * ric[0].transpose() * tic[0] + PCam;
            double pnp_step = (candidate_P - Ps[frameCnt - 1]).norm();
            Eigen::AngleAxisd pnp_angle_axis(Rs[frameCnt - 1].transpose() * candidate_body_R);
            double pnp_rotation = std::abs(pnp_angle_axis.angle());
            double rotation_limit = underwater_pnp_rotation_limit_rad > 0.0 ?
                underwater_pnp_rotation_limit_rad : UNDERWATER_PNP_MAX_ROTATION_RAD;
            rotation_limit = std::max(0.0, rotation_limit);
            if (UNDERWATER_MODE && rotation_limit > 0.0 &&
                (!std::isfinite(pnp_rotation) || pnp_rotation > rotation_limit))
            {
                ROS_WARN("underwater pnp rejected: pose rotation %.1fdeg > %.1fdeg",
                         pnp_rotation * 180.0 / M_PI,
                         rotation_limit * 180.0 / M_PI);
                appendUnderwaterPnPStats(frameCnt, raw_candidates, valid_depth_candidates,
                                         static_cast<int>(pnp_candidates.size()),
                                         static_cast<int>(pts2D.size()), pnp_inliers,
                                         pnp_rmse_px, false, pnp_step, pnp_rotation,
                                         "pose_rotation", reject_young, reject_no_stereo,
                                         reject_current_no_stereo, reject_stereo_geometry);
                return false;
            }
            double step_limit = UNDERWATER_PNP_MAX_STEP_M;
            if (underwater_pnp_step_limit_m > 0.0)
                step_limit = step_limit > 0.0 ? std::min(step_limit, underwater_pnp_step_limit_m)
                                               : underwater_pnp_step_limit_m;
            if (UNDERWATER_MODE && step_limit > 0.0 &&
                (!std::isfinite(pnp_step) || pnp_step > step_limit))
            {
                ROS_WARN("underwater pnp rejected: pose step %.3fm > %.3fm",
                         pnp_step, step_limit);
                appendUnderwaterPnPStats(frameCnt, raw_candidates, valid_depth_candidates,
                                         static_cast<int>(pnp_candidates.size()),
                                         static_cast<int>(pts2D.size()), pnp_inliers,
                                         pnp_rmse_px, false, pnp_step, pnp_rotation, "pose_step",
                                         reject_young, reject_no_stereo,
                                         reject_current_no_stereo, reject_stereo_geometry);
                return false;
            }
            // trans to w_T_imu
            Rs[frameCnt] = candidate_body_R; 
            Ps[frameCnt] = candidate_P;

            Eigen::Quaterniond Q(Rs[frameCnt]);
            //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
            //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
            appendUnderwaterPnPStats(frameCnt, raw_candidates, valid_depth_candidates,
                                     static_cast<int>(pnp_candidates.size()),
                                     static_cast<int>(pts2D.size()), pnp_inliers,
                                     pnp_rmse_px, true, pnp_step, pnp_rotation, "",
                                     reject_young, reject_no_stereo,
                                     reject_current_no_stereo, reject_stereo_geometry);
            return true;
        }
        appendUnderwaterPnPStats(frameCnt, raw_candidates, valid_depth_candidates,
                                 static_cast<int>(pnp_candidates.size()),
                                 static_cast<int>(pts2D.size()), pnp_inliers,
                                 pnp_rmse_px, false, -1.0, -1.0, reject_reason,
                                 reject_young, reject_no_stereo,
                                 reject_current_no_stereo, reject_stereo_geometry);
        return false;
    }
    return true;
}

void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        if (it_per_id.estimated_depth > 0)
            continue;

        if(STEREO && it_per_id.feature_per_frame[0].is_stereo)
        {
            double rectified_depth = -1.0;
            Eigen::Vector3d rectified_local_point;
            if (estimateRectifiedStereoDepth(it_per_id.feature_per_frame[0], tic, ric,
                                             &rectified_depth, &rectified_local_point) &&
                isUnderwaterDepthValueValid(rectified_depth, &rectified_local_point))
            {
                it_per_id.estimated_depth = rectified_depth;
                continue;
            }

            int imu_i = it_per_id.start_frame;
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;
            //cout << "left pose " << leftPose << endl;

            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;
            //cout << "right pose " << rightPose << endl;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2);
            point1 = it_per_id.feature_per_frame[0].pointRight.head(2);
            //cout << "point0 " << point0.transpose() << endl;
            //cout << "point1 " << point1.transpose() << endl;

            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double raw_depth = localPoint.z();
            double depth = correctUnderwaterRefractiveDepth(applyFeatureDepthSign(raw_depth));
            if (std::abs(raw_depth) > 1e-9 && std::isfinite(depth))
                localPoint *= depth / raw_depth;
            if (isUnderwaterDepthValueValid(depth, &localPoint))
                it_per_id.estimated_depth = depth;
            else if (!UNDERWATER_MODE)
                it_per_id.estimated_depth = INIT_DEPTH;
            else
            {
                it_per_id.estimated_depth = -1.0;
                it_per_id.solve_flag = 2;
            }
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue;
        }
        else if(it_per_id.feature_per_frame.size() > 1)
        {
            int imu_i = it_per_id.start_frame;
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;

            imu_i++;
            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[0];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2);
            point1 = it_per_id.feature_per_frame[1].point.head(2);
            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double raw_depth = localPoint.z();
            double depth = correctUnderwaterRefractiveDepth(applyFeatureDepthSign(raw_depth));
            if (std::abs(raw_depth) > 1e-9 && std::isfinite(depth))
                localPoint *= depth / raw_depth;
            if (isUnderwaterDepthValueValid(depth, &localPoint))
                it_per_id.estimated_depth = depth;
            else if (!UNDERWATER_MODE)
                it_per_id.estimated_depth = INIT_DEPTH;
            else
            {
                it_per_id.estimated_depth = -1.0;
                it_per_id.solve_flag = 2;
            }
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue;
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        assert(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = correctUnderwaterRefractiveDepth(applyFeatureDepthSign(svd_method));
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (!isUnderwaterDepthValueValid(it_per_id.estimated_depth))
        {
            if (!UNDERWATER_MODE)
                it_per_id.estimated_depth = INIT_DEPTH;
            else
            {
                it_per_id.estimated_depth = -1.0;
                it_per_id.solve_flag = 2;
            }
        }

    }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        int index = it->feature_id;
        itSet = outlierIndex.find(index);
        if(itSet != outlierIndex.end())
        {
            feature.erase(it);
            //printf("remove outlier %d \n", index);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
