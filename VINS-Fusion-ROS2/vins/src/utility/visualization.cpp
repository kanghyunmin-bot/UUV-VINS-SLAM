/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "visualization.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// using namespace ros;
using namespace Eigen;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry, pub_latest_odometry;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud, pub_margin_cloud;
rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_scan, pub_scan_world_debug;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
nav_msgs::msg::Path path;

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_keyframe_point;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic;

rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track_vins;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track_left;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track_right;
std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

size_t pub_counter = 0;

namespace
{
std::deque<geometry_msgs::msg::Point32> output_point_history;
constexpr size_t kMaxOutputPointHistory = 450;
constexpr int kStablePointCloudMaxObservationLag = 2;
constexpr double kLaserScanAngleMin = -M_PI;
constexpr double kLaserScanAngleMax = M_PI;
constexpr double kLaserScanAngleIncrement = M_PI / 180.0;
constexpr double kLaserScanMinRange = 0.05;

Vector3d outputPosition(const Vector3d &raw_P)
{
    return raw_P;
}

Vector3d outputVector(const Vector3d &raw_V)
{
    return raw_V;
}

bool isFiniteVector(const Vector3d &v)
{
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

struct InlinePointCloudRejectStats
{
    int too_young = 0;
    int no_stereo = 0;
    int bad_disparity = 0;
    int bad_depth = 0;
    int bad_range = 0;
    int invalid_point = 0;
};

int inlinePointCloudMinTrackCount()
{
    if (!UNDERWATER_MODE)
        return 2;
    if (!UNDERWATER_STABLE_POINT_CLOUD)
        return 2;
    return std::max(2, UNDERWATER_STABLE_POINT_CLOUD_MIN_TRACK_CNT);
}

bool passInlinePointDepthRangeGate(double depth, const Vector3d &pts_i,
                                   InlinePointCloudRejectStats *stats)
{
    if (!std::isfinite(depth) || depth <= 0.0)
    {
        if (stats != nullptr)
            stats->bad_depth++;
        return false;
    }

    if (!isFiniteVector(pts_i))
    {
        if (stats != nullptr)
            stats->invalid_point++;
        return false;
    }

    if (UNDERWATER_MODE)
    {
        if (depth < UNDERWATER_MIN_LANDMARK_DEPTH)
        {
            if (stats != nullptr)
                stats->bad_depth++;
            return false;
        }
        if (UNDERWATER_MAX_LANDMARK_DEPTH > 0.0 &&
            depth > UNDERWATER_MAX_LANDMARK_DEPTH)
        {
            if (stats != nullptr)
                stats->bad_depth++;
            return false;
        }
        if (UNDERWATER_STABLE_POINT_CLOUD_MAX_DEPTH > 0.0 &&
            depth > UNDERWATER_STABLE_POINT_CLOUD_MAX_DEPTH)
        {
            if (stats != nullptr)
                stats->bad_depth++;
            return false;
        }

        double range = pts_i.norm();
        if (!std::isfinite(range))
        {
            if (stats != nullptr)
                stats->bad_range++;
            return false;
        }
        if (UNDERWATER_MAX_LANDMARK_RANGE > 0.0 &&
            range > UNDERWATER_MAX_LANDMARK_RANGE)
        {
            if (stats != nullptr)
                stats->bad_range++;
            return false;
        }
        if (UNDERWATER_STABLE_POINT_CLOUD_MAX_RANGE > 0.0 &&
            range > UNDERWATER_STABLE_POINT_CLOUD_MAX_RANGE)
        {
            if (stats != nullptr)
                stats->bad_range++;
            return false;
        }
    }

    return true;
}

bool passInlinePointCloudFilter(const FeaturePerId &feature, Vector3d *local_point,
                                int *frame_offset,
                                InlinePointCloudRejectStats *stats = nullptr)
{
    const int used_num = static_cast<int>(feature.feature_per_frame.size());
    if (used_num < inlinePointCloudMinTrackCount())
    {
        if (stats != nullptr)
            stats->too_young++;
        return false;
    }
    if (feature.feature_per_frame.empty())
        return false;

    if (UNDERWATER_MODE && UNDERWATER_STABLE_POINT_CLOUD)
    {
        bool saw_stereo = false;
        bool saw_disparity = false;
        const int first_recent = std::max(0, used_num - 1 - kStablePointCloudMaxObservationLag);
        double baseline = 0.05;
        if (TIC.size() >= 2)
        {
            double configured_baseline = (TIC[0] - TIC[1]).norm();
            if (std::isfinite(configured_baseline) && configured_baseline > 1e-4)
                baseline = configured_baseline;
        }

        for (int i = used_num - 1; i >= first_recent; i--)
        {
            const FeaturePerFrame &observation = feature.feature_per_frame[i];
            if (!observation.is_stereo)
                continue;
            saw_stereo = true;

            double disparity = std::abs(observation.point.x() - observation.pointRight.x());
            if (!std::isfinite(disparity) || disparity < 1e-4)
                continue;
            saw_disparity = true;

            double depth = baseline / disparity;
            Vector3d pts_i = observation.point * depth;
            if (!passInlinePointDepthRangeGate(depth, pts_i, stats))
                continue;

            if (local_point != nullptr)
                *local_point = pts_i;
            if (frame_offset != nullptr)
                *frame_offset = i;
            return true;
        }

        // If the recent stereo measurement is not usable, fall back to the
        // VINS triangulated landmark below, still gated by age/depth/range.
        (void)saw_stereo;
        (void)saw_disparity;
    }

    if (feature.solve_flag == 1 &&
        std::isfinite(feature.estimated_depth) &&
        feature.estimated_depth > 0.0)
    {
        Vector3d pts_i = feature.feature_per_frame[0].point * feature.estimated_depth;
        if (passInlinePointDepthRangeGate(feature.estimated_depth, pts_i, stats))
        {
            if (local_point != nullptr)
                *local_point = pts_i;
            if (frame_offset != nullptr)
                *frame_offset = 0;
            return true;
        }
    }

    double baseline = 0.05;
    if (TIC.size() >= 2)
    {
        double configured_baseline = (TIC[0] - TIC[1]).norm();
        if (std::isfinite(configured_baseline) && configured_baseline > 1e-4)
            baseline = configured_baseline;
    }

    bool saw_stereo = false;
    bool saw_disparity = false;
    for (int i = 0; i < used_num; i++)
    {
        const FeaturePerFrame &observation = feature.feature_per_frame[i];
        if (!observation.is_stereo)
            continue;
        saw_stereo = true;

        double disparity = std::abs(observation.point.x() - observation.pointRight.x());
        if (!std::isfinite(disparity) || disparity < 1e-4)
            continue;
        saw_disparity = true;

        double depth = baseline / disparity;
        Vector3d pts_i = observation.point * depth;
        if (!passInlinePointDepthRangeGate(depth, pts_i, stats))
            continue;

        if (local_point != nullptr)
            *local_point = pts_i;
        if (frame_offset != nullptr)
            *frame_offset = i;
        return true;
    }

    if (!saw_stereo)
    {
        if (stats != nullptr)
            stats->no_stereo++;
    }
    else if (!saw_disparity)
    {
        if (stats != nullptr)
            stats->bad_disparity++;
    }

    return false;
}

Matrix3d outputRotation(const Matrix3d &raw_R)
{
    return raw_R;
}

Quaterniond odomOrientationForOutput(const Matrix3d &raw_R, const Vector3d &output_P)
{
    (void)output_P;
    return Quaterniond(raw_R);
}

double filteredScanRangeMax()
{
    double range_max = 5.0;
    if (UNDERWATER_STABLE_POINT_CLOUD_MAX_RANGE > 0.0)
        range_max = UNDERWATER_STABLE_POINT_CLOUD_MAX_RANGE;
    else if (UNDERWATER_MAX_LANDMARK_RANGE > 0.0)
        range_max = UNDERWATER_MAX_LANDMARK_RANGE;
    return std::max(range_max, kLaserScanMinRange + 0.1);
}

Matrix3d currentOutputBodyRotation(const Estimator &estimator)
{
    return estimator.Rs[WINDOW_SIZE];
}

void publishFilteredFeatureScan(const Estimator &estimator,
                                const std_msgs::msg::Header &header,
                                const std::vector<Vector3d> &current_world_points)
{
    if (!pub_scan)
        return;

    sensor_msgs::msg::LaserScan scan;
    scan.header = header;
    scan.header.frame_id = BODY_FRAME_ID;
    scan.angle_min = kLaserScanAngleMin;
    scan.angle_max = kLaserScanAngleMax;
    scan.angle_increment = kLaserScanAngleIncrement;
    scan.time_increment = 0.0;
    scan.scan_time = 0.1;
    scan.range_min = kLaserScanMinRange;
    scan.range_max = filteredScanRangeMax();

    const int bins = static_cast<int>(
        std::floor((scan.angle_max - scan.angle_min) / scan.angle_increment)
    ) + 1;
    scan.ranges.assign(bins, std::numeric_limits<float>::infinity());

    const Vector3d body_origin = outputPosition(estimator.Ps[WINDOW_SIZE]);
    const Matrix3d body_R = currentOutputBodyRotation(estimator);
    const double z_gate = std::max(0.25, 0.5 * scan.range_max);
    int finite_bins = 0;

    for (const Vector3d &world_point : current_world_points)
    {
        if (!isFiniteVector(world_point))
            continue;

        Vector3d body_point = body_R.transpose() * (world_point - body_origin);
        if (!isFiniteVector(body_point) || std::abs(body_point.z()) > z_gate)
            continue;

        const double range = std::hypot(body_point.x(), body_point.y());
        if (!std::isfinite(range) || range < scan.range_min || range > scan.range_max)
            continue;

        const double angle = std::atan2(body_point.y(), body_point.x());
        if (angle < scan.angle_min || angle > scan.angle_max)
            continue;

        const int idx = static_cast<int>(std::floor((angle - scan.angle_min) / scan.angle_increment));
        if (idx < 0 || idx >= bins)
            continue;

        float &slot = scan.ranges[idx];
        if (!std::isfinite(slot))
            finite_bins++;
        slot = std::min(slot, static_cast<float>(range));
    }

    pub_scan->publish(scan);

    if (pub_scan_world_debug)
    {
        sensor_msgs::msg::LaserScan world_scan = scan;
        world_scan.header.frame_id = WORLD_FRAME_ID;
        world_scan.range_max = std::max(
            scan.range_max,
            static_cast<float>(body_origin.head<2>().norm() + scan.range_max)
        );
        world_scan.ranges.assign(bins, std::numeric_limits<float>::infinity());

        for (const Vector3d &world_point : current_world_points)
        {
            if (!isFiniteVector(world_point))
                continue;

            const double range = std::hypot(world_point.x(), world_point.y());
            if (!std::isfinite(range) || range < world_scan.range_min || range > world_scan.range_max)
                continue;

            const double angle = std::atan2(world_point.y(), world_point.x());
            if (angle < world_scan.angle_min || angle > world_scan.angle_max)
                continue;

            const int idx = static_cast<int>(std::floor((angle - world_scan.angle_min) / world_scan.angle_increment));
            if (idx < 0 || idx >= bins)
                continue;

            float &slot = world_scan.ranges[idx];
            slot = std::min(slot, static_cast<float>(range));
        }
        pub_scan_world_debug->publish(world_scan);
    }

    if (UNDERWATER_MODE && (pub_counter % 30 == 0))
        ROS_INFO("underwater /scan from filtered VINS features: %d finite bins / %d ranges, %zu current points",
                 finite_bins, bins, current_world_points.size());
}
}

void registerPub(rclcpp::Node::SharedPtr n)
{
    pub_latest_odometry = n->create_publisher<nav_msgs::msg::Odometry>("imu_propagate", 10);
    pub_path = n->create_publisher<nav_msgs::msg::Path>("path", 10);
    pub_odometry = n->create_publisher<nav_msgs::msg::Odometry>("odometry", 10);
    pub_point_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("point_cloud", 10);
    pub_scan = n->create_publisher<sensor_msgs::msg::LaserScan>("scan", 10);
    pub_scan_world_debug = n->create_publisher<sensor_msgs::msg::LaserScan>("scan_world_debug", 10);
    pub_margin_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("margin_cloud", 10);
    pub_key_poses = n->create_publisher<visualization_msgs::msg::Marker>("key_poses", 10);
    pub_camera_pose = n->create_publisher<nav_msgs::msg::Odometry>("camera_pose", 10);
    pub_camera_pose_visual = n->create_publisher<visualization_msgs::msg::MarkerArray>("camera_pose_visual", 10);
    pub_keyframe_pose = n->create_publisher<nav_msgs::msg::Odometry>("keyframe_pose", 10);
    pub_keyframe_point = n->create_publisher<sensor_msgs::msg::PointCloud>("keyframe_point", 10);
    pub_extrinsic = n->create_publisher<nav_msgs::msg::Odometry>("extrinsic", 10);
    pub_image_track = n->create_publisher<sensor_msgs::msg::Image>("image_track", 1);
    pub_image_track_vins = n->create_publisher<sensor_msgs::msg::Image>("/vins_estimator/image_track", 1);
    pub_image_track_left = n->create_publisher<sensor_msgs::msg::Image>("/vins_estimator/image_track_left", 1);
    pub_image_track_right = n->create_publisher<sensor_msgs::msg::Image>("/vins_estimator/image_track_right", 1);
    if (PUBLISH_VINS_TF)
        tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(n);
    else
        tf_broadcaster.reset();

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    nav_msgs::msg::Odometry odometry;

    int sec_ts = (int)t;
    uint nsec_ts = (uint)((t - sec_ts) * 1e9);
    odometry.header.stamp.sec = sec_ts;
    odometry.header.stamp.nanosec = nsec_ts;

    odometry.header.frame_id = WORLD_FRAME_ID;
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry->publish(odometry);
}

void pubTrackImage(
    const cv::Mat &imgTrack,
    const double t,
    const cv::Mat &leftTrack,
    const cv::Mat &rightTrack
)
{
    std_msgs::msg::Header header;
    header.frame_id = WORLD_FRAME_ID;

    int sec_ts = (int)t;
    uint nsec_ts = (uint)((t - sec_ts) * 1e9);
    header.stamp.sec = sec_ts;
    header.stamp.nanosec = nsec_ts;

    sensor_msgs::msg::Image::SharedPtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
    pub_image_track->publish(*imgTrackMsg);
    pub_image_track_vins->publish(*imgTrackMsg);

    if (!leftTrack.empty() && !rightTrack.empty())
    {
        sensor_msgs::msg::Image::SharedPtr leftMsg = cv_bridge::CvImage(header, "bgr8", leftTrack).toImageMsg();
        sensor_msgs::msg::Image::SharedPtr rightMsg = cv_bridge::CvImage(header, "bgr8", rightTrack).toImageMsg();
        pub_image_track_left->publish(*leftMsg);
        pub_image_track_right->publish(*rightMsg);
        return;
    }

    int stereo_split_col = 0;
    if (COL > 0 && imgTrack.cols >= COL * 2)
    {
        stereo_split_col = COL;
    }
    else if (imgTrack.cols > 2 && imgTrack.cols % 2 == 0)
    {
        // The underwater D435i replay can publish a stereo debug image whose
        // runtime width differs from the configured COL. Keep left/right RViz
        // panels as true single-camera views instead of showing the full
        // side-by-side debug image in the right panel.
        stereo_split_col = imgTrack.cols / 2;
    }

    if (stereo_split_col > 0 && stereo_split_col < imgTrack.cols && imgTrack.rows > 0)
    {
        cv::Mat leftTrack = imgTrack(cv::Rect(0, 0, stereo_split_col, imgTrack.rows)).clone();
        cv::Mat rightTrack = imgTrack(
            cv::Rect(stereo_split_col, 0, imgTrack.cols - stereo_split_col, imgTrack.rows)
        ).clone();
        sensor_msgs::msg::Image::SharedPtr leftMsg = cv_bridge::CvImage(header, "bgr8", leftTrack).toImageMsg();
        sensor_msgs::msg::Image::SharedPtr rightMsg = cv_bridge::CvImage(header, "bgr8", rightTrack).toImageMsg();
        pub_image_track_left->publish(*leftMsg);
        pub_image_track_right->publish(*rightMsg);
    }
    else
    {
        pub_image_track_left->publish(*imgTrackMsg);
        pub_image_track_right->publish(*imgTrackMsg);
    }
}


void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    //printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
    // ROS_DEBUG_STREAM("position: " << estimator.Ps[WINDOW_SIZE].transpose());
    // ROS_DEBUG_STREAM("orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    if (ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //ROS_DEBUG("calibration result for camera %d", i);
            // ROS_DEBUG_STREAM("extirnsic tic: " << estimator.tic[i].transpose());
            // ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
            eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_DEBUG("sum of path %f", sum_of_path);
    if (ESTIMATE_TD)
        ROS_INFO("td %f", estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = WORLD_FRAME_ID;
        odometry.child_frame_id = BODY_FRAME_ID;
        Vector3d output_P = outputPosition(estimator.Ps[WINDOW_SIZE]);
        Quaterniond tmp_Q = odomOrientationForOutput(estimator.Rs[WINDOW_SIZE], output_P);
        Vector3d output_V = outputVector(estimator.Vs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = output_P.x();
        odometry.pose.pose.position.y = output_P.y();
        odometry.pose.pose.position.z = output_P.z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = output_V.x();
        odometry.twist.twist.linear.y = output_V.y();
        odometry.twist.twist.linear.z = output_V.z();
        pub_odometry->publish(odometry);

        if (tf_broadcaster)
        {
            geometry_msgs::msg::TransformStamped transform;
            transform.header = odometry.header;
            transform.child_frame_id = BODY_FRAME_ID;
            transform.transform.translation.x = odometry.pose.pose.position.x;
            transform.transform.translation.y = odometry.pose.pose.position.y;
            transform.transform.translation.z = odometry.pose.pose.position.z;
            transform.transform.rotation = odometry.pose.pose.orientation;
            tf_broadcaster->sendTransform(transform);
        }

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = WORLD_FRAME_ID;
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = WORLD_FRAME_ID;
        path.poses.push_back(pose_stamped);
        pub_path->publish(path);

        // write result to file
        ofstream foutC(VINS_RESULT_PATH, ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(9);
        foutC << header.stamp.sec + header.stamp.nanosec * (1e-9) << ",";
        foutC.precision(5);
        foutC << output_P.x() << ","
              << output_P.y() << ","
              << output_P.z() << ","
              << tmp_Q.w() << ","
              << tmp_Q.x() << ","
              << tmp_Q.y() << ","
              << tmp_Q.z() << ","
              << output_V.x() << ","
              << output_V.y() << ","
              << output_V.z() << "," << endl;
        foutC.close();
        Eigen::Vector3d tmp_T = output_P;
        printf("time: %f, t: %f %f %f q: %f %f %f %f \n", header.stamp.sec + header.stamp.nanosec * (1e-9),
                                                          tmp_T.x(), tmp_T.y(), tmp_T.z(),
                                                          tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z());
    }
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::msg::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = WORLD_FRAME_ID;
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::msg::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = rclcpp::Duration(0, 0);

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::msg::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = outputPosition(estimator.key_poses[i]);
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses->publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = outputPosition(estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0]);
        Quaterniond R = Quaterniond(outputRotation(estimator.Rs[i] * estimator.ric[0]));

        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = WORLD_FRAME_ID;
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_camera_pose->publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        if(STEREO)
        {
            Vector3d P = outputPosition(estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1]);
            Quaterniond R = Quaterniond(outputRotation(estimator.Rs[i] * estimator.ric[1]));
            cameraposevisual.add_pose(P, R);
        }
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    sensor_msgs::msg::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;
    point_cloud.header.frame_id = WORLD_FRAME_ID;
    loop_point_cloud.header.frame_id = WORLD_FRAME_ID;

    int cloud_candidates = 0;
    int cloud_published = 0;
    InlinePointCloudRejectStats cloud_rejects;
    std::vector<Vector3d> current_filtered_world_points;

    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num = it_per_id.feature_per_frame.size();
        if (used_num < inlinePointCloudMinTrackCount())
            continue;
        if (it_per_id.start_frame < 0 || it_per_id.start_frame > WINDOW_SIZE)
            continue;
        Vector3d pts_i;
        int frame_offset = 0;
        cloud_candidates++;
        if (!passInlinePointCloudFilter(it_per_id, &pts_i, &frame_offset, &cloud_rejects))
            continue;
        int imu_i = it_per_id.start_frame + frame_offset;
        if (imu_i < 0 || imu_i > WINDOW_SIZE)
            continue;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];
        if (!isFiniteVector(w_pts_i))
            continue;
        w_pts_i = outputPosition(w_pts_i);

        geometry_msgs::msg::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
        current_filtered_world_points.push_back(w_pts_i);
        cloud_published++;
    }
    const int cloud_current_filtered = cloud_published;
    if (UNDERWATER_MODE && UNDERWATER_STABLE_POINT_CLOUD)
    {
        for (const auto &p : point_cloud.points)
            output_point_history.push_back(p);
        while (output_point_history.size() > kMaxOutputPointHistory)
            output_point_history.pop_front();
        point_cloud.points.assign(output_point_history.begin(), output_point_history.end());
    }
    pub_point_cloud->publish(point_cloud);
    publishFilteredFeatureScan(estimator, header, current_filtered_world_points);
    if (UNDERWATER_MODE && (pub_counter++ % 30 == 0))
    {
        ROS_INFO("underwater point_cloud inline filter: %d / %d current, %zu history points published "
                 "(reject young=%d no_stereo=%d disparity=%d depth=%d range=%d invalid=%d)",
                 cloud_current_filtered, cloud_candidates, point_cloud.points.size(),
                 cloud_rejects.too_young,
                 cloud_rejects.no_stereo,
                 cloud_rejects.bad_disparity,
                 cloud_rejects.bad_depth,
                 cloud_rejects.bad_range,
                 cloud_rejects.invalid_point);
    }


    // pub margined potin
    sensor_msgs::msg::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    { 
        int used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        Vector3d pts_i;
        int frame_offset = 0;
        if (it_per_id.start_frame == 0 && passInlinePointCloudFilter(it_per_id, &pts_i, &frame_offset))
        {
            int imu_i = it_per_id.start_frame + frame_offset;
            if (imu_i < 0 || imu_i > WINDOW_SIZE)
                continue;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];
            if (!isFiniteVector(w_pts_i))
                continue;
            w_pts_i = outputPosition(w_pts_i);

            geometry_msgs::msg::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud->publish(margin_cloud);
}



void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    return; // tmp.


    cout << "tf 1" << endl;
    if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;

    std::shared_ptr<tf2_ros::TransformBroadcaster> br;
    geometry_msgs::msg::TransformStamped transform, transform_cam;

    tf2::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    
    cout << "tf 2" << endl;
    correct_t = estimator.Ps[WINDOW_SIZE];
    correct_q = estimator.Rs[WINDOW_SIZE];

    cout << "tf 3" << endl;

    
    cout << header.stamp.sec + header.stamp.nanosec * (1e-9) << endl;
    cout << correct_t << endl;
    cout << correct_q.w() << " " << correct_q.x() << " " << correct_q.y() << " " << correct_q.z() << endl;


    // transform.header.stamp = header.stamp;
    transform.header.frame_id = WORLD_FRAME_ID;
    transform.child_frame_id = BODY_FRAME_ID;

    transform.transform.translation.x = correct_t(0);
    transform.transform.translation.y = correct_t(1);
    transform.transform.translation.z = correct_t(2);

    cout << "tf 4" << endl;


    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    transform.transform.rotation.w = q.w();

    cout << "tf 5" << endl;

    br->sendTransform(transform);


    cout << "tf 6" << endl;



    // camera frame
    transform_cam.header.stamp = header.stamp;
    transform_cam.header.frame_id = BODY_FRAME_ID;
    transform_cam.child_frame_id = CAMERA_FRAME_ID;


    transform_cam.transform.translation.x = estimator.tic[0].x();
    transform_cam.transform.translation.y = estimator.tic[0].y();
    transform_cam.transform.translation.z = estimator.tic[0].z();

    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());

    transform_cam.transform.rotation.x = q.x();
    transform_cam.transform.rotation.y = q.y();
    transform_cam.transform.rotation.z = q.z();
    transform_cam.transform.rotation.w = q.w();

    // br->sendTransform(transform_cam);

    cout << "tf 7" << endl;

    
    nav_msgs::msg::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = WORLD_FRAME_ID;
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();

    cout << "tf 8" << endl;
    pub_extrinsic->publish(odometry);
    cout << "tf 9" << endl;

}


// void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header)
// {
//     if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
//         return;
//     std::shared_ptr<tf2_ros::TransformBroadcaster> br;
//     tf2::Transform transform;
//     tf2::Quaternion q;
//     // body frame
//     Vector3d correct_t;
//     Quaterniond correct_q;
//     correct_t = estimator.Ps[WINDOW_SIZE];
//     correct_q = estimator.Rs[WINDOW_SIZE];

//     transform.setOrigin(tf2::Vector3(correct_t(0),
//                                     correct_t(1),
//                                     correct_t(2)));
//     q.setW(correct_q.w());
//     q.setX(correct_q.x());
//     q.setY(correct_q.y());
//     q.setZ(correct_q.z());
//     transform.setRotation(q);
//     // br->sendTransform(tf2::StampedTransform(transform, header.stamp, "world", "body"));
//     br->sendTransform(tf2::StampedTransform(transform, header.stamp, "world", "body"));

//     // camera frame
//     transform.setOrigin(tf2::Vector3(estimator.tic[0].x(),
//                                     estimator.tic[0].y(),
//                                     estimator.tic[0].z()));
//     q.setW(Quaterniond(estimator.ric[0]).w());
//     q.setX(Quaterniond(estimator.ric[0]).x());
//     q.setY(Quaterniond(estimator.ric[0]).y());
//     q.setZ(Quaterniond(estimator.ric[0]).z());
//     transform.setRotation(q);
//     // br->sendTransform(tf2::StampedTransform(transform, header.stamp, "body", "camera"));
//     br->sendTransform(tf2::StampedTransform(transform, header.stamp, "body", "camera"));

    
//     nav_msgs::msg::Odometry odometry;
//     odometry.header = header;
//     odometry.header.frame_id = WORLD_FRAME_ID;
//     odometry.pose.pose.position.x = estimator.tic[0].x();
//     odometry.pose.pose.position.y = estimator.tic[0].y();
//     odometry.pose.pose.position.z = estimator.tic[0].z();
//     Quaterniond tmp_q{estimator.ric[0]};
//     odometry.pose.pose.orientation.x = tmp_q.x();
//     odometry.pose.pose.orientation.y = tmp_q.y();
//     odometry.pose.pose.orientation.z = tmp_q.z();
//     odometry.pose.pose.orientation.w = tmp_q.w();
//     pub_extrinsic->publish(odometry);

// }

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Vector3d P = outputPosition(estimator.Ps[i]);
        Quaterniond R = Quaterniond(outputRotation(estimator.Rs[i]));

        nav_msgs::msg::Odometry odometry;

        int sec_ts = (int)estimator.Headers[WINDOW_SIZE - 2];
        uint nsec_ts = (uint)((estimator.Headers[WINDOW_SIZE - 2] - sec_ts) * 1e9);
        odometry.header.stamp.sec = sec_ts;
        odometry.header.stamp.nanosec = nsec_ts;

        odometry.header.frame_id = WORLD_FRAME_ID;
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.sec, P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose->publish(odometry);


        sensor_msgs::msg::PointCloud point_cloud;

        sec_ts = (int)estimator.Headers[WINDOW_SIZE - 2];
        nsec_ts = (uint)((estimator.Headers[WINDOW_SIZE - 2] - sec_ts) * 1e9);
        point_cloud.header.stamp.sec = sec_ts;
        point_cloud.header.stamp.nanosec = nsec_ts;

        point_cloud.header.frame_id = WORLD_FRAME_ID;
        for (auto &it_per_id : estimator.f_manager.feature)
        {
            int frame_size = it_per_id.feature_per_frame.size();
            Vector3d pts_i;
            int frame_offset = 0;
            if(it_per_id.start_frame < WINDOW_SIZE - 2 &&
               it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 &&
               passInlinePointCloudFilter(it_per_id, &pts_i, &frame_offset))
            {

                int imu_i = it_per_id.start_frame + frame_offset;
                if (imu_i < 0 || imu_i > WINDOW_SIZE)
                    continue;
                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                                      + estimator.Ps[imu_i];
                if (!isFiniteVector(w_pts_i))
                    continue;
                w_pts_i = outputPosition(w_pts_i);
                geometry_msgs::msg::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::msg::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);
            }

        }
        pub_keyframe_point->publish(point_cloud);
    }
}
