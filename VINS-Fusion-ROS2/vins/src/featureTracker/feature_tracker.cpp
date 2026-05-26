/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "feature_tracker.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace
{
struct NccMatch
{
    bool ok = false;
    cv::Point2f point;
    double score = -1.0;
    double margin = 0.0;
};

void applyUnderwaterStructureMask(const cv::Mat &img, cv::Mat &mask)
{
    if (!UNDERWATER_MODE || !UNDERWATER_FEATURE_STRUCTURE_MASK ||
        img.empty() || img.type() != CV_8UC1 || mask.empty())
        return;

    cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad_mask;
    cv::Sobel(img, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(img, grad_y, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0.0, grad_mask);

    const double min_gradient = std::max(0.0, UNDERWATER_FEATURE_MIN_GRADIENT);
    cv::threshold(grad_mask, grad_mask, min_gradient, 255, cv::THRESH_BINARY);

    int support_window = std::max(3, UNDERWATER_FEATURE_SUPPORT_WINDOW);
    if (support_window % 2 == 0)
        support_window += 1;

    cv::Mat grad_unit, support_count, support_mask;
    grad_mask.convertTo(grad_unit, CV_32F, 1.0 / 255.0);
    cv::boxFilter(
        grad_unit,
        support_count,
        CV_32F,
        cv::Size(support_window, support_window),
        cv::Point(-1, -1),
        false
    );

    const double min_support = std::max(1.0, UNDERWATER_FEATURE_MIN_SUPPORT_PIXELS);
    cv::threshold(support_count, support_mask, min_support, 255.0, cv::THRESH_BINARY);
    support_mask.convertTo(support_mask, CV_8UC1);

    cv::Mat support_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(support_mask, support_mask, cv::MORPH_CLOSE, support_kernel);
    cv::bitwise_and(mask, support_mask, mask);
}

void rejectSmallBrightBlobs(const cv::Mat &img, cv::Mat &mask)
{
    if (!UNDERWATER_MODE || img.empty() || img.type() != CV_8UC1 || mask.empty())
        return;

    const int threshold = std::max(0, std::min(255, UNDERWATER_FEATURE_BRIGHT_BLOB_THRESHOLD));
    const int max_area = std::max(0, UNDERWATER_FEATURE_MAX_BLOB_AREA);
    if (threshold >= 255 || max_area <= 0)
        return;

    cv::Mat bright;
    cv::threshold(img, bright, threshold, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(bright) == 0)
        return;

    cv::Mat labels, stats, centroids;
    int components = cv::connectedComponentsWithStats(bright, labels, stats, centroids, 8, CV_32S);
    cv::Mat reject = cv::Mat::zeros(img.size(), CV_8UC1);
    for (int label = 1; label < components; label++)
    {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        // Pool wall grid lines are large/connected; isolated small bright
        // components are usually suspended particles or specular debris.
        if (area <= max_area && width <= 35 && height <= 35)
            reject.setTo(255, labels == label);
    }

    int dilate_px = std::max(0, UNDERWATER_FEATURE_BLOB_DILATE_PX);
    if (dilate_px > 0 && cv::countNonZero(reject) > 0)
    {
        int k = 2 * dilate_px + 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        cv::dilate(reject, reject, kernel);
    }

    if (cv::countNonZero(reject) > 0)
        mask.setTo(0, reject);
}

cv::Size temporalLkWindow()
{
    int win_size = UNDERWATER_MODE ? std::max(21, UNDERWATER_LK_WIN_SIZE) : 21;
    if (win_size % 2 == 0)
        win_size += 1;
    return cv::Size(win_size, win_size);
}

int temporalLkMaxLevel()
{
    if (!UNDERWATER_MODE)
        return 3;
    return std::max(3, UNDERWATER_LK_MAX_LEVEL);
}

int countStatus(const vector<uchar> &status)
{
    int count = 0;
    for (size_t i = 0; i < status.size(); i++)
        if (status[i])
            count++;
    return count;
}

void appendUnderwaterFrontendStats(double stamp,
                                   int prev_tracks,
                                   int temporal_tracked,
                                   int new_features,
                                   int left_total,
                                   int stereo_candidates,
                                   int stereo_lk_ok,
                                   int stereo_after_ncc,
                                   int stereo_final,
                                   int quad_rejected,
                                   int ncc_attempted,
                                   int ncc_recovered,
                                   double mean_left_flow_x,
                                   double mean_left_flow_y,
                                   double mean_left_flow_norm,
                                   double mean_stereo_disparity,
                                   double elapsed_ms)
{
    if (!UNDERWATER_MODE || !UNDERWATER_FRONTEND_STATS_LOG || OUTPUT_FOLDER.empty())
        return;

    static bool initialized = false;
    const std::string path = OUTPUT_FOLDER + "/underwater_frontend_stats.csv";
    std::ofstream fout;
    if (!initialized)
    {
        fout.open(path, std::ios::out);
        fout << "stamp,prev_tracks,temporal_tracked,new_features,left_total,"
             << "stereo_candidates,stereo_lk_ok,stereo_after_ncc,stereo_final,"
             << "quad_rejected,ncc_attempted,ncc_recovered,"
             << "mean_left_flow_x,mean_left_flow_y,mean_left_flow_norm,"
             << "mean_stereo_disparity,elapsed_ms\n";
        initialized = true;
    }
    else
    {
        fout.open(path, std::ios::app);
    }
    if (!fout.is_open())
        return;
    fout << std::fixed << std::setprecision(9) << stamp << ","
         << prev_tracks << ","
         << temporal_tracked << ","
         << new_features << ","
         << left_total << ","
         << stereo_candidates << ","
         << stereo_lk_ok << ","
         << stereo_after_ncc << ","
         << stereo_final << ","
         << quad_rejected << ","
         << ncc_attempted << ","
         << ncc_recovered << ","
         << std::setprecision(4) << mean_left_flow_x << ","
         << mean_left_flow_y << ","
         << mean_left_flow_norm << ","
         << mean_stereo_disparity << ","
         << std::setprecision(3) << elapsed_ms << "\n";
}

int stereoNccPatchSize()
{
    int patch_size = std::max(5, UNDERWATER_STEREO_NCC_PATCH_SIZE);
    if (patch_size % 2 == 0)
        patch_size += 1;
    return patch_size;
}

bool patchFits(const cv::Mat &img, const cv::Point2f &pt, int radius)
{
    return !img.empty() &&
           pt.x >= radius &&
           pt.y >= radius &&
           pt.x < img.cols - radius &&
           pt.y < img.rows - radius;
}

NccMatch searchEpipolarNcc(const cv::Mat &src,
                           const cv::Mat &dst,
                           const cv::Point2f &src_pt,
                           int disparity_direction)
{
    NccMatch match;
    if (src.empty() || dst.empty() || src.type() != CV_8UC1 || dst.type() != CV_8UC1)
        return match;

    int patch_size = stereoNccPatchSize();
    int radius = patch_size / 2;
    if (!patchFits(src, src_pt, radius))
        return match;

    const double min_disp = std::max(0.0, UNDERWATER_STEREO_MIN_DISPARITY_PX);
    const double max_disp = std::max(min_disp, UNDERWATER_STEREO_MAX_DISPARITY_PX);
    const int y_radius = std::max(0, UNDERWATER_STEREO_NCC_Y_RADIUS_PX);

    double center_x_a = src_pt.x + disparity_direction * min_disp;
    double center_x_b = src_pt.x + disparity_direction * max_disp;
    int center_x_min = static_cast<int>(std::floor(std::min(center_x_a, center_x_b)));
    int center_x_max = static_cast<int>(std::ceil(std::max(center_x_a, center_x_b)));
    int center_y_min = static_cast<int>(std::floor(src_pt.y - y_radius));
    int center_y_max = static_cast<int>(std::ceil(src_pt.y + y_radius));

    center_x_min = std::max(radius, center_x_min);
    center_y_min = std::max(radius, center_y_min);
    center_x_max = std::min(dst.cols - radius - 1, center_x_max);
    center_y_max = std::min(dst.rows - radius - 1, center_y_max);
    if (center_x_min > center_x_max || center_y_min > center_y_max)
        return match;

    cv::Rect template_roi(
        cvRound(src_pt.x) - radius,
        cvRound(src_pt.y) - radius,
        patch_size,
        patch_size
    );
    if ((template_roi & cv::Rect(0, 0, src.cols, src.rows)) != template_roi)
        return match;

    cv::Rect search_roi(
        center_x_min - radius,
        center_y_min - radius,
        center_x_max - center_x_min + patch_size,
        center_y_max - center_y_min + patch_size
    );
    if ((search_roi & cv::Rect(0, 0, dst.cols, dst.rows)) != search_roi ||
        search_roi.width < patch_size || search_roi.height < patch_size)
        return match;

    cv::Mat template_patch = src(template_roi);
    cv::Scalar mean, stddev;
    cv::meanStdDev(template_patch, mean, stddev);
    if (stddev[0] < 3.0)
        return match;

    cv::Mat result;
    cv::matchTemplate(dst(search_roi), template_patch, result, cv::TM_CCOEFF_NORMED);
    if (result.empty())
        return match;

    double min_val = 0.0;
    double max_val = 0.0;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
    if (!std::isfinite(max_val) || max_val < UNDERWATER_STEREO_NCC_MIN_SCORE)
        return match;
    const cv::Point best_loc = max_loc;

    double second_val = -1.0;
    if (UNDERWATER_STEREO_NCC_MIN_MARGIN > 0.0 && result.cols * result.rows > 1)
    {
        cv::Mat suppressed = result.clone();
        int suppress_radius = std::max(2, radius);
        cv::Rect suppress_rect(
            std::max(0, max_loc.x - suppress_radius),
            std::max(0, max_loc.y - suppress_radius),
            std::min(result.cols - std::max(0, max_loc.x - suppress_radius), 2 * suppress_radius + 1),
            std::min(result.rows - std::max(0, max_loc.y - suppress_radius), 2 * suppress_radius + 1)
        );
        suppressed(suppress_rect).setTo(-1.0);
        cv::minMaxLoc(suppressed, &min_val, &second_val, &min_loc, &max_loc);
        if (std::isfinite(second_val) &&
            max_val - second_val < UNDERWATER_STEREO_NCC_MIN_MARGIN)
            return match;
    }

    cv::Point2f dst_pt(
        static_cast<float>(search_roi.x + best_loc.x + radius),
        static_cast<float>(search_roi.y + best_loc.y + radius)
    );
    if (!patchFits(dst, dst_pt, radius))
        return match;

    double disparity = disparity_direction < 0 ?
        static_cast<double>(src_pt.x - dst_pt.x) :
        static_cast<double>(dst_pt.x - src_pt.x);
    if (!std::isfinite(disparity) || disparity < min_disp || disparity > max_disp)
        return match;
    if (std::abs(static_cast<double>(src_pt.y - dst_pt.y)) >
        std::max(0, UNDERWATER_STEREO_NCC_Y_RADIUS_PX))
        return match;

    match.ok = true;
    match.point = dst_pt;
    match.score = max_val;
    match.margin = std::isfinite(second_val) ? max_val - second_val : max_val;
    return match;
}
}

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
    stereo_fallback_log_counter = 0;
    last_ncc_attempted = 0;
    last_ncc_recovered = 0;
    last_quad_rejected = 0;
}

void FeatureTracker::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
    if (UNDERWATER_MODE &&
        (UNDERWATER_FEATURE_MIN_INTENSITY > 0 || UNDERWATER_FEATURE_MAX_INTENSITY < 255) &&
        !cur_img.empty() && cur_img.type() == CV_8UC1)
    {
        cv::Mat intensity_mask;
        int min_i = std::max(0, std::min(255, UNDERWATER_FEATURE_MIN_INTENSITY));
        int max_i = std::max(0, std::min(255, UNDERWATER_FEATURE_MAX_INTENSITY));
        if (min_i < max_i)
        {
            cv::inRange(cur_img, cv::Scalar(min_i), cv::Scalar(max_i), intensity_mask);
            cv::bitwise_and(mask, intensity_mask, mask);
        }
    }
    applyUnderwaterStructureMask(cur_img, mask);
    rejectSmallBrightBlobs(cur_img, mask);

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        cur_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::detectNewFeaturesGrid(int max_new_features)
{
    n_pts.clear();
    if (max_new_features <= 0)
        return;
    if (mask.empty())
        cout << "mask is empty " << endl;
    if (mask.type() != CV_8UC1)
        cout << "mask type wrong " << endl;

    if (UNDERWATER_GRID_ROWS <= 1 || UNDERWATER_GRID_COLS <= 1 || UNDERWATER_CELL_MAX_FEATURES <= 0)
    {
        cv::goodFeaturesToTrack(cur_img, n_pts, max_new_features, 0.01, MIN_DIST, mask);
        return;
    }

    int cell_h = std::max(1, row / UNDERWATER_GRID_ROWS);
    int cell_w = std::max(1, col / UNDERWATER_GRID_COLS);
    for (int gy = 0; gy < UNDERWATER_GRID_ROWS && static_cast<int>(n_pts.size()) < max_new_features; gy++)
    {
        for (int gx = 0; gx < UNDERWATER_GRID_COLS && static_cast<int>(n_pts.size()) < max_new_features; gx++)
        {
            int x0 = gx * cell_w;
            int y0 = gy * cell_h;
            int x1 = (gx == UNDERWATER_GRID_COLS - 1) ? col : std::min(col, x0 + cell_w);
            int y1 = (gy == UNDERWATER_GRID_ROWS - 1) ? row : std::min(row, y0 + cell_h);
            if (x1 <= x0 || y1 <= y0)
                continue;

            cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
            vector<cv::Point2f> cell_pts;
            int remaining = max_new_features - static_cast<int>(n_pts.size());
            int cell_limit = std::min(UNDERWATER_CELL_MAX_FEATURES, remaining);
            cv::goodFeaturesToTrack(cur_img(roi), cell_pts, cell_limit, 0.01, MIN_DIST, mask(roi));
            for (auto &p : cell_pts)
            {
                p.x += static_cast<float>(x0);
                p.y += static_cast<float>(y0);
                n_pts.push_back(p);
            }
        }
    }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img.clone();
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1.empty() ? cv::Mat() : _img1.clone();
    const int prev_track_input = static_cast<int>(prev_pts.size());
    int temporal_tracked_count = 0;
    int new_feature_count = 0;
    int left_total_after_add = 0;
    int stereo_candidate_count = 0;
    int stereo_lk_ok_count = 0;
    int stereo_after_ncc_count = 0;
    int stereo_final_count = 0;
    double mean_left_flow_x = 0.0;
    double mean_left_flow_y = 0.0;
    double mean_left_flow_norm = 0.0;
    double mean_stereo_disparity = 0.0;
    last_ncc_attempted = 0;
    last_ncc_recovered = 0;
    last_quad_rejected = 0;
    if (UNDERWATER_MODE && UNDERWATER_CLAHE_ENABLE && cur_img.type() == CV_8UC1)
    {
        int tile_size = std::max(2, UNDERWATER_CLAHE_TILE_SIZE);
        double clip_limit = std::max(0.1, UNDERWATER_CLAHE_CLIP_LIMIT);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(tile_size, tile_size));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty() && rightImg.type() == CV_8UC1)
            clahe->apply(rightImg, rightImg);
    }
    cur_pts.clear();

    if (prev_pts.size() > 0)
    {
        vector<uchar> status;
        if(!USE_GPU_ACC_FLOW)
        {
            TicToc t_o;
            
            vector<float> err;
            if(hasPrediction)
            {
                cur_pts = predict_pts;
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, temporalLkWindow(), 1,
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                
                int succ_num = 0;
                for (size_t i = 0; i < status.size(); i++)
                {
                    if (status[i])
                        succ_num++;
                }
                if (succ_num < 10)
                    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, temporalLkWindow(), temporalLkMaxLevel());
            }
            else
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, temporalLkWindow(), temporalLkMaxLevel());
            // reverse check
            if(FLOW_BACK)
            {
                const double flow_back_threshold =
                    UNDERWATER_MODE ? std::max(0.5, UNDERWATER_FLOW_BACK_THRESHOLD_PX) : 0.5;
                vector<uchar> reverse_status;
                vector<cv::Point2f> reverse_pts = prev_pts;
                cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, temporalLkWindow(), 1,
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= flow_back_threshold)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }
            }
            // printf("temporal optical flow costs: %fms\n", t_o.toc());
        }
#ifdef GPU_MODE
        else
        {
            TicToc t_og;
            cv::cuda::GpuMat prev_gpu_img(prev_img);
            cv::cuda::GpuMat cur_gpu_img(cur_img);
            cv::cuda::GpuMat prev_gpu_pts(prev_pts);
            cv::cuda::GpuMat cur_gpu_pts(cur_pts);
            cv::cuda::GpuMat gpu_status;
            if(hasPrediction)
            {
                cur_gpu_pts = cv::cuda::GpuMat(predict_pts);
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 1, 30, true);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);
                
                vector<cv::Point2f> tmp_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp_cur_pts);
                cur_pts = tmp_cur_pts;

                vector<uchar> tmp_status(gpu_status.cols);
                gpu_status.download(tmp_status);
                status = tmp_status;

                int succ_num = 0;
                for (size_t i = 0; i < tmp_status.size(); i++)
                {
                    if (tmp_status[i])
                        succ_num++;
                }
                if (succ_num < 10)
                {
                    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 3, 30, false);
                    d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                    vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                    cur_gpu_pts.download(tmp1_cur_pts);
                    cur_pts = tmp1_cur_pts;

                    vector<uchar> tmp1_status(gpu_status.cols);
                    gpu_status.download(tmp1_status);
                    status = tmp1_status;
                }
            }
            else
            {
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp1_cur_pts);
                cur_pts = tmp1_cur_pts;

                vector<uchar> tmp1_status(gpu_status.cols);
                gpu_status.download(tmp1_status);
                status = tmp1_status;
            }
            if(FLOW_BACK)
            {
                cv::cuda::GpuMat reverse_gpu_status;
                cv::cuda::GpuMat reverse_gpu_pts = prev_gpu_pts;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 1, 30, true);
                d_pyrLK_sparse->calc(cur_gpu_img, prev_gpu_img, cur_gpu_pts, reverse_gpu_pts, reverse_gpu_status);

                vector<cv::Point2f> reverse_pts(reverse_gpu_pts.cols);
                reverse_gpu_pts.download(reverse_pts);

                vector<uchar> reverse_status(reverse_gpu_status.cols);
                reverse_gpu_status.download(reverse_status);

                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }
            }
            // printf("gpu temporal optical flow costs: %f ms\n",t_og.toc());
        }
#endif
    
        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        if (UNDERWATER_MODE && UNDERWATER_TEMPORAL_RANSAC)
            rejectWithF();
        temporal_tracked_count = static_cast<int>(cur_pts.size());
        if (UNDERWATER_MODE && prev_pts.size() == cur_pts.size() && !cur_pts.empty())
        {
            for (size_t i = 0; i < cur_pts.size(); i++)
            {
                const double dx = static_cast<double>(cur_pts[i].x - prev_pts[i].x);
                const double dy = static_cast<double>(cur_pts[i].y - prev_pts[i].y);
                mean_left_flow_x += dx;
                mean_left_flow_y += dy;
                mean_left_flow_norm += std::sqrt(dx * dx + dy * dy);
            }
            const double inv_count = 1.0 / static_cast<double>(cur_pts.size());
            mean_left_flow_x *= inv_count;
            mean_left_flow_y *= inv_count;
            mean_left_flow_norm *= inv_count;
        }
        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        
        //printf("track cnt %d\n", (int)ids.size());
    }

    for (auto &n : track_cnt)
        n++;

    if (1)
    {
        //rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        // ROS_DEBUG("set mask costs %fms", t_m.toc());
        // printf("set mask costs %fms\n", t_m.toc());
        ROS_DEBUG("detect feature begins");
        
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if(!USE_GPU)
        {
            if (n_max_cnt > 0)
            {
                TicToc t_t;
                detectNewFeaturesGrid(MAX_CNT - cur_pts.size());
                new_feature_count = static_cast<int>(n_pts.size());
                // printf("good feature to track costs: %fms\n", t_t.toc());
                ROS_DEBUG("new feature points: %zu", n_pts.size());
            }
            else
                n_pts.clear();
            // sum_n += n_pts.size();
            // printf("total point from non-gpu: %d\n",sum_n);
        }
#ifdef GPU_MODE
        // ROS_DEBUG("detect feature costs: %fms", t_t.toc());
        // printf("good feature to track costs: %fms\n", t_t.toc());
        else
        {
            if (n_max_cnt > 0)
            {
                if(mask.empty())
                    cout << "mask is empty " << endl;
                if (mask.type() != CV_8UC1)
                    cout << "mask type wrong " << endl;
                TicToc t_g;
                cv::cuda::GpuMat cur_gpu_img(cur_img);
                cv::cuda::GpuMat d_prevPts;
                TicToc t_gg;
                cv::cuda::GpuMat gpu_mask(mask);
                // printf("gpumat cost: %fms\n",t_gg.toc());
                cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(cur_gpu_img.type(), MAX_CNT - cur_pts.size(), 0.01, MIN_DIST);
                // cout << "new gpu points: "<< MAX_CNT - cur_pts.size()<<endl;
                detector->detect(cur_gpu_img, d_prevPts, gpu_mask);
                // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
                if(!d_prevPts.empty())
                    n_pts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
                else
                    n_pts.clear();
                // sum_n += n_pts.size();
                // printf("total point from gpu: %d\n",sum_n);
                // printf("gpu good feature to track cost: %fms\n", t_g.toc());
            }
            else 
                n_pts.clear();
        }
#endif

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        left_total_after_add = static_cast<int>(cur_pts.size());
        // ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
        // printf("selectFeature costs: %fms\n", t_a.toc());
    }

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    if(!_img1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty())
        {
            stereo_candidate_count = static_cast<int>(cur_pts.size());
            //printf("stereo image; track feature on right image\n");
            
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            if(!USE_GPU_ACC_FLOW)
            {
                TicToc t_check;
                vector<float> err;
                // cur left ---- cur right
                cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, temporalLkWindow(), temporalLkMaxLevel());
                // reverse check cur right ---- cur left
                if(FLOW_BACK)
                {
                    cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, temporalLkWindow(), temporalLkMaxLevel());
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) &&
                           distance(cur_pts[i], reverseLeftPts[i]) <=
                               (UNDERWATER_MODE ? std::max(0.5, UNDERWATER_FLOW_BACK_THRESHOLD_PX) : 0.5))
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }
                }
                // printf("left right optical flow cost %fms\n",t_check.toc());
            }
#ifdef GPU_MODE
            else
            {
                TicToc t_og1;
                cv::cuda::GpuMat cur_gpu_img(cur_img);
                cv::cuda::GpuMat right_gpu_Img(rightImg);
                cv::cuda::GpuMat cur_gpu_pts(cur_pts);
                cv::cuda::GpuMat cur_right_gpu_pts;
                cv::cuda::GpuMat gpu_status;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(cur_gpu_img, right_gpu_Img, cur_gpu_pts, cur_right_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp_cur_right_pts(cur_right_gpu_pts.cols);
                cur_right_gpu_pts.download(tmp_cur_right_pts);
                cur_right_pts = tmp_cur_right_pts;

                vector<uchar> tmp_status(gpu_status.cols);
                gpu_status.download(tmp_status);
                status = tmp_status;

                if(FLOW_BACK)
                {   
                    cv::cuda::GpuMat reverseLeft_gpu_Pts;
                    cv::cuda::GpuMat status_gpu_RightLeft;
                    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 3, 30, false);
                    d_pyrLK_sparse->calc(right_gpu_Img, cur_gpu_img, cur_right_gpu_pts, reverseLeft_gpu_Pts, status_gpu_RightLeft);

                    vector<cv::Point2f> tmp_reverseLeft_Pts(reverseLeft_gpu_Pts.cols);
                    reverseLeft_gpu_Pts.download(tmp_reverseLeft_Pts);
                    reverseLeftPts = tmp_reverseLeft_Pts;

                    vector<uchar> tmp1_status(status_gpu_RightLeft.cols);
                    status_gpu_RightLeft.download(tmp1_status);
                    statusRightLeft = tmp1_status;
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }
                }
                // printf("gpu left right optical flow cost %fms\n",t_og1.toc());
            }
#endif
            stereo_lk_ok_count = countStatus(status);
            if (UNDERWATER_MODE && UNDERWATER_STEREO_MAX_Y_DIFF_PX > 0.0)
            {
                recoverStereoWithNccFallback(rightImg, status);
                {
                    for (size_t i = 0; i < status.size(); i++)
                    {
                        if (!status[i])
                            continue;
                        double y_diff = std::abs(static_cast<double>(cur_pts[i].y - cur_right_pts[i].y));
                        if (y_diff > UNDERWATER_STEREO_MAX_Y_DIFF_PX)
                            status[i] = 0;
                    }
                }
            }
            else if (UNDERWATER_MODE)
                recoverStereoWithNccFallback(rightImg, status);
            stereo_after_ncc_count = countStatus(status);
            if (UNDERWATER_MODE)
                rejectWithQuadStereoTemporal(rightImg, status);
            stereo_final_count = countStatus(status);
            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            if (UNDERWATER_MODE && !ids_right.empty())
            {
                map<int, cv::Point2f> left_pts_by_id;
                for (size_t i = 0; i < ids.size(); i++)
                    left_pts_by_id[ids[i]] = cur_pts[i];
                double disparity_sum = 0.0;
                int disparity_count = 0;
                for (size_t i = 0; i < ids_right.size(); i++)
                {
                    auto it = left_pts_by_id.find(ids_right[i]);
                    if (it == left_pts_by_id.end())
                        continue;
                    disparity_sum += static_cast<double>(it->second.x - cur_right_pts[i].x);
                    disparity_count++;
                }
                if (disparity_count > 0)
                    mean_stereo_disparity = disparity_sum / static_cast<double>(disparity_count);
            }
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
            
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
        prevRightPtsMap.clear();
        for(size_t i = 0; i < ids_right.size(); i++)
            prevRightPtsMap[ids_right[i]] = cur_right_pts[i];
    }
    if(SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    prev_img = cur_img;
    if (!rightImg.empty() && stereo_cam)
        prev_right_img = rightImg.clone();
    else
    {
        prev_right_img.release();
        prevRightPtsMap.clear();
    }
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }

    //printf("feature track whole time %f\n", t_r.toc());
    appendUnderwaterFrontendStats(
        cur_time,
        prev_track_input,
        temporal_tracked_count,
        new_feature_count,
        left_total_after_add,
        stereo_candidate_count,
        stereo_lk_ok_count,
        stereo_after_ncc_count,
        stereo_final_count,
        last_quad_rejected,
        last_ncc_attempted,
        last_ncc_recovered,
        mean_left_flow_x,
        mean_left_flow_y,
        mean_left_flow_norm,
        mean_stereo_disparity,
        t_r.toc()
    );
    return featureFrame;
}

void FeatureTracker::recoverStereoWithNccFallback(const cv::Mat &rightImg, vector<uchar> &status)
{
    if (!UNDERWATER_MODE || !UNDERWATER_STEREO_NCC_FALLBACK || !stereo_cam ||
        cur_img.empty() || rightImg.empty() || cur_pts.empty() ||
        cur_right_pts.size() != cur_pts.size() || status.size() != cur_pts.size())
        return;
    last_ncc_attempted = 0;
    last_ncc_recovered = 0;

    const double stereo_y_limit = std::max(0.0, UNDERWATER_STEREO_MAX_Y_DIFF_PX);
    const double min_disp = std::max(0.0, UNDERWATER_STEREO_MIN_DISPARITY_PX);
    const double max_disp = std::max(min_disp, UNDERWATER_STEREO_MAX_DISPARITY_PX);
    const double fb_limit = std::max(0.5, UNDERWATER_FLOW_BACK_THRESHOLD_PX);
    const int min_track_count = std::max(1, UNDERWATER_STEREO_NCC_MIN_TRACK_CNT);
    const double temporal_prediction_limit = std::max(
        3.0,
        std::max(UNDERWATER_QUAD_RIGHT_CLOSURE_THRESHOLD_PX,
                 UNDERWATER_QUAD_MOTION_CONSISTENCY_THRESHOLD_PX)
    );
    vector<double> signed_disparities;
    signed_disparities.reserve(cur_pts.size());
    for (size_t i = 0; i < cur_pts.size(); i++)
    {
        if (!status[i] || !inBorder(cur_right_pts[i]))
            continue;
        double y_diff = std::abs(static_cast<double>(cur_pts[i].y - cur_right_pts[i].y));
        double disparity = static_cast<double>(cur_pts[i].x - cur_right_pts[i].x);
        double abs_disparity = std::abs(disparity);
        if ((stereo_y_limit <= 0.0 || y_diff <= stereo_y_limit) &&
            std::isfinite(disparity) &&
            abs_disparity >= min_disp &&
            abs_disparity <= max_disp)
            signed_disparities.push_back(disparity);
    }
    static int dominant_disparity_sign = 0;
    int expected_disparity_sign = dominant_disparity_sign;
    int left_to_right_direction = -1;
    if (signed_disparities.size() >= 8)
    {
        size_t middle = signed_disparities.size() / 2;
        std::nth_element(signed_disparities.begin(),
                         signed_disparities.begin() + middle,
                         signed_disparities.end());
        expected_disparity_sign = signed_disparities[middle] < 0.0 ? -1 : 1;
        if (dominant_disparity_sign == 0)
            dominant_disparity_sign = expected_disparity_sign;
        else
            expected_disparity_sign = dominant_disparity_sign;
        if (expected_disparity_sign < 0)
            left_to_right_direction = 1;
    }

    struct RecoveryCandidate
    {
        size_t index;
        int track_count;
        bool has_temporal_prediction;
    };
    vector<RecoveryCandidate> recovery_candidates;
    recovery_candidates.reserve(cur_pts.size());
    for (size_t i = 0; i < cur_pts.size(); i++)
    {
        bool needs_recovery = !status[i] || !inBorder(cur_right_pts[i]);
        if (!needs_recovery)
        {
            double y_diff = std::abs(static_cast<double>(cur_pts[i].y - cur_right_pts[i].y));
            double disparity = static_cast<double>(cur_pts[i].x - cur_right_pts[i].x);
            double abs_disparity = std::abs(disparity);
            if ((stereo_y_limit > 0.0 && y_diff > stereo_y_limit) ||
                !std::isfinite(disparity) ||
                abs_disparity < min_disp ||
                abs_disparity > max_disp ||
                (expected_disparity_sign != 0 && disparity * expected_disparity_sign <= 0.0))
                needs_recovery = true;
        }
        if (!needs_recovery)
            continue;
        status[i] = 0;
        if (i >= track_cnt.size() || track_cnt[i] < min_track_count)
            continue;
        auto prev_right_it = prevRightPtsMap.find(ids[i]);
        auto prev_left_it = prevLeftPtsMap.find(ids[i]);
        const bool has_temporal_prediction =
            prev_right_it != prevRightPtsMap.end() && prev_left_it != prevLeftPtsMap.end();
        if (UNDERWATER_STEREO_NCC_REQUIRE_TEMPORAL && !has_temporal_prediction)
            continue;
        recovery_candidates.push_back({i, track_cnt[i], has_temporal_prediction});
    }

    std::sort(recovery_candidates.begin(), recovery_candidates.end(),
              [](const RecoveryCandidate &a, const RecoveryCandidate &b)
              {
                  if (a.has_temporal_prediction != b.has_temporal_prediction)
                      return a.has_temporal_prediction > b.has_temporal_prediction;
                  return a.track_count > b.track_count;
              });
    const int max_candidates = std::max(0, UNDERWATER_STEREO_NCC_MAX_CANDIDATES);
    if (max_candidates > 0 &&
        static_cast<int>(recovery_candidates.size()) > max_candidates)
        recovery_candidates.resize(max_candidates);

    int attempted = 0;
    int recovered = 0;
    for (size_t candidate_i = 0; candidate_i < recovery_candidates.size(); candidate_i++)
    {
        size_t i = recovery_candidates[candidate_i].index;
        auto prev_right_it = prevRightPtsMap.find(ids[i]);
        auto prev_left_it = prevLeftPtsMap.find(ids[i]);
        const bool has_temporal_prediction =
            prev_right_it != prevRightPtsMap.end() && prev_left_it != prevLeftPtsMap.end();
        cv::Point2f predicted_right;
        if (has_temporal_prediction)
        {
            predicted_right = prev_right_it->second + (cur_pts[i] - prev_left_it->second);
            if (!inBorder(predicted_right))
                continue;
        }

        attempted++;
        NccMatch left_to_right = searchEpipolarNcc(cur_img, rightImg, cur_pts[i], left_to_right_direction);
        if (!left_to_right.ok && expected_disparity_sign == 0)
            left_to_right = searchEpipolarNcc(cur_img, rightImg, cur_pts[i], -left_to_right_direction);
        if (!left_to_right.ok)
            continue;
        if (has_temporal_prediction && distance(predicted_right, left_to_right.point) > temporal_prediction_limit)
            continue;

        if (UNDERWATER_STEREO_NCC_FORWARD_BACK)
        {
            NccMatch right_to_left = searchEpipolarNcc(rightImg, cur_img, left_to_right.point, -left_to_right_direction);
            if (!right_to_left.ok || distance(cur_pts[i], right_to_left.point) > fb_limit)
                continue;
        }

        cur_right_pts[i] = left_to_right.point;
        status[i] = 1;
        recovered++;
    }
    last_ncc_attempted = attempted;
    last_ncc_recovered = recovered;

    stereo_fallback_log_counter++;
    if (attempted > 0 && (recovered > 0 || stereo_fallback_log_counter % 30 == 0))
    {
        ROS_INFO("underwater stereo NCC fallback recovered %d / %d candidates "
                 "(score>=%.2f disp=[%.1f, %.1f] y_radius=%d direction=%d min_track=%d temporal=%d)",
                 recovered,
                 attempted,
                 UNDERWATER_STEREO_NCC_MIN_SCORE,
                 min_disp,
                 max_disp,
                 std::max(0, UNDERWATER_STEREO_NCC_Y_RADIUS_PX),
                 left_to_right_direction,
                 min_track_count,
                 UNDERWATER_STEREO_NCC_REQUIRE_TEMPORAL);
    }
}

void FeatureTracker::rejectWithQuadStereoTemporal(const cv::Mat &rightImg, vector<uchar> &status)
{
    if (!UNDERWATER_MODE || !UNDERWATER_QUAD_ENABLE || !stereo_cam || prev_right_img.empty() || rightImg.empty())
        return;
    if (cur_pts.empty() || cur_right_pts.size() != cur_pts.size() || status.size() != cur_pts.size())
        return;

    const double stereo_y_limit = std::max(0.0, UNDERWATER_STEREO_MAX_Y_DIFF_PX);
    const double closure_limit = std::max(0.0, UNDERWATER_QUAD_RIGHT_CLOSURE_THRESHOLD_PX);
    const double right_fb_limit = std::max(0.0, UNDERWATER_QUAD_RIGHT_FB_THRESHOLD_PX);
    const double motion_limit = std::max(0.0, UNDERWATER_QUAD_MOTION_CONSISTENCY_THRESHOLD_PX);
    const bool need_motion_evidence = motion_limit > 0.0;

    vector<cv::Point2f> prev_right_pts;
    vector<cv::Point2f> prev_left_pts;
    vector<int> candidate_indices;
    prev_right_pts.reserve(cur_pts.size());
    prev_left_pts.reserve(cur_pts.size());
    candidate_indices.reserve(cur_pts.size());
    for (size_t i = 0; i < ids.size(); i++)
    {
        if (!status[i])
            continue;
        auto prev_right_it = prevRightPtsMap.find(ids[i]);
        if (prev_right_it == prevRightPtsMap.end())
            continue;

        cv::Point2f prev_left_pt;
        if (need_motion_evidence)
        {
            auto prev_left_it = prevLeftPtsMap.find(ids[i]);
            if (prev_left_it == prevLeftPtsMap.end())
                continue;
            prev_left_pt = prev_left_it->second;
            if (stereo_y_limit > 0.0)
            {
                double prev_y_diff = std::abs(static_cast<double>(prev_left_it->second.y - prev_right_it->second.y));
                if (prev_y_diff > stereo_y_limit)
                    continue;
            }
        }
        prev_right_pts.push_back(prev_right_it->second);
        prev_left_pts.push_back(prev_left_pt);
        candidate_indices.push_back(static_cast<int>(i));
    }
    if (static_cast<int>(candidate_indices.size()) < UNDERWATER_QUAD_MIN_CANDIDATES)
        return;

    vector<cv::Point2f> temporal_right_pts;
    vector<uchar> temporal_status;
    vector<float> temporal_err;
    cv::calcOpticalFlowPyrLK(prev_right_img, rightImg, prev_right_pts, temporal_right_pts,
                             temporal_status, temporal_err, temporalLkWindow(), temporalLkMaxLevel());
    if (temporal_right_pts.size() != prev_right_pts.size() ||
        temporal_status.size() != prev_right_pts.size())
        return;

    vector<cv::Point2f> reverse_prev_right_pts;
    vector<uchar> reverse_status;
    vector<float> reverse_err;
    if (right_fb_limit > 0.0)
    {
        reverse_prev_right_pts = prev_right_pts;
        cv::calcOpticalFlowPyrLK(rightImg, prev_right_img, temporal_right_pts, reverse_prev_right_pts,
                                 reverse_status, reverse_err, temporalLkWindow(), 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
    }

    vector<uchar> original_status = status;
    int rejected_closure = 0;
    int rejected_motion = 0;
    int skipped_temporal = 0;
    int skipped_right_fb = 0;
    for (size_t k = 0; k < candidate_indices.size(); k++)
    {
        int idx = candidate_indices[k];
        bool reject = false;
        bool right_temporal_ok = temporal_status[k] && inBorder(temporal_right_pts[k]);
        if (!right_temporal_ok)
        {
            skipped_temporal++;
            continue;
        }

        if (right_fb_limit > 0.0)
        {
            bool right_fb_ok = k < reverse_status.size() && reverse_status[k] &&
                               k < reverse_prev_right_pts.size() &&
                               inBorder(reverse_prev_right_pts[k]);
            double right_fb_error = right_fb_ok ? distance(prev_right_pts[k], reverse_prev_right_pts[k])
                                                : std::numeric_limits<double>::infinity();
            if (!right_fb_ok || right_fb_error > right_fb_limit)
            {
                skipped_right_fb++;
                continue;
            }
        }

        if (closure_limit > 0.0)
        {
            double dx = temporal_right_pts[k].x - cur_right_pts[idx].x;
            double dy = temporal_right_pts[k].y - cur_right_pts[idx].y;
            double closure_error = sqrt(dx * dx + dy * dy);
            if (closure_error > closure_limit)
            {
                reject = true;
                rejected_closure++;
            }
        }

        if (motion_limit > 0.0)
        {
            cv::Point2f left_flow = cur_pts[idx] - prev_left_pts[k];
            cv::Point2f right_flow = temporal_right_pts[k] - prev_right_pts[k];
            double dx = static_cast<double>(left_flow.x - right_flow.x);
            double dy = static_cast<double>(left_flow.y - right_flow.y);
            double flow_delta = sqrt(dx * dx + dy * dy);
            if (flow_delta > motion_limit)
            {
                reject = true;
                rejected_motion++;
            }
        }

        if (reject)
            status[idx] = 0;
    }

    int kept = 0;
    for (size_t i = 0; i < status.size(); i++)
        if (status[i])
            kept++;
    int rejected = rejected_closure + rejected_motion;
    last_quad_rejected = rejected;
    if (kept < UNDERWATER_QUAD_MIN_TRACKS_AFTER_REJECTION)
    {
        status = original_status;
        if (rejected > 0)
        {
            ROS_DEBUG("underwater quad rollback: rejected %d would leave %d tracks (< %d)",
                      rejected, kept, UNDERWATER_QUAD_MIN_TRACKS_AFTER_REJECTION);
        }
        return;
    }
    else if (rejected > 0)
    {
        ROS_DEBUG("underwater quad rejected %d tracks, kept %d (closure=%d motion=%d skipped_temporal=%d skipped_fb=%d)",
                  rejected, kept, rejected_closure, rejected_motion, skipped_temporal, skipped_right_fb);
    }
}

void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        if (status.size() != cur_pts.size())
            return;
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        if (cur_un_pts.size() == status.size())
            reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    imTrackLeft.release();
    imTrackRight.release();
    imTrack.release();

    if (imLeft.channels() == 1)
        cv::cvtColor(imLeft, imTrackLeft, cv::COLOR_GRAY2BGR);
    else
        imTrackLeft = imLeft.clone();

    if (!imRight.empty() && stereo_cam)
    {
        if (imRight.channels() == 1)
            cv::cvtColor(imRight, imTrackRight, cv::COLOR_GRAY2BGR);
        else
            imTrackRight = imRight.clone();
    }

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrackLeft, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imTrackRight.empty())
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::circle(imTrackRight, curRightPts[i], 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrackLeft, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    if (!imTrackRight.empty())
        cv::hconcat(imTrackLeft, imTrackRight, imTrack);
    else
        imTrack = imTrackLeft.clone();

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}


void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}


void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}


cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}

cv::Mat FeatureTracker::getTrackImageLeft()
{
    return imTrackLeft;
}

cv::Mat FeatureTracker::getTrackImageRight()
{
    return imTrackRight;
}
