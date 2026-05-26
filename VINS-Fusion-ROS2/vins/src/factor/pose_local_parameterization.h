/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/version.h>
#include "../utility/utility.h"

#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
#define VINS_USE_CERES_MANIFOLD 1
#else
#define VINS_USE_CERES_MANIFOLD 0
#endif

#if VINS_USE_CERES_MANIFOLD
class PoseLocalParameterization : public ceres::Manifold
#else
class PoseLocalParameterization : public ceres::LocalParameterization
#endif
{
public:
    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override;
#if VINS_USE_CERES_MANIFOLD
    bool PlusJacobian(const double *x, double *jacobian) const override;
    bool Minus(const double *y, const double *x, double *y_minus_x) const override;
    bool MinusJacobian(const double *x, double *jacobian) const override;
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }
#else
    bool ComputeJacobian(const double *x, double *jacobian) const override;
    int GlobalSize() const override { return 7; }
    int LocalSize() const override { return 6; }
#endif
};
