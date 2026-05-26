#ifndef EIGENQUATERNIONPARAMETERIZATION_H
#define EIGENQUATERNIONPARAMETERIZATION_H

#include <ceres/ceres.h>
#include <ceres/version.h>

#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
#define CAMODOCAL_USE_CERES_MANIFOLD 1
#else
#define CAMODOCAL_USE_CERES_MANIFOLD 0
#endif

namespace camodocal
{

#if CAMODOCAL_USE_CERES_MANIFOLD
class EigenQuaternionParameterization : public ceres::Manifold
#else
class EigenQuaternionParameterization : public ceres::LocalParameterization
#endif
{
public:
    virtual ~EigenQuaternionParameterization() {}
    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const override;
#if CAMODOCAL_USE_CERES_MANIFOLD
    bool PlusJacobian(const double* x,
                      double* jacobian) const override;
    bool Minus(const double* y,
               const double* x,
               double* y_minus_x) const override;
    bool MinusJacobian(const double* x,
                       double* jacobian) const override;
    int AmbientSize() const override { return 4; }
    int TangentSize() const override { return 3; }
#else
    bool ComputeJacobian(const double* x,
                         double* jacobian) const override;
    int GlobalSize() const override { return 4; }
    int LocalSize() const override { return 3; }
#endif

private:
    template<typename T>
    void EigenQuaternionProduct(const T z[4], const T w[4], T zw[4]) const;
};


template<typename T>
void
EigenQuaternionParameterization::EigenQuaternionProduct(const T z[4], const T w[4], T zw[4]) const
{
    zw[0] = z[3] * w[0] + z[0] * w[3] + z[1] * w[2] - z[2] * w[1];
    zw[1] = z[3] * w[1] - z[0] * w[2] + z[1] * w[3] + z[2] * w[0];
    zw[2] = z[3] * w[2] + z[0] * w[1] - z[1] * w[0] + z[2] * w[3];
    zw[3] = z[3] * w[3] - z[0] * w[0] - z[1] * w[1] - z[2] * w[2];
}

}

#endif
