#pragma once
#include <eigen3/Eigen/Dense>
#include <utility>

class KalmanFilter
{
public:
    static constexpr int N = 6;               // [x y θ vx vy ω]^T
    KalmanFilter();

    std::pair<Eigen::VectorXd,Eigen::MatrixXd>
    step(const Eigen::VectorXd&  x_prev,
         const Eigen::MatrixXd&  P_prev,
         const Eigen::VectorXd&  u,
         const Eigen::VectorXd&  z,
         double dt);

    const Eigen::VectorXd& state() const { return x_; }
    const Eigen::MatrixXd& cov()   const { return P_; }

    void setProcessNoiseStd(double sa,double salpha);
    void setMeasurementNoise(const Eigen::Matrix<double,6,6>& R_in);

private:
    Eigen::MatrixXd Qscaled(double dt) const;

    Eigen::MatrixXd A_,B_,H_,I_,R_;
    double sigma_a2_,sigma_alpha2_;
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
};
