#pragma once
#include <eigen3/Eigen/Dense>
#include <utility>

class ExtendedKalmanFilter
{
public:
    static constexpr int N = 6; // [x y θ vx vy ω]ᵀ
    ExtendedKalmanFilter();

    std::pair<Eigen::VectorXd, Eigen::MatrixXd>
    step(const Eigen::VectorXd &x_prev,
         const Eigen::MatrixXd &P_prev,
         const Eigen::VectorXd &u,
         const Eigen::VectorXd &z,
         double dt);

    /* Getter */
    const Eigen::VectorXd &state() const { return x_; }
    const Eigen::MatrixXd &cov() const { return P_; }

    /* Tuning */
    void setProcessNoiseStd(double sa, double salpha);
    void setMeasurementNoise(const Eigen::Matrix<double, 6, 6> &R_in) { R6_ = R_in; }

private:
    /* Dynamik + Ableitungen */
    Eigen::VectorXd g(const Eigen::VectorXd &x,
                      const Eigen::VectorXd &u,
                      double dt) const;
    Eigen::MatrixXd F(const Eigen::VectorXd &x,
                      const Eigen::VectorXd &u,
                      double dt) const;
    Eigen::MatrixXd Qscaled(double dt) const;

    /* Member ------------------------------------------------------- */
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    Eigen::Matrix<double, 6, 6> R6_; // Messrauschen (6×6)
    double sigma_a2_, sigma_alpha2_;
    const Eigen::MatrixXd I_;
};
