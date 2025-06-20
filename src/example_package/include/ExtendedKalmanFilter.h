#pragma once

#include <eigen3/Eigen/Dense>
#include <functional>

class ExtendedKalmanFilter
{
public:
    ExtendedKalmanFilter(double delta_t);

    void initialize(const Eigen::VectorXd &mu0, const Eigen::MatrixXd &Sigma0);
    void setProcessNoise(const Eigen::MatrixXd &Q);
    void setMeasurementNoise(const Eigen::MatrixXd &R);

    // Set nonlinear functions and their Jacobians
    void setStateTransitionFunction(
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&, double)> g,
        std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&, double)> G_jac);
    void setMeasurementFunction(
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
        std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H_jac);

    void predict(const Eigen::VectorXd &u);
    void updateMeasurement(const Eigen::VectorXd &z);

    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;

private:
    double delta_t_;
    int n_;
    bool initialized_ = false;

    Eigen::VectorXd mu_;
    Eigen::MatrixXd Sigma_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;

    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&, double)> g_;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&, double)> G_jac_;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h_;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H_jac_;
};
