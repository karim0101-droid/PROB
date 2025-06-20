#include "ExtendedKalmanFilter.h"

ExtendedKalmanFilter::ExtendedKalmanFilter(double delta_t)
    : delta_t_(delta_t), initialized_(false) {}

void ExtendedKalmanFilter::initialize(const Eigen::VectorXd &mu0, const Eigen::MatrixXd &Sigma0)
{
    mu_ = mu0;
    Sigma_ = Sigma0;
    n_ = mu0.size();
    initialized_ = true;
}

void ExtendedKalmanFilter::setProcessNoise(const Eigen::MatrixXd &Q) { Q_ = Q; }
void ExtendedKalmanFilter::setMeasurementNoise(const Eigen::MatrixXd &R) { R_ = R; }

void ExtendedKalmanFilter::setStateTransitionFunction(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&, double)> g,
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&, double)> G_jac)
{
    g_ = g;
    G_jac_ = G_jac;
}

void ExtendedKalmanFilter::setMeasurementFunction(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H_jac)
{
    h_ = h;
    H_jac_ = H_jac;
}

void ExtendedKalmanFilter::predict(const Eigen::VectorXd &u)
{
    if (!initialized_) return;
    Eigen::VectorXd mu_bar = g_(u, mu_, delta_t_);
    Eigen::MatrixXd G = G_jac_(u, mu_, delta_t_);
    Sigma_ = G * Sigma_ * G.transpose() + Q_;
    mu_ = mu_bar;
}

void ExtendedKalmanFilter::updateMeasurement(const Eigen::VectorXd &z)
{
    if (!initialized_) return;
    Eigen::VectorXd z_hat = h_(mu_);
    Eigen::MatrixXd H = H_jac_(mu_);

    Eigen::VectorXd y = z - z_hat;
    Eigen::MatrixXd S = H * Sigma_ * H.transpose() + R_;
    Eigen::MatrixXd K = Sigma_ * H.transpose() * S.inverse();

    mu_ = mu_ + K * y;
    Sigma_ = (Eigen::MatrixXd::Identity(n_, n_) - K * H) * Sigma_;
}

Eigen::VectorXd ExtendedKalmanFilter::getState() const { return mu_; }
Eigen::MatrixXd ExtendedKalmanFilter::getCovariance() const { return Sigma_; }
