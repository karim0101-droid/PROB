#include "KalmanFilter.h"
#include <cmath>
#include <iostream>

KalmanFilter::KalmanFilter(double delta_t)
    : n_(6), m_(5), delta_t_(delta_t)
{
    // Systemmatrix A
    A_ = Eigen::MatrixXd::Identity(n_, n_);
    A_(0, 3) = delta_t_;
    A_(1, 4) = delta_t_;
    A_(2, 5) = delta_t_;

    // Steuermatrix B
    B_ = Eigen::MatrixXd::Zero(n_, 2);

    // Messmatrix C (x, y, theta, v_x, v_y)
    C_ = Eigen::MatrixXd::Zero(m_, n_);
    C_(0, 0) = 1; // x
    C_(1, 1) = 1; // y
    C_(2, 2) = 1; // theta
    C_(3, 3) = 1; // v_x
    C_(4, 4) = 1; // v_y

    Q_ = Eigen::MatrixXd::Identity(n_, n_) * 0.5;
    R_ = Eigen::MatrixXd::Identity(m_, m_) * 0.1;

    mu_ = Eigen::VectorXd::Zero(n_);
    Sigma_ = Eigen::MatrixXd::Identity(n_, n_);
}

void KalmanFilter::initialize(const Eigen::VectorXd &mu0, const Eigen::MatrixXd &Sigma0)
{
    mu_ = mu0;
    Sigma_ = Sigma0;
    initialized_ = true;
}

void KalmanFilter::setProcessNoise(const Eigen::MatrixXd &Q) { Q_ = Q; }
void KalmanFilter::setMeasurementNoise(const Eigen::MatrixXd &R) { R_ = R; }
void KalmanFilter::setControlMatrix(const Eigen::MatrixXd &B) { B_ = B; }

void KalmanFilter::predict(const Eigen::VectorXd &u)
{
    if (!initialized_) return;

    double theta = mu_(2);
    B_.setZero();
    B_(3, 0) = std::cos(theta);
    B_(4, 0) = std::sin(theta);
    B_(5, 1) = 1.0;

    //if (u.size() == 2)
    mu_ = A_ * mu_ + B_ * u;
    //mu_ = B_ * u;
    //else
    //    mu_ = A_ * mu_;

    std::cout << "state of kf:\n" << mu_ << std::endl;
    std::cout << "A:\n" << A_ << std::endl;
    std::cout << "A*mu:\n" << A_*mu_ << std::endl;
    std::cout << "B:\n" << B_ << std::endl;
    std::cout << "B*u:\n" << B_*u << std::endl;
    Sigma_ = A_ * Sigma_ * A_.transpose() + Q_;
    //std::cout << "Sigma_ after predict:\n" << Sigma_ << std::endl; // DEBUG
}

// Hilfsfunktion zur Umwandlung Roboter->Welt
Eigen::Vector2d KalmanFilter::velocityRobotToWorld(double v_robot, double theta) const {
    Eigen::Vector2d v_world;
    v_world(0) = v_robot * std::cos(theta); // v_x_world
    v_world(1) = v_robot * std::sin(theta); // v_y_world
    return v_world;
}

void KalmanFilter::updateMeasurement(const Eigen::VectorXd &z_raw, bool velocity_is_robot_frame)
{
    if (!initialized_) return;

    Eigen::VectorXd z = z_raw;

    if (velocity_is_robot_frame && z.size() >= 5) {
        // Geschwindigkeit aus Roboter-KS in Welt-KS umrechnen!
        double theta = mu_(2); // aktuelle Schätzung
        Eigen::Vector2d v_world = velocityRobotToWorld(z_raw(3), theta);
        z(3) = v_world(0); // v_x_world
        z(4) = v_world(1); // v_y_world
    }
    // Standard-Kalman-Update
    Eigen::VectorXd y = z - C_ * mu_;
    Eigen::MatrixXd S = C_ * Sigma_ * C_.transpose() + R_;
    Eigen::MatrixXd K = Sigma_ * C_.transpose() * S.inverse();

    mu_ = mu_ + K * y;
    Sigma_ = (Eigen::MatrixXd::Identity(n_, n_) - K * C_) * Sigma_;
}

Eigen::VectorXd KalmanFilter::getState() const { return mu_; }
Eigen::MatrixXd KalmanFilter::getCovariance() const { return Sigma_; }
