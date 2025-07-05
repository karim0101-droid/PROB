#include "linear_kf.h"

// SPDX-License-Identifier: MIT
// linear_kf.cpp
// ------------------------------------------------------------------------

LinearKF::LinearKF(double dt)
{
    setDt(dt);
    reset();
}

void LinearKF::setDt(double dt)
{
    dt_ = dt;

    A_ = Eigen::MatrixXd::Zero(6, 6);
    A_(0, 0) = A_(1, 1) = A_(2, 2) = 1.0;
    A_(0, 3) = dt_;
    A_(1, 4) = dt_;
    A_(2, 5) = dt_;

    B_ = Eigen::MatrixXd::Zero(6, 2);
    B_(3, 0) = 1.0;
    B_(5, 1) = 1.0;
}

void LinearKF::reset()
{
    x_ = Eigen::VectorXd::Zero(6);
    x_(0) = 0.5;
    x_(1) = 0.5;
    x_(2) = 0.0;
    x_(3) = 0.0;
    x_(4) = 0.0;
    x_(5) = 0.0;

    P_ = Eigen::MatrixXd::Identity(6, 6) * 1e-3;

    H_ = Eigen::MatrixXd::Identity(6, 6);

    Q_ = Eigen::MatrixXd::Identity(6, 6) * 2e-3;
    Q_(1, 1) = 1e0;
    Q_(4, 4) = 1e0;

    R_ = Eigen::MatrixXd::Identity(6, 6) * 1e-3;
    R_(1, 1) = 1e-5;
    R_(4, 4) = 1e-5;
}

void LinearKF::predict(const Eigen::Vector2d &u)
{
    double v = u(0);
    double w = u(1);
    double theta = x_(2); // theta ist der Gierwinkel des Roboters

    x_ = A_ * x_;

    x_(3) = v * 1.0; // vx = v (da Roboter nur vorwärts fährt)
    x_(4) = v * 0.0; // vy = 0
    x_(5) = w;       // omega = w

    x_(2) = angles::normalize_angle(x_(2)); // Winkel normalisieren

    P_ = A_ * P_ * A_.transpose() + Q_;
}

void LinearKF::update(const Eigen::VectorXd &z)
{
    Eigen::VectorXd y = z - H_ * x_;
    y(2) = angles::normalize_angle(y(2)); // Nur den Winkel im Innovationsvektor normalisieren

    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ += K * y;
    x_(2) = angles::normalize_angle(x_(2)); // Winkel im Zustand normalisieren
    P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * P_;
}

const Eigen::VectorXd &LinearKF::state() const { return x_; }
const Eigen::MatrixXd &LinearKF::cov() const { return P_; }