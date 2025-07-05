#include "extended_kf.h"

// SPDX-License-Identifier: MIT
// extended_kf.cpp
// ------------------------------------------------------------------------

ExtendedKF::ExtendedKF(double dt)
{
    setDt(dt);
    reset();
}

void ExtendedKF::setDt(double dt) { dt_ = dt; }

void ExtendedKF::reset()
{
    x_ = Eigen::VectorXd::Zero(6);
    x_(0) = 0.5;
    x_(1) = 0.5;
    x_(2) = 0.0;
    x_(3) = 0.0;
    x_(4) = 0.0;
    x_(5) = 0.0;

    P_ = Eigen::MatrixXd::Identity(6, 6) * 1e-3;

    H_ = Eigen::MatrixXd::Zero(2, 6);
    H_(0, 2) = 1.0; // Misst Yaw
    H_(1, 5) = 1.0; // Misst Omega

    Q_ = Eigen::MatrixXd::Identity(6, 6) * 1e-4;
    R_ = Eigen::MatrixXd::Identity(2, 2) * 5e-3;
}

void ExtendedKF::predict(const Eigen::Vector2d &u)
{
    double v = u(0);
    double w = u(1);
    double theta = x_(2);

    Eigen::VectorXd x_pred = x_;
    x_pred(0) += v * dt_ * std::cos(theta);
    x_pred(1) += v * dt_ * std::sin(theta);
    x_pred(2) += w * dt_;
    x_pred(2) = angles::normalize_angle(x_pred(2));
    x_pred(3) = v * std::cos(x_pred(2));
    x_pred(4) = v * std::sin(x_pred(2));
    x_pred(5) = w;
    x_ = x_pred;

    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(6, 6);
    G(0, 2) = -v * dt_ * std::sin(theta);
    G(1, 2) = v * dt_ * std::cos(theta);
    G(3, 2) = -v * std::sin(x_pred(2));
    G(4, 2) = v * std::cos(x_pred(2));

    P_ = G * P_ * G.transpose() + Q_;
}

void ExtendedKF::update(const Eigen::Vector2d &z)
{
    Eigen::Vector2d y = z - H_ * x_;
    y(0) = angles::normalize_angle(y(0)); // Nur den Yaw im Innovationsvektor normalisieren

    Eigen::Matrix2d S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ += K * y;
    x_(2) = angles::normalize_angle(x_(2)); // Den Yaw im Zustand normalisieren
    P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * P_;
}

const Eigen::VectorXd &ExtendedKF::state() const { return x_; }
const Eigen::MatrixXd &ExtendedKF::cov() const { return P_; }