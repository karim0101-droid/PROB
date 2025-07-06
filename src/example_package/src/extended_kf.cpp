#include "extended_kf.h"

// SPDX-License-Identifier: MIT
// extended_kf.cpp
// ------------------------------------------------------------------------
// Extended Kalman Filter implementation for a mobile robot
// State vector: [x, y, theta, vx, vy, omega]
// where:
//   - x, y: position
//   - theta: yaw angle (heading)
//   - vx, vy: linear velocity components
//   - omega: angular velocity

// Constructor: Initializes the filter with a given time step
ExtendedKF::ExtendedKF(double dt)
{
    setDt(dt);  // Store time step
    reset();    // Initialize filter state and covariance
}

// Set the time step used in prediction
void ExtendedKF::setDt(double dt) { dt_ = dt; }

// Resets the filter to its initial state and parameters
void ExtendedKF::reset()
{
    // Initialize state vector
    x_ = Eigen::VectorXd::Zero(6);
    x_(0) = 0.5; // Initial x position
    x_(1) = 0.5; // Initial y position
    x_(2) = 0.0; // Initial orientation
    // Other states (vx, vy, omega) are zero

    // Initial covariance matrix: low uncertainty
    P_ = Eigen::MatrixXd::Identity(6, 6) * 1e-3;

    // Measurement matrix H (2x6): maps state to measurable quantities
    // We only measure yaw (theta) and omega (angular velocity)
    H_ = Eigen::MatrixXd::Zero(2, 6);
    H_(0, 2) = 1.0; // Measure yaw
    H_(1, 5) = 1.0; // Measure omega

    // Process noise covariance Q: models uncertainty in motion
    Q_ = Eigen::MatrixXd::Identity(6, 6) * 1e-4;

    // Measurement noise covariance R: sensor measurement uncertainty
    R_ = Eigen::MatrixXd::Identity(2, 2) * 5e-3;
}

// Prediction step using nonlinear motion model
void ExtendedKF::predict(const Eigen::Vector2d &u)
{
    double v = u(0);      // Linear velocity input
    double w = u(1);      // Angular velocity input
    double theta = x_(2); // Current yaw

    // Predict next state using nonlinear motion model
    Eigen::VectorXd x_pred = x_;
    x_pred(0) += v * dt_ * std::cos(theta); // x += v * dt * cos(theta)
    x_pred(1) += v * dt_ * std::sin(theta); // y += v * dt * sin(theta)
    x_pred(2) += w * dt_;                   // theta += w * dt
    x_pred(2) = angles::normalize_angle(x_pred(2)); // Normalize angle

    // Update velocity components (vx, vy, omega)
    x_pred(3) = v * std::cos(x_pred(2)); // vx = v * cos(theta)
    x_pred(4) = v * std::sin(x_pred(2)); // vy = v * sin(theta)
    x_pred(5) = w;                       // omega = w

    // Save predicted state
    x_ = x_pred;

    // Linearize the motion model around the current state using Jacobian G
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(6, 6);
    G(0, 2) = -v * dt_ * std::sin(theta);         // d(x)/d(theta)
    G(1, 2) =  v * dt_ * std::cos(theta);         // d(y)/d(theta)
    G(3, 2) = -v * std::sin(x_pred(2));           // d(vx)/d(theta)
    G(4, 2) =  v * std::cos(x_pred(2));           // d(vy)/d(theta)

    // Propagate covariance
    P_ = G * P_ * G.transpose() + Q_;
}

// Update step: correct state using measurements
void ExtendedKF::update(const Eigen::Vector2d &z)
{
    // Innovation vector: difference between measurement and prediction
    Eigen::Vector2d y = z - H_ * x_;
    y(0) = angles::normalize_angle(y(0)); // Normalize yaw difference only

    // Innovation covariance
    Eigen::Matrix2d S = H_ * P_ * H_.transpose() + R_;

    // Kalman gain: determines how much to trust measurement
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

    // Correct the state estimate
    x_ += K * y;
    x_(2) = angles::normalize_angle(x_(2)); // Normalize yaw in state

    // Update covariance
    P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * P_;
}

// Getter for the current state estimate
const Eigen::VectorXd &ExtendedKF::state() const { return x_; }

// Getter for the current covariance matrix
const Eigen::MatrixXd &ExtendedKF::cov() const { return P_; }
