#include "linear_kf.h"

// SPDX-License-Identifier: MIT
// linear_kf.cpp
// ------------------------------------------------------------------------
// A simple linear Kalman Filter implementation for a 6D state vector
// The state vector: [x, y, theta, vx, vy, omega]
// where:
//   - x, y: position
//   - theta: orientation (yaw)
//   - vx, vy: linear velocities
//   - omega: angular velocity

// Constructor: Initializes the filter with a given time step
LinearKF::LinearKF(double dt)
{
    setDt(dt); // Setup system matrices with time step
    reset();   // Initialize state and covariance
}

// Set the time step and update the system matrices accordingly
void LinearKF::setDt(double dt)
{
    dt_ = dt;

    // State transition matrix A: models how the state evolves
    A_ = Eigen::MatrixXd::Zero(6, 6);
    A_(0, 0) = A_(1, 1) = A_(2, 2) = 1.0; // Position and orientation stay the same if velocity is zero
    A_(0, 3) = dt_; // x += vx * dt
    A_(1, 4) = dt_; // y += vy * dt
    A_(2, 5) = dt_; // theta += omega * dt

    // Control matrix B: models how control input affects the state
    // u = [v, w] where:
    //   v: linear speed (forward)
    //   w: angular speed (yaw rate)
    B_ = Eigen::MatrixXd::Zero(6, 2);
    B_(3, 0) = 1.0; // vx affected by linear speed v
    B_(5, 1) = 1.0; // omega affected by angular speed w
}

// Reset filter to initial state
void LinearKF::reset()
{
    // Initial state vector [x, y, theta, vx, vy, omega]
    x_ = Eigen::VectorXd::Zero(6);
    x_(0) = 0.5; // start x
    x_(1) = 0.5; // start y
    x_(2) = 0.0; // start orientation
    // rest already zero

    // Initial covariance matrix P: uncertainty in the state
    P_ = Eigen::MatrixXd::Identity(6, 6) * 1e-3;

    // Measurement matrix H: maps state space to measurement space
    H_ = Eigen::MatrixXd::Identity(6, 6); // Full-state observable

    // Process noise covariance Q: uncertainty in model dynamics
    Q_ = Eigen::MatrixXd::Identity(6, 6) * 2e-3;
    Q_(1, 1) = 1e0; // higher uncertainty in y
    Q_(4, 4) = 1e0; // higher uncertainty in vy

    // Measurement noise covariance R: uncertainty in sensor data
    R_ = Eigen::MatrixXd::Identity(6, 6) * 1e-3;
    R_(1, 1) = 1e-5; // very confident in y measurement
    R_(4, 4) = 1e-5; // very confident in vy measurement
}

// Prediction step: uses motion model to estimate new state
void LinearKF::predict(const Eigen::Vector2d &u)
{
    double v = u(0);         // linear velocity command
    double w = u(1);         // angular velocity command
    double theta = x_(2);    // current orientation

    // Predict state using the linear model
    x_ = A_ * x_; // Basic kinematic update

    // Inject control input into velocity terms directly (simplified dynamics)
    x_(3) = v;     // forward velocity
    x_(4) = 0.0;   // no sideways velocity
    x_(5) = w;     // angular velocity

    // Normalize angle to keep theta within [-pi, pi]
    x_(2) = angles::normalize_angle(x_(2));

    // Update covariance matrix to account for process noise
    P_ = A_ * P_ * A_.transpose() + Q_;
}

// Update step: corrects prediction using new measurement z
void LinearKF::update(const Eigen::VectorXd &z)
{
    // Innovation (residual) between measurement and prediction
    Eigen::VectorXd y = z - H_ * x_;
    y(2) = angles::normalize_angle(y(2)); // normalize angular difference

    // Innovation covariance
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;

    // Kalman Gain: how much to trust measurement vs prediction
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

    // Update state estimate
    x_ += K * y;
    x_(2) = angles::normalize_angle(x_(2)); // normalize angle again

    // Update estimate covariance
    P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * P_;
}

// Getter: return current state vector
const Eigen::VectorXd &LinearKF::state() const { return x_; }

// Getter: return current covariance matrix
const Eigen::MatrixXd &LinearKF::cov() const { return P_; }
