#include "particle_filter.h"
#include <cmath>       // For sqrt, exp
#include <algorithm>   // For std::fill
#include <ros/ros.h>   // For ROS_WARN

// SPDX-License-Identifier: MIT
// particle_filter.cpp
// ------------------------------------------------------------------------
// Particle filter implementation for a robot with state vector:
// [x, y, theta, vx, vy, omega]

// Constructor: Initializes particle filter with time step and number of particles
ParticleFilter::ParticleFilter(double dt, int num_particles)
    : dt_(dt), num_particles_(num_particles),
      estimated_x_(Eigen::VectorXd::Zero(6)),
      estimated_P_(Eigen::MatrixXd::Zero(6, 6))
{
    std::random_device rd;
    gen_.seed(rd()); // Seed the random generator
    reset();         // Initialize particles and noise models
}

// Set the timestep
void ParticleFilter::setDt(double dt) { dt_ = dt; }

// Reset the particle filter to initial conditions
void ParticleFilter::reset()
{
    particles_.clear();
    weights_.clear();

    // Initialize particles around an assumed start pose with Gaussian noise
    std::normal_distribution<double> pos_dist_x(0.5, 0.1);
    std::normal_distribution<double> pos_dist_y(0.5, 0.1);
    std::normal_distribution<double> orient_dist(0.0, 0.1);
    std::normal_distribution<double> vel_dist(0.0, 0.05);
    std::normal_distribution<double> ang_vel_dist(0.0, 0.05);

    for (int i = 0; i < num_particles_; ++i)
    {
        Eigen::VectorXd p_state(6);
        p_state(0) = pos_dist_x(gen_);    // x
        p_state(1) = pos_dist_y(gen_);    // y
        p_state(2) = orient_dist(gen_);   // theta
        p_state(3) = vel_dist(gen_);      // vx
        p_state(4) = vel_dist(gen_);      // vy
        p_state(5) = ang_vel_dist(gen_);  // omega
        particles_.push_back(p_state);
        weights_.push_back(1.0 / num_particles_); // Uniform weights
    }

    // Process noise covariance matrix Q_pf_ (used in prediction)
    Q_pf_ = Eigen::MatrixXd::Identity(6, 6);
    Q_pf_(0, 0) = 0.005;  // x noise
    Q_pf_(1, 1) = 0.005;  // y noise
    Q_pf_(2, 2) = 0.005;  // theta noise
    Q_pf_(3, 3) = 0.005;  // vx noise
    Q_pf_(4, 4) = 0.005;  // vy noise
    Q_pf_(5, 5) = 0.005;  // omega noise

    // Measurement noise covariance matrix R_pf_ (used in update)
    // Measurement model: only yaw (theta) and omega
    R_pf_ = Eigen::MatrixXd::Identity(2, 2);
    R_pf_(0, 0) = 0.01;   // yaw noise
    R_pf_(1, 1) = 0.02;   // omega noise
}

// Prediction step: apply motion model and add process noise to each particle
void ParticleFilter::predict(const Eigen::Vector2d &u_odom)
{
    // Sample noise for each state variable
    std::normal_distribution<double> x_noise_dist(0.0, std::sqrt(Q_pf_(0, 0)));
    std::normal_distribution<double> y_noise_dist(0.0, std::sqrt(Q_pf_(1, 1)));
    std::normal_distribution<double> theta_noise_dist(0.0, std::sqrt(Q_pf_(2, 2)));
    std::normal_distribution<double> vx_noise_dist(0.0, std::sqrt(Q_pf_(3, 3)));
    std::normal_distribution<double> vy_noise_dist(0.0, std::sqrt(Q_pf_(4, 4)));
    std::normal_distribution<double> omega_noise_dist(0.0, std::sqrt(Q_pf_(5, 5)));

    for (int i = 0; i < num_particles_; ++i)
    {
        Eigen::VectorXd &p_state = particles_[i];
        double v = u_odom(0);    // linear speed
        double w = u_odom(1);    // angular speed
        double theta = p_state(2);

        // Apply motion model + noise
        p_state(0) += v * dt_ * std::cos(theta) + x_noise_dist(gen_);
        p_state(1) += v * dt_ * std::sin(theta) + y_noise_dist(gen_);
        p_state(2) += w * dt_ + theta_noise_dist(gen_);
        p_state(2) = angles::normalize_angle(p_state(2));

        // Update velocities (optional, for richer state)
        p_state(3) = v * std::cos(p_state(2)) + vx_noise_dist(gen_);
        p_state(4) = v * std::sin(p_state(2)) + vy_noise_dist(gen_);
        p_state(5) = w + omega_noise_dist(gen_);
    }
}

// Update step: reweight particles using measurement likelihood
void ParticleFilter::update(const Eigen::Vector2d &z)
{
    Eigen::Matrix2d R_inv = R_pf_.inverse();
    double sum_weights = 0.0;

    for (int i = 0; i < num_particles_; ++i)
    {
        Eigen::VectorXd &p_state = particles_[i];

        // Measurement prediction from particle state
        Eigen::Vector2d h_x;
        h_x(0) = p_state(2); // theta
        h_x(1) = p_state(5); // omega

        // Innovation (measurement - prediction)
        Eigen::Vector2d innovation = z - h_x;
        innovation(0) = angles::normalize_angle(innovation(0)); // Normalize angle

        // Compute importance weight using Gaussian likelihood
        double exponent = -0.5 * innovation.transpose() * R_inv * innovation;
        weights_[i] *= std::exp(exponent);
        sum_weights += weights_[i];
    }

    // Normalize weights and handle degenerate cases
    if (sum_weights == 0.0)
    {
        ROS_WARN("All particle weights are zero. Resetting to uniform weights.");
        for (int i = 0; i < num_particles_; ++i)
            weights_[i] = 1.0 / num_particles_;
        sum_weights = 1.0;
    }

    // Normalize weights
    for (int i = 0; i < num_particles_; ++i)
    {
        weights_[i] /= sum_weights;
    }

    resample(); // Perform resampling to avoid degeneracy
}

// Systematic resampling: replaces low-weight particles with high-weight ones
void ParticleFilter::resample()
{
    std::vector<Eigen::VectorXd> new_particles;
    std::vector<double> new_weights;
    new_particles.reserve(num_particles_);
    new_weights.reserve(num_particles_);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(gen_) / num_particles_;
    double c = weights_[0];
    int i = 0;

    for (int j = 0; j < num_particles_; ++j)
    {
        double u = r + (double)j / num_particles_;
        while (u > c && i < num_particles_ - 1)
        {
            i++;
            c += weights_[i];
        }
        new_particles.push_back(particles_[i]);
        new_weights.push_back(1.0 / num_particles_);
    }

    particles_ = new_particles;
    weights_ = new_weights;
}

// Compute weighted average state across all particles
const Eigen::VectorXd &ParticleFilter::state()
{
    estimated_x_ = Eigen::VectorXd::Zero(6);
    for (int i = 0; i < num_particles_; ++i)
    {
        estimated_x_ += particles_[i] * weights_[i];
    }
    estimated_x_(2) = angles::normalize_angle(estimated_x_(2)); // Normalize orientation
    return estimated_x_;
}

// Compute covariance of the particle distribution around mean state
const Eigen::MatrixXd &ParticleFilter::cov()
{
    estimated_P_ = Eigen::MatrixXd::Zero(6, 6);
    Eigen::VectorXd mean_x = state();

    for (int i = 0; i < num_particles_; ++i)
    {
        Eigen::VectorXd diff = particles_[i] - mean_x;
        diff(2) = angles::normalize_angle(diff(2)); // Normalize angular difference
        estimated_P_ += weights_[i] * diff * diff.transpose();
    }
    return estimated_P_;
}
