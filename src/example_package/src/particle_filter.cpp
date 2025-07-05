#include "particle_filter.h"
#include <cmath>     // Für std::sqrt, std::exp
#include <algorithm> // Für std::fill
#include <ros/ros.h> // Hinzugefügt für ROS_WARN

// SPDX-License-Identifier: MIT
// particle_filter.cpp
// ------------------------------------------------------------------------

ParticleFilter::ParticleFilter(double dt, int num_particles)
    : dt_(dt), num_particles_(num_particles),
      estimated_x_(Eigen::VectorXd::Zero(6)),
      estimated_P_(Eigen::MatrixXd::Zero(6, 6))
{
    std::random_device rd;
    gen_.seed(rd()); // Seed the random number generator
    reset();
}

void ParticleFilter::setDt(double dt) { dt_ = dt; }

void ParticleFilter::reset()
{
    particles_.clear();
    weights_.clear();
    // Initialisiere Partikel um eine Startpose herum
    // Annahme: Startpose ist (0.5, 0.5, 0.0) mit kleinen Unsicherheiten
    std::normal_distribution<double> pos_dist_x(0.5, 0.1); // Mean, StdDev
    std::normal_distribution<double> pos_dist_y(0.5, 0.1);
    std::normal_distribution<double> orient_dist(0.0, 0.1);
    std::normal_distribution<double> vel_dist(0.0, 0.05);
    std::normal_distribution<double> ang_vel_dist(0.0, 0.05);

    for (int i = 0; i < num_particles_; ++i)
    {
        Eigen::VectorXd p_state(6);
        p_state(0) = pos_dist_x(gen_);   // x
        p_state(1) = pos_dist_y(gen_);   // y
        p_state(2) = orient_dist(gen_);  // theta
        p_state(3) = vel_dist(gen_);     // vx
        p_state(4) = vel_dist(gen_);     // vy
        p_state(5) = ang_vel_dist(gen_); // omega
        particles_.push_back(p_state);
        weights_.push_back(1.0 / num_particles_);
    }

    // Beispiel-Rauschkovarianzen (müssen feinabgestimmt werden!)
    // Q_pf_ ist das Prozessrauschen, das beim Prädiktionsschritt hinzugefügt wird.
    Q_pf_ = Eigen::MatrixXd::Identity(6, 6);
    Q_pf_(0, 0) = 0.005; // Rauschen auf X
    Q_pf_(1, 1) = 0.005; // Rauschen auf Y
    Q_pf_(2, 2) = 0.005; // Rauschen auf Theta
    Q_pf_(3, 3) = 0.005; // Rauschen auf Vx
    Q_pf_(4, 4) = 0.005; // Rauschen auf Vy
    Q_pf_(5, 5) = 0.005; // Rauschen auf Omega

    // R_pf_ ist das Messrauschen für die einzelnen Sensoren
    // MESSVEKTOR PF: Nur Yaw (Index 0) und Omega (Index 1) -> 2 Dimensionen
    R_pf_ = Eigen::MatrixXd::Identity(2, 2);
    R_pf_(0, 0) = 0.01; // Rauschen Gierwinkel (aus IMU)
    R_pf_(1, 1) = 0.02; // Rauschen Gierrate (aus IMU)
}

void ParticleFilter::predict(const Eigen::Vector2d &u_odom)
{
    std::normal_distribution<double> x_noise_dist(0.0, std::sqrt(Q_pf_(0, 0)));
    std::normal_distribution<double> y_noise_dist(0.0, std::sqrt(Q_pf_(1, 1)));
    std::normal_distribution<double> theta_noise_dist(0.0, std::sqrt(Q_pf_(2, 2)));
    std::normal_distribution<double> vx_noise_dist(0.0, std::sqrt(Q_pf_(3, 3)));
    std::normal_distribution<double> vy_noise_dist(0.0, std::sqrt(Q_pf_(4, 4)));
    std::normal_distribution<double> omega_noise_dist(0.0, std::sqrt(Q_pf_(5, 5)));

    for (int i = 0; i < num_particles_; ++i)
    {
        Eigen::VectorXd &p_state = particles_[i];
        double v = u_odom(0);
        double w = u_odom(1);
        double theta = p_state(2);

        p_state(0) += v * dt_ * std::cos(theta) + x_noise_dist(gen_);
        p_state(1) += v * dt_ * std::sin(theta) + y_noise_dist(gen_);
        p_state(2) += w * dt_ + theta_noise_dist(gen_);
        p_state(2) = angles::normalize_angle(p_state(2));

        p_state(3) = v * std::cos(p_state(2)) + vx_noise_dist(gen_);
        p_state(4) = v * std::sin(p_state(2)) + vy_noise_dist(gen_);
        p_state(5) = w + omega_noise_dist(gen_);
    }
}

void ParticleFilter::update(const Eigen::Vector2d &z)
{ // Messvektor ist 2-dimensional
    Eigen::Matrix2d R_inv = R_pf_.inverse();
    double sum_weights = 0.0;

    for (int i = 0; i < num_particles_; ++i)
    {
        Eigen::VectorXd &p_state = particles_[i];

        Eigen::Vector2d h_x;
        h_x(0) = p_state(2); // Partikel-Yaw
        h_x(1) = p_state(5); // Partikel-Omega

        Eigen::Vector2d innovation = z - h_x;
        innovation(0) = angles::normalize_angle(innovation(0)); // Nur Yaw normalisieren

        double exponent = -0.5 * innovation.transpose() * R_inv * innovation;
        weights_[i] *= std::exp(exponent);
        sum_weights += weights_[i];
    }

    if (sum_weights == 0.0)
    {
        ROS_WARN("All particle weights are zero. Resetting to uniform weights.");
        for (int i = 0; i < num_particles_; ++i)
        {
            weights_[i] = 1.0 / num_particles_;
        }
        sum_weights = 1.0;
    }

    for (int i = 0; i < num_particles_; ++i)
    {
        weights_[i] /= sum_weights;
    }

    resample();
}

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

const Eigen::VectorXd &ParticleFilter::state()
{
    estimated_x_ = Eigen::VectorXd::Zero(6);
    for (int i = 0; i < num_particles_; ++i)
    {
        estimated_x_ += particles_[i] * weights_[i];
    }
    estimated_x_(2) = angles::normalize_angle(estimated_x_(2));
    return estimated_x_;
}

const Eigen::MatrixXd &ParticleFilter::cov()
{
    estimated_P_ = Eigen::MatrixXd::Zero(6, 6);
    Eigen::VectorXd mean_x = state();

    for (int i = 0; i < num_particles_; ++i)
    {
        Eigen::VectorXd diff = particles_[i] - mean_x;
        diff(2) = angles::normalize_angle(diff(2));
        estimated_P_ += weights_[i] * diff * diff.transpose();
    }
    return estimated_P_;
}