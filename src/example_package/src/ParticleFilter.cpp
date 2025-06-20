#include "ParticleFilter.h"
#include <cmath>
#include <numeric>
#include <random>

ParticleFilter::ParticleFilter(int num_particles, double delta_t)
    : num_particles_(num_particles), delta_t_(delta_t), gen_(std::random_device{}())
{
    state_dim_ = 6; // x, y, theta, v_x, v_y, omega
    Q_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_) * 0.5;
    R_ = Eigen::MatrixXd::Identity(5, 5) * 0.1;
}

void ParticleFilter::initialize(const Eigen::VectorXd &mu0, const Eigen::MatrixXd &Sigma0)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    particles_.clear();
    for(int i=0; i<num_particles_; ++i)
    {
        Particle p;
        p.x = mu0;
        for(int j=0; j<state_dim_; ++j)
            p.x(j) += dist(gen_) * std::sqrt(Sigma0(j, j));
        p.weight = 1.0 / num_particles_;
        particles_.push_back(p);
    }
}

void ParticleFilter::predict(const Eigen::VectorXd &u)
{
    std::normal_distribution<double> noise(0.0, 1.0);

    for(auto &p : particles_)
    {
        double theta = p.x(2);
        double v_x = u(0);
        double v_y = u(1);
        double omega = u.size() > 2 ? u(2) : 0.0;

        // 2D Kinematik
        p.x(0) += (std::cos(theta) * v_x - std::sin(theta) * v_y) * delta_t_;
        p.x(1) += (std::sin(theta) * v_x + std::cos(theta) * v_y) * delta_t_;
        p.x(2) += omega * delta_t_;

        p.x(3) = v_x;
        p.x(4) = v_y;
        p.x(5) = omega;

        // Prozessrauschen
        for(int j=0; j<state_dim_; ++j)
            p.x(j) += noise(gen_) * std::sqrt(Q_(j, j));
    }
}

void ParticleFilter::updateMeasurement(const Eigen::VectorXd &z)
{
    double norm_factor = 0.0;
    for(auto &p : particles_)
    {
        Eigen::VectorXd z_hat(5);
        z_hat << p.x(0), p.x(1), p.x(2), p.x(3), p.x(5);
        Eigen::VectorXd diff = z - z_hat;
        double w = std::exp(-0.5 * diff.transpose() * R_.inverse() * diff);
        p.weight = w;
        norm_factor += w;
    }
    // Normalisieren
    for(auto &p : particles_)
        p.weight /= (norm_factor + 1e-12);

    resample();
}

void ParticleFilter::resample()
{
    std::vector<Particle> new_particles;
    std::vector<double> cdf(particles_.size());
    cdf[0] = particles_[0].weight;
    for(size_t i=1; i<particles_.size(); ++i)
        cdf[i] = cdf[i-1] + particles_[i].weight;

    double step = 1.0 / num_particles_;
    double u = ((double)rand() / RAND_MAX) * step;
    size_t i = 0;
    for(int j=0; j<num_particles_; ++j)
    {
        while(u > cdf[i] && i < particles_.size()-1) i++;
        Particle p = particles_[i];
        p.weight = 1.0 / num_particles_;
        new_particles.push_back(p);
        u += step;
    }
    particles_ = new_particles;
}

Eigen::VectorXd ParticleFilter::getState() const
{
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(state_dim_);
    for(const auto &p : particles_)
        mean += p.weight * p.x;
    return mean;
}

Eigen::MatrixXd ParticleFilter::getCovariance() const
{
    Eigen::VectorXd mean = getState();
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
    for(const auto &p : particles_)
    {
        Eigen::VectorXd d = p.x - mean;
        cov += p.weight * d * d.transpose();
    }
    return cov;
}
