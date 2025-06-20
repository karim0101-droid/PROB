#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>
#include <random>

struct Particle {
    Eigen::VectorXd x;
    double weight;
};

class ParticleFilter
{
public:
    ParticleFilter(int num_particles, double delta_t);

    void initialize(const Eigen::VectorXd &mu0, const Eigen::MatrixXd &Sigma0);
    void predict(const Eigen::VectorXd &u);
    void updateMeasurement(const Eigen::VectorXd &z);

    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;

    const std::vector<Particle>& getParticles() const { return particles_; }

    void setProcessNoise(const Eigen::MatrixXd& Q) { Q_ = Q; }
    void setMeasurementNoise(const Eigen::MatrixXd& R) { R_ = R; }

private:
    int num_particles_;
    double delta_t_;
    int state_dim_;

    std::vector<Particle> particles_;
    std::default_random_engine gen_;

    Eigen::MatrixXd Q_; // Prozessrauschen
    Eigen::MatrixXd R_; // Messrauschen

    void resample();
};
