#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <eigen3/Eigen/Dense>
#include <angles/angles.h> // Für angles::normalize_angle
#include <vector>
#include <random> // Für Zufallszahlen-Generierung

class ParticleFilter
{
public:
    explicit ParticleFilter(double dt = 0.01, int num_particles = 1000);

    void setDt(double dt);
    void reset();
    void predict(const Eigen::Vector2d &u_odom);
    void update(const Eigen::Vector2d &z); // Messvektor ist 2-dimensional (Yaw, Omega)

    const Eigen::VectorXd &state(); // Nicht const, da estimated_x_ aktualisiert wird
    const Eigen::MatrixXd &cov();   // Nicht const, da estimated_P_ aktualisiert wird

private:
    void resample();

    double dt_;
    int num_particles_;
    std::vector<Eigen::VectorXd> particles_;
    std::vector<double> weights_;
    Eigen::VectorXd estimated_x_;
    Eigen::MatrixXd estimated_P_;
    Eigen::MatrixXd Q_pf_, R_pf_; // Prozess- und Messrauschkovarianzen für PF

    std::mt19937 gen_; // Zufallszahlengenerator
};

#endif // PARTICLE_FILTER_H