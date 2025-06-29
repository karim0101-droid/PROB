#pragma once
#include <eigen3/Eigen/Dense>
#include <utility>

class KalmanFilter
{
public:
    KalmanFilter();

    // Gibt Zustand und Kovarianz als std::pair zurück
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> algorithm(
        const Eigen::VectorXd &mu_1,
        const Eigen::MatrixXd &P_1,
        const Eigen::VectorXd &u,
        const Eigen::VectorXd &z,
        double dt
    );

    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;

    void setProcessNoise(const Eigen::MatrixXd& Q_in);
    void setMeasurementNoise(const Eigen::MatrixXd& R_in);

private:
    static constexpr int N = 6;

    Eigen::MatrixXd A; // Systemmatrix (6x6)
    Eigen::MatrixXd B; // Steuermatrix (6x2)
    Eigen::MatrixXd H; // Messmatrix (6x6)
    Eigen::MatrixXd Q; // Prozessrauschen (6x6)
    Eigen::MatrixXd R; // Messrauschen (6x6)
    Eigen::MatrixXd I; // Einheitsmatrix (6x6)

    Eigen::VectorXd mu; // Zustand (6x1)
    Eigen::MatrixXd P;  // Kovarianzmatrix (6x6)
};
