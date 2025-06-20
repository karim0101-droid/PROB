#pragma once
#include <eigen3/Eigen/Dense>

class KalmanFilter
{
public:
    KalmanFilter(double delta_t = 0.05);

    void initialize(const Eigen::VectorXd &mu0, const Eigen::MatrixXd &Sigma0);

    // Für externe Anpassung (optional)
    void setProcessNoise(const Eigen::MatrixXd &Q);
    void setMeasurementNoise(const Eigen::MatrixXd &R);

    // Optional für spezielle Fälle
    void setControlMatrix(const Eigen::MatrixXd &B);

    // Vorhersage
    void predict(const Eigen::VectorXd &u = Eigen::VectorXd());

    // Korrektur (Messung!)
    void updateMeasurement(const Eigen::VectorXd &z_raw, bool velocity_is_robot_frame = true);

    // Getter
    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;

    double getDt() const { return delta_t_; }

private:
    int n_; // Zustandsdimension
    int m_; // Messdimension
    double delta_t_;

    Eigen::VectorXd mu_;      // Schätzwert (Zustand)
    Eigen::MatrixXd Sigma_;   // Kovarianz

    Eigen::MatrixXd A_;       // Systemmatrix
    Eigen::MatrixXd B_;       // Steuermatrix
    Eigen::MatrixXd C_;       // Messmatrix
    Eigen::MatrixXd Q_;       // Prozessrauschen
    Eigen::MatrixXd R_;       // Messrauschen

    bool initialized_ = false;

    // Hilfsfunktion: Geschwindigkeit transformieren
    Eigen::Vector2d velocityRobotToWorld(double v_robot, double theta) const;
};
