#include "KalmanFilter.h"
#include <cmath>
#include <iostream>

KalmanFilter::KalmanFilter()
{
    A = Eigen::MatrixXd::Identity(N, N);
    B = Eigen::MatrixXd::Zero(N, 2);
    H = Eigen::MatrixXd::Identity(N, N);
    H(0, 0) = 0.0;
    H(1, 1) = 0.0;
    Q = Eigen::MatrixXd::Identity(N, N) * 0.001;
    R = Eigen::MatrixXd::Identity(N, N) * 0.006;
    I = Eigen::MatrixXd::Identity(N, N);

    mu = Eigen::VectorXd::Zero(N);
    P = Eigen::MatrixXd::Identity(N, N);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::algorithm(
    const Eigen::VectorXd &mu_1,
    const Eigen::MatrixXd &P_1,
    const Eigen::VectorXd &u,
    const Eigen::VectorXd &z,
    double dt)
{
    // Prediction Schritt q      
    //dt = dt*1000.0;
    A.setIdentity();
    A(0, 3) = dt;
    A(1, 4) = dt;
    A(2, 5) = dt;

    B.setZero();
    B(3, 0) = std::cos(z(2));  // v_x = v * cos(theta)
    B(4, 0) = std::sin(z(2));  // v_y = v * sin(theta)
    B(5, 1) = 1.0;

    Eigen::VectorXd mu_11 = A * mu_1;
    Eigen::VectorXd vel_11 = B * u;;
    Eigen::VectorXd mu_pred = A * mu_1;
    Eigen::MatrixXd P_pred  = A * P_1 * A.transpose() + Q;
    //Eigen::MatrixXd P_pred = Eigen::MatrixXd::Identity(N, N);

    //mu_pred.segment<3>(3) = vel_11;
    mu_pred(3) = vel_11(3);
    mu_pred(4) = vel_11(4);
    mu_pred(5) = vel_11(5);


    //std::cout <<"Predicted States: " << vel_11(3) << std::endl;

    // Update Schritt
    Eigen::VectorXd y = z - H * mu_pred;
    Eigen::MatrixXd S = H * P_pred * H.transpose() + R;
    Eigen::MatrixXd K = P_pred * H.transpose() * S.inverse();

    mu = mu_pred + K * y;
    P  = (I - K * H) * P_pred;
    //P = P_pred;
    //std::cout <<"Updated States: " << mu << std::endl;
    //std::cout <<"time sequence dt: " << dt << std::endl;

    return std::make_pair(mu, P);
}

Eigen::VectorXd KalmanFilter::getState() const { return mu; }
Eigen::MatrixXd KalmanFilter::getCovariance() const { return P; }

void KalmanFilter::setProcessNoise(const Eigen::MatrixXd& Q_in) { Q = Q_in; }
void KalmanFilter::setMeasurementNoise(const Eigen::MatrixXd& R_in) { R = R_in; }
