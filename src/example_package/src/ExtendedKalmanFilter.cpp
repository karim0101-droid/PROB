#include "ExtendedKalmanFilter.h"
#include <cmath>

ExtendedKalmanFilter::ExtendedKalmanFilter()
    : x_(Eigen::VectorXd::Zero(N)),
      P_(Eigen::MatrixXd::Identity(N, N) * 1e-3),
      R6_(Eigen::Matrix<double, 6, 6>::Identity() * 0.002),
      sigma_a2_(0.08 * 0.08),
      sigma_alpha2_(0.04 * 0.04),
      I_(Eigen::MatrixXd::Identity(N, N))
{
}

void ExtendedKalmanFilter::setProcessNoiseStd(double sa, double salpha)
{
    sigma_a2_ = sa * sa;
    sigma_alpha2_ = salpha * salpha;
}

/* =========================== STEP ================================== */
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
ExtendedKalmanFilter::step(const Eigen::VectorXd &x_prev,
                           const Eigen::MatrixXd &P_prev,
                           const Eigen::VectorXd &u,
                           const Eigen::VectorXd &z,
                           double dt)
{
    /* ---------- Prediction ---------------------------------------- */
    Eigen::VectorXd x_pred = g(x_prev, u, dt);
    Eigen::MatrixXd Fk = F(x_prev, u, dt);
    Eigen::MatrixXd P_pred = Fk * P_prev * Fk.transpose() + Qscaled(dt);

    /* ---------- Innovation (H = I) -------------------------------- */
    Eigen::Matrix<double, 6, 1> z_vec;
    z_vec << z(0), z(1), z(2), z(3), z(4), z(5);
    Eigen::Matrix<double, 6, 1> y = z_vec - x_pred;
    y(2) = std::atan2(std::sin(y(2)), std::cos(y(2))); // θ wrap

    /* ---------- Kalman Gain & Update ------------------------------ */
    Eigen::Matrix<double, 6, 6> S = P_pred + R6_; // H = I
    Eigen::Matrix<double, 6, 6> K = P_pred * S.inverse();

    x_ = x_pred + K * y;
    P_ = (I_ - K) * P_pred; // H = I

    x_(2) = std::atan2(std::sin(x_(2)), std::cos(x_(2)));
    return {x_, P_};
}

/* ---------- Bewegungsmodell g(x,u,dt) ----------------------------- */
Eigen::VectorXd ExtendedKalmanFilter::g(const Eigen::VectorXd &x,
                                        const Eigen::VectorXd &u,
                                        double dt) const
{
    double v = u(0), w = u(1);
    double th = x(2), s = std::sin(th), c = std::cos(th);

    Eigen::VectorXd xp = x;
    if (std::fabs(w) > 1e-6)
    {
        double R = v / w, wt = w * dt;
        xp(0) += R * (std::sin(th + wt) - s);
        xp(1) += R * (-std::cos(th + wt) + c);
    }
    else
    {
        xp(0) += v * dt * c;
        xp(1) += v * dt * s;
    }
    xp(2) += w * dt;
    xp(3) = v * c;
    xp(4) = v * s;
    xp(5) = w;
    return xp;
}

/* ---------- Jacobian F ------------------------------------------- */
Eigen::MatrixXd ExtendedKalmanFilter::F(const Eigen::VectorXd &x,
                                        const Eigen::VectorXd &u,
                                        double dt) const
{
    double v = u(0), w = u(1), th = x(2);
    double s = std::sin(th), c = std::cos(th);

    Eigen::MatrixXd Fk = Eigen::MatrixXd::Identity(N, N);
    if (std::fabs(w) > 1e-6)
    {
        double R = v / w, wt = w * dt;
        Fk(0, 2) = R * (std::cos(th + wt) - c);
        Fk(1, 2) = R * (std::sin(th + wt) - s);
    }
    else
    {
        Fk(0, 2) = -v * dt * s;
        Fk(1, 2) = v * dt * c;
    }
    Fk(0, 3) = dt;
    Fk(1, 4) = dt;
    Fk(2, 5) = dt;
    Fk(3, 2) = -v * s;
    Fk(4, 2) = v * c;
    return Fk;
}

/* ---------- Prozessrauschen Q(dt) -------------------------------- */
Eigen::MatrixXd ExtendedKalmanFilter::Qscaled(double dt) const
{
    double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N, N);

    Q(0, 0) = Q(1, 1) = 0.25 * dt4 * sigma_a2_;
    Q(0, 3) = Q(3, 0) = 0.5 * dt3 * sigma_a2_;
    Q(1, 4) = Q(4, 1) = 0.5 * dt3 * sigma_a2_;
    Q(3, 3) = Q(4, 4) = dt2 * sigma_a2_;

    Q(2, 2) = 0.25 * dt4 * sigma_alpha2_;
    Q(2, 5) = Q(5, 2) = 0.5 * dt3 * sigma_alpha2_;
    Q(5, 5) = dt2 * sigma_alpha2_;
    return Q;
}
