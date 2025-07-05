#include "KalmanFilter.h"
#include <cmath>

/* ---------- Konstruktor ---------------------------------------- */
KalmanFilter::KalmanFilter()
    : A_(Eigen::MatrixXd::Identity(N, N)),
      B_(Eigen::MatrixXd::Zero(N, 2)),
      H_(Eigen::Matrix<double, 6, N>::Identity()), // 6×6 Einheits-H
      I_(Eigen::MatrixXd::Identity(N, N)),
      R_(Eigen::Matrix<double, 6, 6>::Zero()),
      sigma_a2_(0.08 * 0.08),     // 0.08 m/s²
      sigma_alpha2_(0.04 * 0.04), // 0.04 rad/s²
      x_(Eigen::VectorXd::Zero(N)),
      P_(Eigen::MatrixXd::Identity(N, N) * 1e-3)
{
    /* --- R: große Varianzen für Pseudo-Messungen -------------- */
    R_.diagonal() << 0.04 * 0.04, 0.04 * 0.04, // x, y   (4 cm 1 σ)
        0.03 * 0.03,                           // θ
        0.03 * 0.03, 0.03 * 0.03,              // v_x, v_y
        0.012 * 0.012;                         // ω
}

/* ---------- σ_Q anpassen --------------------------------------- */
void KalmanFilter::setProcessNoiseStd(double sa, double salpha)
{
    sigma_a2_ = sa * sa;
    sigma_alpha2_ = salpha * salpha;
}

void KalmanFilter::setMeasurementNoise(const Eigen::Matrix<double, 6, 6> &R_in)
{
    R_ = R_in;
}

/* =====================  STEP  ================================== */
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
KalmanFilter::step(const Eigen::VectorXd &x_prev,
                   const Eigen::MatrixXd &P_prev,
                   const Eigen::VectorXd &u,
                   const Eigen::VectorXd &z,
                   double dt)
{
    dt = std::clamp(dt, 1e-4, 0.05);

    /* ---------- A & B ------------------------------------------ */
    A_.setIdentity();
    A_(0, 3) = A_(1, 4) = A_(2, 5) = dt;

    double th = x_prev(2);
    double c = std::cos(th), s = std::sin(th);

    B_.setZero();
    B_(3, 0) = c; // vx = v_body·cosθ_{k-1}
    B_(4, 0) = s; // vy = v_body·sinθ_{k-1}
    // B_(2,1)=dt;          // θ  += ω_cmd·dt
    B_(5, 1) = 1.0; // ω  =  ω_cmd

    /* ---------- Prädiktion ------------------------------------- */
    Eigen::VectorXd x_pred_1 = B_ * u;
    Eigen::VectorXd x_pred = A_ * x_prev; // **nur 1×**
    x_pred(3) = x_pred_1(3);
    x_pred(4) = x_pred_1(4);
    x_pred(5) = x_pred_1(5);
    Eigen::MatrixXd P_pred = A_ * P_prev * A_.transpose() + Qscaled(dt);

    /* ---------- Update (6-D) ----------------------------------- */
    /* --- Innovation / Update (6-D) -------------------------------- */
    /* --- Innovation / Update (6-D) -------------------------------- */
    Eigen::Matrix<double, 6, 1> y = z - x_pred; // H = I
    y(2) = std::atan2(std::sin(y(2)), std::cos(y(2)));

    Eigen::Matrix<double, 6, 6> S = P_pred + R_;
    Eigen::Matrix<double, N, 6> K = P_pred * S.inverse();

    x_ = x_pred + K * y;
    P_ = (I_ - K) * P_pred; // H = I

    x_(2) = std::atan2(std::sin(x_(2)), std::cos(x_(2)));

    return {x_, P_};
}

/* ---------- Q(dt) wie gehabt ---------------------------------- */
Eigen::MatrixXd KalmanFilter::Qscaled(double dt) const
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
