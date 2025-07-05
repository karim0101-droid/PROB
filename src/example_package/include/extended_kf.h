#ifndef EXTENDED_KF_H
#define EXTENDED_KF_H

#include <eigen3/Eigen/Dense>
#include <angles/angles.h> // Für angles::normalize_angle

class ExtendedKF {
public:
  explicit ExtendedKF(double dt = 0.01);

  void setDt(double dt);
  void reset();
  void predict(const Eigen::Vector2d &u);
  void update(const Eigen::Vector2d &z); // Messvektor ist 2-dimensional (Yaw, Omega)

  const Eigen::VectorXd &state() const;
  const Eigen::MatrixXd &cov()   const;

private:
  double       dt_;
  Eigen::VectorXd x_;
  Eigen::MatrixXd P_, H_, Q_, R_;
};

#endif // EXTENDED_KF_H