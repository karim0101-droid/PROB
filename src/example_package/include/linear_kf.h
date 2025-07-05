#ifndef LINEAR_KF_H
#define LINEAR_KF_H

#include <eigen3/Eigen/Dense>
#include <angles/angles.h> // Für angles::normalize_angle

class LinearKF
{
public:
  explicit LinearKF(double dt = 0.01);

  void setDt(double dt);
  void reset();
  void predict(const Eigen::Vector2d &u);
  void update(const Eigen::VectorXd &z);

  const Eigen::VectorXd &state() const;
  const Eigen::MatrixXd &cov() const;

private:
  double dt_;
  Eigen::VectorXd x_;
  Eigen::MatrixXd P_, A_, B_, H_, Q_, R_;
};

#endif // LINEAR_KF_H