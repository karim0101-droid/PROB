// SPDX-License-Identifier: MIT
// filter_node.cpp – Linear Kalman Filter vs. Extended Kalman Filter (ROS 1)
// ------------------------------------------------------------------------

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h> // Hinzugefügt für das Veröffentlichen von Geschwindigkeiten
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <angles/angles.h>
#include <boost/bind/bind.hpp>
#include <algorithm>
#include <cmath>
#include <memory>
#include <sensor_msgs/JointState.h>
#include <map>

// ======================= 1.  LINEAR  KALMAN FILTER =======================
class LinearKF {
public:
  explicit LinearKF(double dt = 0.01) { setDt(dt); reset(); }

  void setDt(double dt) {
    dt_ = dt;

    A_ = Eigen::MatrixXd::Zero(6, 6);
    A_(0,0) = A_(1,1) = A_(2,2) = 1.0;
    A_(0,3) = dt_;
    A_(1,4) = dt_;
    A_(2,5) = dt_;

    B_ = Eigen::MatrixXd::Zero(6, 2);
    B_(3,0) = 1.0;
    B_(5,1) = 1.0;
  }

  void reset() {
    x_ = Eigen::VectorXd::Zero(6);
    x_(0) = 0.5;
    x_(1) = 0.5;
    x_(2) = 0.0;
    x_(3) = 0.0;
    x_(4) = 0.0;
    x_(5) = 0.0;

    P_ = Eigen::MatrixXd::Identity(6,6) * 1e-3;

    H_ = Eigen::MatrixXd::Identity(6,6);

    Q_ = Eigen::MatrixXd::Identity(6,6) * 2e-3;
    Q_(1,1) = 1e0;
    Q_(4,4) = 1e0;
    
    R_ = Eigen::MatrixXd::Identity(6,6) * 1e-3;
    R_(1,1) = 1e-5;
    R_(4,4) = 1e-5;
  }

  void predict(const Eigen::Vector2d &u) {
    double v = u(0);
    double w = u(1);
    double theta = x_(2);

    x_ = A_*x_;

    x_(3) = v * 1.0;
    x_(4) = v * 0.0;
    x_(5) = w;

    x_(2)  = angles::normalize_angle(x_(2));

    P_ = A_ * P_ * A_.transpose() + Q_;
  }

  void update(const Eigen::VectorXd &z) {
    Eigen::VectorXd y = z - H_ * x_;
    y(2) = angles::normalize_angle(y(2));

    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ += K * y;
    x_(2) = angles::normalize_angle(x_(2));
    P_  = (Eigen::MatrixXd::Identity(6,6) - K*H_) * P_;
  }

  const Eigen::VectorXd &state() const { return x_; }
  const Eigen::MatrixXd &cov()   const { return P_; }

private:
  double       dt_{0.01};
  Eigen::VectorXd x_{Eigen::VectorXd::Zero(6)};
  Eigen::MatrixXd P_, A_, B_, H_, Q_, R_;
};

// ======================= 2.  EXTENDED  KALMAN FILTER =====================
class ExtendedKF {
public:
  explicit ExtendedKF(double dt = 0.01) { setDt(dt); reset(); }

  void setDt(double dt) { dt_ = dt; }

  void reset() {
    x_ = Eigen::VectorXd::Zero(6);
    x_(0) = 0.5; x_(1) = 0.5;
    x_(2) = 0.0;
    x_(3) = 0.0;
    x_(4) = 0.0;
    x_(5) = 0.0;

    P_ = Eigen::MatrixXd::Identity(6,6) * 1e-3;

    H_ = Eigen::MatrixXd::Zero(2,6);
    H_(0,2) = 1.0;
    H_(1,5) = 1.0;

    Q_ = Eigen::MatrixXd::Identity(6,6) * 1e-4;
    R_ = Eigen::MatrixXd::Identity(2,2) * 5e-3;
  }

  void predict(const Eigen::Vector2d &u) {
    double v = u(0);
    double w = u(1);
    double theta = x_(2);

    Eigen::VectorXd x_pred = x_;
    x_pred(0) += v * dt_ * std::cos(theta);
    x_pred(1) += v * dt_ * std::sin(theta);
    x_pred(2) += w * dt_;
    x_pred(2)  = angles::normalize_angle(x_pred(2));
    x_pred(3)  = v * std::cos(x_pred(2));
    x_pred(4)  = v * std::sin(x_pred(2));
    x_pred(5)  = w;
    x_ = x_pred;

    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(6, 6);
    G(0,2) = -v * dt_ * std::sin(theta);
    G(1,2) =  v * dt_ * std::cos(theta);
    G(3,2) = -v * std::sin(x_pred(2));
    G(4,2) =  v * std::cos(x_pred(2));

    P_ = G * P_ * G.transpose() + Q_;
  }

  void update(const Eigen::Vector2d &z) {
    Eigen::Vector2d y = z - H_ * x_;
    y(0) = angles::normalize_angle(y(0));

    Eigen::Matrix2d S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ += K * y;
    x_(2) = angles::normalize_angle(x_(2));
    P_  = (Eigen::MatrixXd::Identity(6,6) - K*H_) * P_;
  }

  const Eigen::VectorXd &state() const { return x_; }
  const Eigen::MatrixXd &cov()   const { return P_; }

private:
  double       dt_{0.01};
  Eigen::VectorXd x_{Eigen::VectorXd::Zero(6)};
  Eigen::MatrixXd P_, H_, Q_, R_;
};

// =========================== 3. Helper ================================
static void toPoseMsg(const Eigen::VectorXd &x,
                      const Eigen::MatrixXd &P,
                      const ros::Time      &stamp,
                      const std::string    &frame,
                      geometry_msgs::PoseWithCovarianceStamped &msg) {
  msg.header.stamp    = stamp;
  msg.header.frame_id = frame;

  msg.pose.pose.position.x = x(0);
  msg.pose.pose.position.y = x(1);
  tf2::Quaternion q; q.setRPY(0,0,x(2));
  msg.pose.pose.orientation = tf2::toMsg(q);

  std::fill(std::begin(msg.pose.covariance),
            std::end(msg.pose.covariance), 0.0);
  msg.pose.covariance[0*6+0] = P(0,0);
  msg.pose.covariance[0*6+1] = P(0,1);
  msg.pose.covariance[1*6+0] = P(1,0);
  msg.pose.covariance[1*6+1] = P(1,1);
  msg.pose.covariance[5*6+5] = P(2,2);
  msg.pose.covariance[0*6+5] = P(0,2);
  msg.pose.covariance[5*6+0] = P(2,0);
  msg.pose.covariance[1*6+5] = P(1,2);
  msg.pose.covariance[5*6+1] = P(2,1);
}

// =========================== 4. ROS Node ===============================
class FilterNode {
public:
  explicit FilterNode(ros::NodeHandle &nh)
      : kf_(0.01), ekf_(0.01), wheel_radius_(0.0), wheel_base_(0.0),
        current_kinematic_x_(0.5), current_kinematic_y_(0.5), current_kinematic_yaw_(0.0),
        is_first_measurement_(true), last_joint_state_stamp_(ros::Time(0)) {

    nh.param("wheel_radius", wheel_radius_, 0.033);
    nh.param("wheel_base", wheel_base_, 0.160);

    odom_sub_.subscribe(nh,"/odom",10);
    imu_sub_ .subscribe(nh,"/imu", 10);
    joint_state_sub_.subscribe(nh, "/joint_states", 10);

    sync_ = std::make_shared<
              message_filters::TimeSynchronizer<
                nav_msgs::Odometry, sensor_msgs::Imu, sensor_msgs::JointState>>(
                  odom_sub_, imu_sub_, joint_state_sub_, 10);
    using namespace boost::placeholders;
    sync_->registerCallback(boost::bind(&FilterNode::sensorCb,this,_1,_2,_3));

    kf_pub_  = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>(
                 "/kf_prediction",10);
    ekf_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>(
                 "/ekf_prediction",10);
    // NEU: Publisher für geschätzte Geschwindigkeiten
    kf_vel_pub_ = nh.advertise<nav_msgs::Odometry>("/kf_velocity_prediction", 10);
    ekf_vel_pub_ = nh.advertise<nav_msgs::Odometry>("/ekf_velocity_prediction", 10);
  }

private:
  void sensorCb(const nav_msgs::Odometry::ConstPtr &odom,
                const sensor_msgs::Imu   ::ConstPtr &imu,
                const sensor_msgs::JointState::ConstPtr &joint_state) {

    double dt_ = 0.0;
    if(!last_stamp_.isZero()){
      double dt = (odom->header.stamp - last_stamp_).toSec();
      if(dt>1e-6){
          kf_.setDt(dt);
          ekf_.setDt(dt);
          dt_ = dt;
      }
    }
    last_stamp_ = odom->header.stamp;

    Eigen::Vector2d u_odom;
    u_odom << odom->twist.twist.linear.x,
              odom->twist.twist.angular.z;

    tf2::Quaternion q_imu; tf2::fromMsg(imu->orientation,q_imu);
    double roll_imu,pitch_imu,yaw_imu; tf2::Matrix3x3(q_imu).getRPY(roll_imu,pitch_imu,yaw_imu);
    yaw_imu = angles::normalize_angle(yaw_imu);
    double omega_imu = imu->angular_velocity.z;

    if (is_first_measurement_) {
        current_kinematic_yaw_ = yaw_imu;
        is_first_measurement_ = false;
    }

    double left_wheel_vel = 0.0;
    double right_wheel_vel = 0.0;
    bool velocities_calculated = false;

    int left_joint_idx = -1;
    int right_joint_idx = -1;
    for (size_t i = 0; i < joint_state->name.size(); ++i) {
        if (joint_state->name[i] == "wheel_left_joint") {
            left_joint_idx = i;
        } else if (joint_state->name[i] == "wheel_right_joint") {
            right_joint_idx = i;
        }
    }

    if (left_joint_idx != -1 && right_joint_idx != -1 && joint_state->position.size() > std::max(left_joint_idx, right_joint_idx)) {
        if (!last_joint_state_stamp_.isZero() && joint_state->header.stamp > last_joint_state_stamp_) {
            double dt_joint_state = (joint_state->header.stamp - last_joint_state_stamp_).toSec();
            if (dt_joint_state > 1e-6) {
                double current_left_pos = joint_state->position[left_joint_idx];
                double current_right_pos = joint_state->position[right_joint_idx];

                if (previous_joint_positions_.count("wheel_left_joint") && previous_joint_positions_.count("wheel_right_joint")) {
                    left_wheel_vel = (current_left_pos - previous_joint_positions_["wheel_left_joint"]) / dt_joint_state;
                    right_wheel_vel = (current_right_pos - previous_joint_positions_["wheel_right_joint"]) / dt_joint_state;
                    velocities_calculated = true;
                }
            }
        }
        previous_joint_positions_["wheel_left_joint"] = joint_state->position[left_joint_idx];
        previous_joint_positions_["wheel_right_joint"] = joint_state->position[right_joint_idx];
        last_joint_state_stamp_ = joint_state->header.stamp;
    } else {
        ROS_WARN_THROTTLE(1.0, "Wheel joint names or positions not found in joint_states. Cannot calculate velocities from positions.");
    }

    geometry_msgs::PoseWithCovarianceStamped kf_msg, ekf_msg;
    nav_msgs::Odometry kf_vel_msg, ekf_vel_msg; // NEU: Für Geschwindigkeits-Nachrichten

    if (!velocities_calculated) {
        ROS_WARN_THROTTLE(1.0, "Falling back to IMU and Odometry for velocity updates due to missing joint state velocity calculation.");
        Eigen::VectorXd z_kf_fallback(6);
        z_kf_fallback << kf_.state()(0), kf_.state()(1), yaw_imu, kf_.state()(3), kf_.state()(4), omega_imu;
        kf_.predict(u_odom); kf_.update(z_kf_fallback);

        Eigen::Vector2d z_ekf_fallback(yaw_imu, omega_imu);
        ekf_.predict(u_odom); ekf_.update(z_ekf_fallback);

        toPoseMsg(kf_.state(), kf_.cov(),  last_stamp_, "map", kf_msg);
        toPoseMsg(ekf_.state(),ekf_.cov(), last_stamp_, "map", ekf_msg);
        kf_pub_.publish(kf_msg);  ekf_pub_.publish(ekf_msg);
        // NEU: Geschwindigkeits-Fallback publizieren (aus Odometrie)
        kf_vel_msg.header.stamp = last_stamp_;
        kf_vel_msg.header.frame_id = "odom"; // Oder ein passender Frame
        kf_vel_msg.twist.twist.linear.x = u_odom(0);
        kf_vel_msg.twist.twist.angular.z = u_odom(1);
        ekf_vel_msg = kf_vel_msg; // EKF-Fallback ist dasselbe wie KF-Fallback
        kf_vel_pub_.publish(kf_vel_msg);
        ekf_vel_pub_.publish(ekf_vel_msg);

        return;
    }

    double v_linear_wheels = wheel_radius_ * (right_wheel_vel + left_wheel_vel) / 2.0;
    double omega_angular_wheels = wheel_radius_ * (right_wheel_vel - left_wheel_vel) / wheel_base_;

    if (dt_ > 1e-6) {
        double delta_s = v_linear_wheels * dt_;
        double delta_theta = omega_angular_wheels * dt_;

        current_kinematic_x_ += delta_s * std::cos(current_kinematic_yaw_ + delta_theta / 2.0);
        current_kinematic_y_ += delta_s * std::sin(current_kinematic_yaw_ + delta_theta / 2.0);
        current_kinematic_yaw_ = angles::normalize_angle(current_kinematic_yaw_ + delta_theta);
    }

    Eigen::VectorXd z_kf_extended(6);
    z_kf_extended(0) = current_kinematic_x_;
    z_kf_extended(1) = current_kinematic_y_;
    z_kf_extended(2) = yaw_imu;
    z_kf_extended(3) = v_linear_wheels * std::cos(yaw_imu);
    z_kf_extended(4) = v_linear_wheels * std::sin(yaw_imu);
    z_kf_extended(5) = omega_angular_wheels;

    Eigen::Vector2d z_ekf_original;
    z_ekf_original << yaw_imu, omega_imu;

    kf_.predict(u_odom);  kf_.update(z_kf_extended);
    ekf_.predict(u_odom); ekf_.update(z_ekf_original);

    toPoseMsg(kf_.state(), kf_.cov(), last_stamp_, "map", kf_msg);
    toPoseMsg(ekf_.state(),ekf_.cov(), last_stamp_, "map", ekf_msg);
    kf_pub_.publish(kf_msg);  ekf_pub_.publish(ekf_msg);

    // NEU: Geschätzte Geschwindigkeiten veröffentlichen
    kf_vel_msg.header.stamp = last_stamp_;
    kf_vel_msg.header.frame_id = "odom"; // Passender Frame, z.B. "odom" oder "base_link"
    kf_vel_msg.twist.twist.linear.x = kf_.state()(3); // v_x aus KF-Zustand
    kf_vel_msg.twist.twist.linear.y = kf_.state()(4); // v_y aus KF-Zustand
    kf_vel_msg.twist.twist.angular.z = kf_.state()(5); // omega aus KF-Zustand
    kf_vel_pub_.publish(kf_vel_msg);

    ekf_vel_msg.header.stamp = last_stamp_;
    ekf_vel_msg.header.frame_id = "odom"; // Passender Frame
    ekf_vel_msg.twist.twist.linear.x = ekf_.state()(3); // v_x aus EKF-Zustand
    ekf_vel_msg.twist.twist.linear.y = ekf_.state()(4); // v_y aus EKF-Zustand
    ekf_vel_msg.twist.twist.angular.z = ekf_.state()(5); // omega aus EKF-Zustand
    ekf_vel_pub_.publish(ekf_vel_msg);
  }

  LinearKF   kf_;
  ExtendedKF ekf_;

  message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
  message_filters::Subscriber<sensor_msgs::Imu>   imu_sub_;
  message_filters::Subscriber<sensor_msgs::JointState> joint_state_sub_;

  std::shared_ptr<
    message_filters::TimeSynchronizer<
      nav_msgs::Odometry,sensor_msgs::Imu,sensor_msgs::JointState>>       sync_;
  ros::Publisher kf_pub_, ekf_pub_;
  ros::Publisher kf_vel_pub_, ekf_vel_pub_; // NEU: Publisher für Geschwindigkeiten

  ros::Time last_stamp_;

  double wheel_radius_;
  double wheel_base_;

  double current_kinematic_x_;
  double current_kinematic_y_;
  double current_kinematic_yaw_;
  bool   is_first_measurement_;

  std::map<std::string, double> previous_joint_positions_;
  ros::Time last_joint_state_stamp_;
};

// ============================ 5. main ================================
int main(int argc,char**argv){
  ros::init(argc,argv,"filter_node_1");
  ros::NodeHandle nh("~");
  FilterNode node(nh);
  ros::spin();
  return 0;
}
