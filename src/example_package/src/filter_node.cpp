// SPDX-License-Identifier: MIT
// filter_node.cpp – Linear Kalman Filter vs. Extended Kalman Filter vs. Particle Filter (ROS 1)
// ------------------------------------------------------------------------
// This ROS node demonstrates and compares three filters:
// 1. Linear Kalman Filter (KF)
// 2. Extended Kalman Filter (EKF)
// 3. Particle Filter (PF)
//
// Inputs: Odometry, IMU, and JointState messages
// Outputs: Pose and velocity estimates for each filter

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <angles/angles.h>
#include <boost/bind/bind.hpp>
#include <cmath>
#include <memory>
#include <sensor_msgs/JointState.h>
#include <map>

// Custom filters
#include "linear_kf.h"
#include "extended_kf.h"
#include "particle_filter.h"

// Converts Eigen-based state and covariance into a ROS PoseWithCovarianceStamped
static void toPoseMsg(const Eigen::VectorXd &x,
                      const Eigen::MatrixXd &P,
                      const ros::Time &stamp,
                      const std::string &frame,
                      geometry_msgs::PoseWithCovarianceStamped &msg)
{
    msg.header.stamp = stamp;
    msg.header.frame_id = frame;

    msg.pose.pose.position.x = x(0);
    msg.pose.pose.position.y = x(1);

    tf2::Quaternion q;
    q.setRPY(0, 0, x(2)); // Only yaw used
    msg.pose.pose.orientation = tf2::toMsg(q);

    std::fill(std::begin(msg.pose.covariance), std::end(msg.pose.covariance), 0.0);
    msg.pose.covariance[0 * 6 + 0] = P(0, 0); // x
    msg.pose.covariance[1 * 6 + 1] = P(1, 1); // y
    msg.pose.covariance[5 * 6 + 5] = P(2, 2); // yaw
    msg.pose.covariance[0 * 6 + 5] = P(0, 2);
    msg.pose.covariance[5 * 6 + 0] = P(2, 0);
    msg.pose.covariance[1 * 6 + 5] = P(1, 2);
    msg.pose.covariance[5 * 6 + 1] = P(2, 1);
}

// Main filtering node
class FilterNode
{
public:
    explicit FilterNode(ros::NodeHandle &nh)
        : kf_(0.01), ekf_(0.01), pf_(0.01),
          wheel_radius_(0.0), wheel_base_(0.0),
          current_kinematic_x_(0.5), current_kinematic_y_(0.5), current_kinematic_yaw_(0.0),
          is_first_measurement_(true), last_joint_state_stamp_(ros::Time(0))
    {
        // Load parameters
        nh.param("wheel_radius", wheel_radius_, 0.033);
        nh.param("wheel_base", wheel_base_, 0.160);
        int num_particles_param;
        nh.param("num_particles", num_particles_param, 1000);
        pf_ = ParticleFilter(0.01, num_particles_param);

        // Setup subscribers
        odom_sub_.subscribe(nh, "/odom", 10);
        imu_sub_.subscribe(nh, "/imu", 10);
        joint_state_sub_.subscribe(nh, "/joint_states", 10);

        sync_ = std::make_shared<message_filters::TimeSynchronizer<
            nav_msgs::Odometry, sensor_msgs::Imu, sensor_msgs::JointState>>(
            odom_sub_, imu_sub_, joint_state_sub_, 10);
        sync_->registerCallback(boost::bind(&FilterNode::sensorCb, this, _1, _2, _3));

        // Setup publishers
        kf_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/kf_prediction", 10);
        ekf_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/ekf_prediction", 10);
        pf_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/pf_prediction", 10);

        kf_vel_pub_ = nh.advertise<nav_msgs::Odometry>("/kf_velocity_prediction", 10);
        ekf_vel_pub_ = nh.advertise<nav_msgs::Odometry>("/ekf_velocity_prediction", 10);
        pf_vel_pub_ = nh.advertise<nav_msgs::Odometry>("/pf_velocity_prediction", 10);
    }

private:
    void sensorCb(const nav_msgs::Odometry::ConstPtr &odom,
                  const sensor_msgs::Imu::ConstPtr &imu,
                  const sensor_msgs::JointState::ConstPtr &joint_state)
    {
        // --- TIMING AND DT ---
        double dt_ = 0.0;
        if (!last_stamp_.isZero())
        {
            double dt = (odom->header.stamp - last_stamp_).toSec();
            if (dt > 1e-6)
            {
                kf_.setDt(dt);
                ekf_.setDt(dt);
                pf_.setDt(dt);
                dt_ = dt;
            }
        }
        last_stamp_ = odom->header.stamp;

        // --- CONTROL INPUT (u) ---
        Eigen::Vector2d u_odom;
        u_odom << odom->twist.twist.linear.x,
                  odom->twist.twist.angular.z;

        // --- EXTRACT IMU DATA ---
        tf2::Quaternion q_imu;
        tf2::fromMsg(imu->orientation, q_imu);
        double roll_imu, pitch_imu, yaw_imu;
        tf2::Matrix3x3(q_imu).getRPY(roll_imu, pitch_imu, yaw_imu);
        yaw_imu = angles::normalize_angle(yaw_imu);
        double omega_imu = imu->angular_velocity.z;

        if (is_first_measurement_)
        {
            current_kinematic_x_ = 0.5;
            current_kinematic_y_ = 0.5;
            current_kinematic_yaw_ = yaw_imu;
            is_first_measurement_ = false;
        }

        // --- COMPUTE WHEEL VELOCITIES FROM JOINTS ---
        double left_wheel_vel = 0.0, right_wheel_vel = 0.0;
        bool velocities_calculated = false;
        int left_idx = -1, right_idx = -1;

        for (size_t i = 0; i < joint_state->name.size(); ++i)
        {
            if (joint_state->name[i] == "wheel_left_joint") left_idx = i;
            if (joint_state->name[i] == "wheel_right_joint") right_idx = i;
        }

        if (left_idx != -1 && right_idx != -1 &&
            joint_state->position.size() > std::max(left_idx, right_idx))
        {
            if (!last_joint_state_stamp_.isZero() && joint_state->header.stamp > last_joint_state_stamp_)
            {
                double dt_joint = (joint_state->header.stamp - last_joint_state_stamp_).toSec();
                if (dt_joint > 1e-6)
                {
                    double pos_left = joint_state->position[left_idx];
                    double pos_right = joint_state->position[right_idx];

                    if (previous_joint_positions_.count("wheel_left_joint") &&
                        previous_joint_positions_.count("wheel_right_joint"))
                    {
                        left_wheel_vel = (pos_left - previous_joint_positions_["wheel_left_joint"]) / dt_joint;
                        right_wheel_vel = (pos_right - previous_joint_positions_["wheel_right_joint"]) / dt_joint;
                        velocities_calculated = true;
                    }
                }
            }
            previous_joint_positions_["wheel_left_joint"] = joint_state->position[left_idx];
            previous_joint_positions_["wheel_right_joint"] = joint_state->position[right_idx];
            last_joint_state_stamp_ = joint_state->header.stamp;
        }
        else
        {
            ROS_WARN_THROTTLE(1.0, "Joint state data incomplete. Skipping wheel velocity computation.");
        }

        geometry_msgs::PoseWithCovarianceStamped kf_msg, ekf_msg, pf_msg;
        nav_msgs::Odometry kf_vel_msg, ekf_vel_msg, pf_vel_msg;

        // --- FALLBACK (no valid joint velocities) ---
        if (!velocities_calculated)
        {
            ROS_WARN_THROTTLE(1.0, "Falling back to odometry + IMU due to missing joint velocity.");

            Eigen::VectorXd z_kf_fallback(6);
            z_kf_fallback << kf_.state()(0), kf_.state()(1), yaw_imu,
                              kf_.state()(3), kf_.state()(4), omega_imu;

            kf_.predict(u_odom);
            kf_.update(z_kf_fallback);

            ekf_.predict(u_odom);
            ekf_.update(Eigen::Vector2d(yaw_imu, omega_imu));

            pf_.predict(u_odom);
            pf_.update(Eigen::Vector2d(yaw_imu, omega_imu));

            // Publish estimates
            toPoseMsg(kf_.state(), kf_.cov(), last_stamp_, "map", kf_msg);
            toPoseMsg(ekf_.state(), ekf_.cov(), last_stamp_, "map", ekf_msg);
            toPoseMsg(pf_.state(), pf_.cov(), last_stamp_, "map", pf_msg);
            kf_pub_.publish(kf_msg);
            ekf_pub_.publish(ekf_msg);
            pf_pub_.publish(pf_msg);

            // Publish velocities
            auto publishVel = [&](nav_msgs::Odometry &msg, const Eigen::VectorXd &x) {
                msg.header.stamp = last_stamp_;
                msg.header.frame_id = "odom";
                msg.twist.twist.linear.x = x(3);
                msg.twist.twist.linear.y = x(4);
                msg.twist.twist.angular.z = x(5);
            };
            publishVel(kf_vel_msg, kf_.state());
            publishVel(ekf_vel_msg, ekf_.state());
            publishVel(pf_vel_msg, pf_.state());

            kf_vel_pub_.publish(kf_vel_msg);
            ekf_vel_pub_.publish(ekf_vel_msg);
            pf_vel_pub_.publish(pf_vel_msg);
            return;
        }

        // --- If valid joint velocities are available ---
        double v = wheel_radius_ * (right_wheel_vel + left_wheel_vel) / 2.0;
        double omega = wheel_radius_ * (right_wheel_vel - left_wheel_vel) / wheel_base_;

        if (dt_ > 1e-6)
        {
            // Update pose estimate from simple kinematic model
            current_kinematic_x_ += v * dt_ * std::cos(current_kinematic_yaw_ + omega * dt_ / 2.0);
            current_kinematic_y_ += v * dt_ * std::sin(current_kinematic_yaw_ + omega * dt_ / 2.0);
            current_kinematic_yaw_ = angles::normalize_angle(current_kinematic_yaw_ + omega * dt_);
        }

        // KF full measurement vector (6D)
        Eigen::VectorXd z_kf(6);
        z_kf << current_kinematic_x_, current_kinematic_y_, yaw_imu,
                v * std::cos(current_kinematic_yaw_),
                v * std::sin(current_kinematic_yaw_),
                omega;

        // EKF and PF use only yaw and omega from IMU
        Eigen::Vector2d z_imu(yaw_imu, omega_imu);

        kf_.predict(u_odom);
        kf_.update(z_kf);

        ekf_.predict(u_odom);
        ekf_.update(z_imu);

        pf_.predict(u_odom);
        pf_.update(z_imu);

        toPoseMsg(kf_.state(), kf_.cov(), last_stamp_, "map", kf_msg);
        toPoseMsg(ekf_.state(), ekf_.cov(), last_stamp_, "map", ekf_msg);
        toPoseMsg(pf_.state(), pf_.cov(), last_stamp_, "map", pf_msg);

        kf_pub_.publish(kf_msg);
        ekf_pub_.publish(ekf_msg);
        pf_pub_.publish(pf_msg);

        auto publishVel = [&](nav_msgs::Odometry &msg, const Eigen::VectorXd &x) {
            msg.header.stamp = last_stamp_;
            msg.header.frame_id = "odom";
            msg.twist.twist.linear.x = x(3);
            msg.twist.twist.linear.y = x(4);
            msg.twist.twist.angular.z = x(5);
        };
        publishVel(kf_vel_msg, kf_.state());
        publishVel(ekf_vel_msg, ekf_.state());
        publishVel(pf_vel_msg, pf_.state());

        kf_vel_pub_.publish(kf_vel_msg);
        ekf_vel_pub_.publish(ekf_vel_msg);
        pf_vel_pub_.publish(pf_vel_msg);
    }

    // Filter instances
    LinearKF kf_;
    ExtendedKF ekf_;
    ParticleFilter pf_;

    // Message filters
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    message_filters::Subscriber<sensor_msgs::JointState> joint_state_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<
        nav_msgs::Odometry, sensor_msgs::Imu, sensor_msgs::JointState>> sync_;

    // Publishers
    ros::Publisher kf_pub_, ekf_pub_, pf_pub_;
    ros::Publisher kf_vel_pub_, ekf_vel_pub_, pf_vel_pub_;

    // Internal state tracking
    ros::Time last_stamp_;
    double wheel_radius_, wheel_base_;
    double current_kinematic_x_, current_kinematic_y_, current_kinematic_yaw_;
    bool is_first_measurement_;
    std::map<std::string, double> previous_joint_positions_;
    ros::Time last_joint_state_stamp_;
};

// Entry point
int main(int argc, char **argv)
{
    ros::init(argc, argv, "filter_node");
    ros::NodeHandle nh("~");
    FilterNode node(nh);
    ros::spin();
    return 0;
}
