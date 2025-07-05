// SPDX-License-Identifier: MIT
// filter_node.cpp – Linear Kalman Filter vs. Extended Kalman Filter vs. Particle Filter (ROS 1)
// ------------------------------------------------------------------------

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
#include <algorithm> // Für std::max
#include <cmath>     // Für std::cos, std::sin
#include <memory>
#include <sensor_msgs/JointState.h>
#include <map>

// Include the custom filter headers
#include "linear_kf.h"
#include "extended_kf.h"
#include "particle_filter.h"

// Helper-Funktion (kann bei Bedarf auch ausgelagert werden)
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
    q.setRPY(0, 0, x(2));
    msg.pose.pose.orientation = tf2::toMsg(q);

    std::fill(std::begin(msg.pose.covariance),
              std::end(msg.pose.covariance), 0.0);
    msg.pose.covariance[0 * 6 + 0] = P(0, 0);
    msg.pose.covariance[0 * 6 + 1] = P(0, 1);
    msg.pose.covariance[1 * 6 + 0] = P(1, 0);
    msg.pose.covariance[1 * 6 + 1] = P(1, 1);
    msg.pose.covariance[5 * 6 + 5] = P(2, 2);
    msg.pose.covariance[0 * 6 + 5] = P(0, 2);
    msg.pose.covariance[5 * 6 + 0] = P(2, 0);
    msg.pose.covariance[1 * 6 + 5] = P(1, 2);
    msg.pose.covariance[5 * 6 + 1] = P(2, 1);
}

// 5. ROS Node
class FilterNode
{
public:
    explicit FilterNode(ros::NodeHandle &nh)
        : kf_(0.01), ekf_(0.01), pf_(0.01), // PF-Instanz hinzugefügt
          wheel_radius_(0.0), wheel_base_(0.0),
          current_kinematic_x_(0.5), current_kinematic_y_(0.5), current_kinematic_yaw_(0.0),
          is_first_measurement_(true), last_joint_state_stamp_(ros::Time(0))
    {

        nh.param("wheel_radius", wheel_radius_, 0.033);
        nh.param("wheel_base", wheel_base_, 0.160);
        int num_particles_param;
        nh.param("num_particles", num_particles_param, 1000); // Anzahl der Partikel als Parameter
        pf_ = ParticleFilter(0.01, num_particles_param);      // PF mit Parameter initialisieren

        odom_sub_.subscribe(nh, "/odom", 10);
        imu_sub_.subscribe(nh, "/imu", 10);
        joint_state_sub_.subscribe(nh, "/joint_states", 10);

        sync_ = std::make_shared<
            message_filters::TimeSynchronizer<
                nav_msgs::Odometry, sensor_msgs::Imu, sensor_msgs::JointState>>(
            odom_sub_, imu_sub_, joint_state_sub_, 10);
        using namespace boost::placeholders;
        sync_->registerCallback(boost::bind(&FilterNode::sensorCb, this, _1, _2, _3));

        kf_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>(
            "/kf_prediction", 10);
        ekf_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>(
            "/ekf_prediction", 10);
        pf_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>( // PF Publisher
            "/pf_prediction", 10);

        kf_vel_pub_ = nh.advertise<nav_msgs::Odometry>("/kf_velocity_prediction", 10);
        ekf_vel_pub_ = nh.advertise<nav_msgs::Odometry>("/ekf_velocity_prediction", 10);
        pf_vel_pub_ = nh.advertise<nav_msgs::Odometry>("/pf_velocity_prediction", 10); // PF Velocity Publisher
    }

private:
    void sensorCb(const nav_msgs::Odometry::ConstPtr &odom,
                  const sensor_msgs::Imu ::ConstPtr &imu,
                  const sensor_msgs::JointState::ConstPtr &joint_state)
    {

        double dt_ = 0.0;
        if (!last_stamp_.isZero())
        {
            double dt = (odom->header.stamp - last_stamp_).toSec();
            if (dt > 1e-6)
            {
                kf_.setDt(dt);
                ekf_.setDt(dt);
                pf_.setDt(dt); // PF dt setzen
                dt_ = dt;
            }
        }
        last_stamp_ = odom->header.stamp;

        // Odometrie als Steuerungseingabe (predict step) bleibt erhalten
        Eigen::Vector2d u_odom;
        u_odom << odom->twist.twist.linear.x,
            odom->twist.twist.angular.z;

        tf2::Quaternion q_imu;
        tf2::fromMsg(imu->orientation, q_imu);
        double roll_imu, pitch_imu, yaw_imu;
        tf2::Matrix3x3(q_imu).getRPY(roll_imu, pitch_imu, yaw_imu);
        yaw_imu = angles::normalize_angle(yaw_imu);
        double omega_imu = imu->angular_velocity.z;

        if (is_first_measurement_)
        {
            // Initialisiere Kinematik-Schätzung (current_kinematic_x/y/yaw)
            // mit festen Werten, um nicht von odom-Messungen abhängig zu sein
            current_kinematic_x_ = 0.5;
            current_kinematic_y_ = 0.5;
            current_kinematic_yaw_ = yaw_imu; // Yaw kann von IMU genommen werden
            is_first_measurement_ = false;
        }

        // JointState-Verarbeitung zur Geschwindigkeitsberechnung
        // Diese Kinematik wird nur für den KF-Messvektor verwendet,
        // und dient zur Berechnung des Messvektors für den KF.
        double left_wheel_vel = 0.0;
        double right_wheel_vel = 0.0;
        bool velocities_calculated = false;

        int left_joint_idx = -1;
        int right_joint_idx = -1;
        for (size_t i = 0; i < joint_state->name.size(); ++i)
        {
            if (joint_state->name[i] == "wheel_left_joint")
            {
                left_joint_idx = i;
            }
            else if (joint_state->name[i] == "wheel_right_joint")
            {
                right_joint_idx = i;
            }
        }

        if (left_joint_idx != -1 && right_joint_idx != -1 && joint_state->position.size() > std::max(left_joint_idx, right_joint_idx))
        {
            if (!last_joint_state_stamp_.isZero() && joint_state->header.stamp > last_joint_state_stamp_)
            {
                double dt_joint_state = (joint_state->header.stamp - last_joint_state_stamp_).toSec();
                if (dt_joint_state > 1e-6)
                {
                    double current_left_pos = joint_state->position[left_joint_idx];
                    double current_right_pos = joint_state->position[right_joint_idx];

                    if (previous_joint_positions_.count("wheel_left_joint") && previous_joint_positions_.count("wheel_right_joint"))
                    {
                        left_wheel_vel = (current_left_pos - previous_joint_positions_["wheel_left_joint"]) / dt_joint_state;
                        right_wheel_vel = (current_right_pos - previous_joint_positions_["wheel_right_joint"]) / dt_joint_state;
                        velocities_calculated = true;
                    }
                }
            }
            previous_joint_positions_["wheel_left_joint"] = joint_state->position[left_joint_idx];
            previous_joint_positions_["wheel_right_joint"] = joint_state->position[right_joint_idx];
            last_joint_state_stamp_ = joint_state->header.stamp;
        }
        else
        {
            ROS_WARN_THROTTLE(1.0, "Wheel joint names or positions not found in joint_states. Cannot calculate velocities from positions.");
        }

        geometry_msgs::PoseWithCovarianceStamped kf_msg, ekf_msg, pf_msg;
        nav_msgs::Odometry kf_vel_msg, ekf_vel_msg, pf_vel_msg;

        // Fallback-Logik für den Fall, dass keine JointState-Geschwindigkeiten berechnet werden können
        if (!velocities_calculated)
        {
            ROS_WARN_THROTTLE(1.0, "Falling back to IMU and Odometry for velocity updates due to missing joint state velocity calculation.");

            // KF Fallback: Nutzt eigene Schätzung für Pos/Vel, IMU für Yaw/Omega
            Eigen::VectorXd z_kf_fallback(6);
            z_kf_fallback << kf_.state()(0), kf_.state()(1), yaw_imu, kf_.state()(3), kf_.state()(4), omega_imu;
            kf_.predict(u_odom);
            kf_.update(z_kf_fallback);

            // EKF Fallback: Nutzt NUR IMU für Yaw/Omega (unverändert)
            Eigen::Vector2d z_ekf_fallback(yaw_imu, omega_imu);
            ekf_.predict(u_odom);
            ekf_.update(z_ekf_fallback);

            // PF Fallback: Nutzt IMU für Yaw/Omega (2D Messvektor)
            Eigen::Vector2d z_pf_imu_only(yaw_imu, omega_imu);

            pf_.predict(u_odom);
            pf_.update(z_pf_imu_only); // PF Update mit 2D Messung

            toPoseMsg(kf_.state(), kf_.cov(), last_stamp_, "map", kf_msg);
            toPoseMsg(ekf_.state(), ekf_.cov(), last_stamp_, "map", ekf_msg);
            toPoseMsg(pf_.state(), pf_.cov(), last_stamp_, "map", pf_msg); // PF Msg publizieren

            kf_pub_.publish(kf_msg);
            ekf_pub_.publish(ekf_msg);
            pf_pub_.publish(pf_msg);

            kf_vel_msg.header.stamp = last_stamp_;
            kf_vel_msg.header.frame_id = "odom";
            kf_vel_msg.twist.twist.linear.x = kf_.state()(3);
            kf_vel_msg.twist.twist.angular.z = kf_.state()(5);
            ekf_vel_msg.header.stamp = last_stamp_;
            ekf_vel_msg.header.frame_id = "odom";
            ekf_vel_msg.twist.twist.linear.x = ekf_.state()(3);
            ekf_vel_msg.twist.twist.angular.z = ekf_.state()(5);
            pf_vel_msg.header.stamp = last_stamp_;
            pf_vel_msg.header.frame_id = "odom";
            pf_vel_msg.twist.twist.linear.x = pf_.state()(3);
            pf_vel_msg.twist.twist.linear.y = pf_.state()(4);
            pf_vel_msg.twist.twist.angular.z = pf_.state()(5);

            kf_vel_pub_.publish(kf_vel_msg);
            ekf_vel_pub_.publish(ekf_vel_msg);
            pf_vel_pub_.publish(pf_vel_msg);

            return;
        }

        // Wenn JointState-Geschwindigkeiten verfügbar sind
        double v_linear_wheels = wheel_radius_ * (right_wheel_vel + left_wheel_vel) / 2.0;
        double omega_angular_wheels = wheel_radius_ * (right_wheel_vel - left_wheel_vel) / wheel_base_;

        if (dt_ > 1e-6)
        {
            // Die Integration der kinemantischen Position dient hier als ZWISCHENWERT für den Messvektor des KF.
            // PF und EKF nutzen nur IMU-Messungen für ihren Update-Schritt.
            current_kinematic_x_ += v_linear_wheels * dt_ * std::cos(current_kinematic_yaw_ + omega_angular_wheels * dt_ / 2.0);
            current_kinematic_y_ += v_linear_wheels * dt_ * std::sin(current_kinematic_yaw_ + omega_angular_wheels * dt_ / 2.0);
            current_kinematic_yaw_ = angles::normalize_angle(current_kinematic_yaw_ + omega_angular_wheels * dt_);
        }

        // MESSVEKTOR FÜR LINEAR KALMAN FILTER (KF)
        // Behält die 6D-Messung bei, um volle Beobachtbarkeit zu haben.
        Eigen::VectorXd z_kf_full(6);
        z_kf_full(0) = current_kinematic_x_;
        z_kf_full(1) = current_kinematic_y_;
        z_kf_full(2) = yaw_imu;
        z_kf_full(3) = v_linear_wheels * std::cos(current_kinematic_yaw_); // Geschwindigkeit basierend auf Kinematik
        z_kf_full(4) = v_linear_wheels * std::sin(current_kinematic_yaw_); // Geschwindigkeit basierend auf Kinematik
        z_kf_full(5) = omega_angular_wheels;

        // MESSVEKTOR FÜR EXTENDED KALMAN FILTER (EKF)
        // Nur Yaw und Omega von der IMU (unverändert wie vom Nutzer vorgegeben)
        Eigen::Vector2d z_ekf_imu_only;
        z_ekf_imu_only << yaw_imu, omega_imu;

        // MESSVEKTOR FÜR PARTIKEL FILTER (PF)
        // Nur Yaw und Omega von der IMU
        Eigen::Vector2d z_pf_imu_only;
        z_pf_imu_only << yaw_imu, omega_imu;

        kf_.predict(u_odom);
        kf_.update(z_kf_full);
        ekf_.predict(u_odom);
        ekf_.update(z_ekf_imu_only); // EKF Update mit 2D Messung
        pf_.predict(u_odom);
        pf_.update(z_pf_imu_only); // PF Update mit 2D Messung

        toPoseMsg(kf_.state(), kf_.cov(), last_stamp_, "map", kf_msg);
        toPoseMsg(ekf_.state(), ekf_.cov(), last_stamp_, "map", ekf_msg);
        toPoseMsg(pf_.state(), pf_.cov(), last_stamp_, "map", pf_msg);

        kf_pub_.publish(kf_msg);
        ekf_pub_.publish(ekf_msg);
        pf_pub_.publish(pf_msg);

        kf_vel_msg.header.stamp = last_stamp_;
        kf_vel_msg.header.frame_id = "odom";
        kf_vel_msg.twist.twist.linear.x = kf_.state()(3);
        kf_vel_msg.twist.twist.linear.y = kf_.state()(4);
        kf_vel_msg.twist.twist.angular.z = kf_.state()(5);
        kf_vel_pub_.publish(kf_vel_msg);

        ekf_vel_msg.header.stamp = last_stamp_;
        ekf_vel_msg.header.frame_id = "odom";
        ekf_vel_msg.twist.twist.linear.x = ekf_.state()(3);
        ekf_vel_msg.twist.twist.linear.y = ekf_.state()(4);
        ekf_vel_msg.twist.twist.angular.z = ekf_.state()(5);
        ekf_vel_pub_.publish(ekf_vel_msg);

        pf_vel_msg.header.stamp = last_stamp_;
        pf_vel_msg.header.frame_id = "odom";
        pf_vel_msg.twist.twist.linear.x = pf_.state()(3);
        pf_vel_msg.twist.twist.linear.y = pf_.state()(4);
        pf_vel_msg.twist.twist.angular.z = pf_.state()(5);
        pf_vel_pub_.publish(pf_vel_msg);
    }

    LinearKF kf_;
    ExtendedKF ekf_;
    ParticleFilter pf_; // Partikelfilter Instanz

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    message_filters::Subscriber<sensor_msgs::JointState> joint_state_sub_;

    std::shared_ptr<
        message_filters::TimeSynchronizer<
            nav_msgs::Odometry, sensor_msgs::Imu, sensor_msgs::JointState>>
        sync_;
    ros::Publisher kf_pub_, ekf_pub_, pf_pub_;             // PF Publisher hinzugefügt
    ros::Publisher kf_vel_pub_, ekf_vel_pub_, pf_vel_pub_; // PF Velocity Publisher hinzugefügt

    ros::Time last_stamp_;

    double wheel_radius_;
    double wheel_base_;

    double current_kinematic_x_;
    double current_kinematic_y_;
    double current_kinematic_yaw_;
    bool is_first_measurement_;

    std::map<std::string, double> previous_joint_positions_;
    ros::Time last_joint_state_stamp_;
};

// 6. Main Funktion (Unverändert)
int main(int argc, char **argv)
{
    ros::init(argc, argv, "filter_node");
    ros::NodeHandle nh("~");
    FilterNode node(nh);
    ros::spin();
    return 0;
}