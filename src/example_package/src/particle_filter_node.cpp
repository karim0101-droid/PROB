// SPDX-License-Identifier: MIT
// particle_filter_node.cpp – Standalone Particle Filter (ROS 1)
// ------------------------------------------------------------------------
// Finale Version mit abgestimmten Rausch-Parametern, um eine effektive
// Gewichtung und ein aussagekräftiges Resampling zu fördern.

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseArray.h>
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
#include <random>

// ======================= 1.  PARTICLE FILTER KLASSE =====================
class ParticleFilter
{
public:
    struct Particle
    {
        Eigen::VectorXd state;
        double weight;
    };

    explicit ParticleFilter(int num_particles = 100, double dt = 0.01)
        : num_particles_(num_particles)
    {
        setDt(dt);
        generator_.seed(std::random_device{}());
        reset();
    }

    void setDt(double dt) { dt_ = dt; }

    void reset()
    {
        is_initialized_ = false;
        particles_.clear();
        particles_.resize(num_particles_);

        Eigen::VectorXd initial_state(6);
        initial_state << 0.5, 0.5, 0.0, 0.0, 0.0, 0.0;

        // Anfangsunsicherheit für eine breite Startverteilung
        Eigen::MatrixXd P_initial = Eigen::MatrixXd::Identity(6, 6);
        P_initial(0, 0) = 0.25 * 0.25; // std::dev von 25 cm in x
        P_initial(1, 1) = 0.25 * 0.25; // std::dev von 25 cm in y
        P_initial(2, 2) = 0.1 * 0.1;   // std::dev von ~6 Grad in yaw

        // Prozessrauschen Q: Sorgt für Vielfalt und verhindert, dass alle Partikel "im Gleichschritt" laufen.
        // Ein höheres Q erlaubt der Wolke, sich stärker auszubreiten und Fehler in der Odometrie auszugleichen.
        Q_ = Eigen::MatrixXd::Identity(6, 6);
        Q_(0, 0) = 0.05 * 0.05; // Erhöhte Unsicherheit in X-Bewegung (war 0.02)
        Q_(1, 1) = 0.05 * 0.05; // Erhöhte Unsicherheit in Y-Bewegung (war 0.02)
        Q_(2, 2) = 0.02 * 0.02; // Erhöhte Unsicherheit in der Drehung

        // Messrauschen R: Definiert das Vertrauen in die Sensormessung.
        // Ein kleineres R bedeutet, dass die Messung als genauer angesehen wird und die Partikel
        // stärker zur gemessenen Position "hingezogen" werden. Dies ermöglicht eine strenge Selektion.
        R_ = Eigen::MatrixXd::Identity(6, 6);
        R_(0, 0) = 0.05 * 0.05; // Hohes Vertrauen in die X-Messung
        R_(1, 1) = 0.05 * 0.05; // Hohes Vertrauen in die Y-Messung
        R_(2, 2) = 0.01 * 0.01; // Sehr hohes Vertrauen in die IMU-Orientierung
        R_(3, 3) = 0.05 * 0.05;
        R_(4, 4) = 0.05 * 0.05;
        R_(5, 5) = 0.02 * 0.02;
        R_inv_ = R_.inverse();

        std::normal_distribution<double> distX(0.0, std::sqrt(P_initial(0, 0)));
        std::normal_distribution<double> distY(0.0, std::sqrt(P_initial(1, 1)));
        std::normal_distribution<double> distYaw(0.0, std::sqrt(P_initial(2, 2)));

        for (int i = 0; i < num_particles_; ++i)
        {
            Eigen::VectorXd p_state = initial_state;
            p_state(0) += distX(generator_);
            p_state(1) += distY(generator_);
            p_state(2) += distYaw(generator_);
            p_state(2) = angles::normalize_angle(p_state(2));
            particles_[i].state = p_state;
            particles_[i].weight = 1.0 / num_particles_;
        }

        mean_state_ = initial_state;
        P_ = P_initial;
        is_initialized_ = true;
    }

    void predict(const Eigen::Vector2d &u)
    {
        if (!is_initialized_)
            return;
        double v = u(0);
        double w = u(1);

        std::normal_distribution<double> distQ_x(0.0, std::sqrt(Q_(0, 0)));
        std::normal_distribution<double> distQ_y(0.0, std::sqrt(Q_(1, 1)));
        std::normal_distribution<double> distQ_yaw(0.0, std::sqrt(Q_(2, 2)));

        for (auto &p : particles_)
        {
            double theta = p.state(2);
            p.state(0) += v * dt_ * std::cos(theta) + distQ_x(generator_);
            p.state(1) += v * dt_ * std::sin(theta) + distQ_y(generator_);
            p.state(2) += w * dt_ + distQ_yaw(generator_);
            p.state(2) = angles::normalize_angle(p.state(2));
            p.state(3) = v * std::cos(p.state(2));
            p.state(4) = v * std::sin(p.state(2));
            p.state(5) = w;
        }
    }

    void update(const Eigen::VectorXd &z)
    {
        if (!is_initialized_)
            return;
        double total_weight = 0.0;
        for (auto &p : particles_)
        {
            Eigen::VectorXd innovation = z - p.state;
            innovation(2) = angles::normalize_angle(innovation(2));
            double dist_sq = (innovation.transpose() * R_inv_ * innovation)(0, 0);
            p.weight = std::exp(-0.5 * dist_sq);
            total_weight += p.weight;
        }
        if (total_weight < 1e-9)
        {
            ROS_WARN_THROTTLE(1.0, "Total particle weight is near zero. Resetting weights to uniform.");
            for (auto &p : particles_)
            {
                p.weight = 1.0 / num_particles_;
            }
        }
        else
        {
            for (auto &p : particles_)
            {
                p.weight /= total_weight;
            }
        }
        resample();
        calculateMeanAndCovariance();
    }

    const Eigen::VectorXd &state() const { return mean_state_; }
    const Eigen::MatrixXd &cov() const { return P_; }
    int getNumParticles() const { return num_particles_; }
    const Eigen::VectorXd &getParticleState(int i) const { return particles_[i].state; }

private:
    void resample()
    {
        std::vector<Particle> new_particles;
        new_particles.reserve(num_particles_);
        std::uniform_real_distribution<double> dist(0.0, 1.0 / num_particles_);
        double r = dist(generator_);
        double c = particles_[0].weight;
        int i = 0;
        for (int m = 0; m < num_particles_; ++m)
        {
            double u = r + m * (1.0 / num_particles_);
            while (u > c && i < num_particles_ - 1)
            {
                i++;
                c += particles_[i].weight;
            }
            new_particles.push_back(particles_[i]);
        }
        particles_ = new_particles;
        for (auto &p : particles_)
        {
            p.weight = 1.0 / num_particles_;
        }
    }

    void calculateMeanAndCovariance()
    {
        mean_state_.setZero(6);
        for (const auto &p : particles_)
        {
            mean_state_ += p.state;
        }
        mean_state_ /= num_particles_;
        mean_state_(2) = angles::normalize_angle(mean_state_(2));
        P_.setZero(6, 6);
        for (const auto &p : particles_)
        {
            Eigen::VectorXd diff = p.state - mean_state_;
            diff(2) = angles::normalize_angle(diff(2));
            P_ += diff * diff.transpose();
        }
        P_ /= num_particles_;
    }

    int num_particles_;
    double dt_{0.01};
    bool is_initialized_{false};
    std::vector<Particle> particles_;
    Eigen::VectorXd mean_state_{Eigen::VectorXd::Zero(6)};
    Eigen::MatrixXd P_, Q_, R_, R_inv_;
    std::default_random_engine generator_;
};

// =========================== 2. Helper-Funktion ================================
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
    auto &cov = msg.pose.covariance;
    std::fill(cov.begin(), cov.end(), 0.0);
    cov[0 * 6 + 0] = P(0, 0);
    cov[0 * 6 + 1] = P(0, 1);
    cov[0 * 6 + 5] = P(0, 2);
    cov[1 * 6 + 0] = P(1, 0);
    cov[1 * 6 + 1] = P(1, 1);
    cov[1 * 6 + 5] = P(1, 2);
    cov[5 * 6 + 0] = P(2, 0);
    cov[5 * 6 + 1] = P(2, 1);
    cov[5 * 6 + 5] = P(2, 2);
}

// =========================== 3. ROS Node-Klasse ===============================
class FilterNode
{
public:
    explicit FilterNode(ros::NodeHandle &nh)
        : wheel_radius_(0.0), wheel_base_(0.0),
          current_kinematic_x_(0.5), current_kinematic_y_(0.5), current_kinematic_yaw_(0.0),
          is_first_measurement_(true), last_joint_state_stamp_(ros::Time(0))
    {
        int num_particles;
        nh.param("wheel_radius", wheel_radius_, 0.033);
        nh.param("wheel_base", wheel_base_, 0.160);
        nh.param("num_particles", num_particles, 500);
        pf_ = std::make_unique<ParticleFilter>(num_particles, 0.01);
        ROS_INFO("Particle Filter Node started with %d particles.", num_particles);

        odom_sub_.subscribe(nh, "/odom", 10);
        imu_sub_.subscribe(nh, "/imu", 10);
        joint_state_sub_.subscribe(nh, "/joint_states", 10);

        sync_ = std::make_shared<
            message_filters::TimeSynchronizer<
                nav_msgs::Odometry, sensor_msgs::Imu, sensor_msgs::JointState>>(
            odom_sub_, imu_sub_, joint_state_sub_, 10);
        sync_->registerCallback(boost::bind(&FilterNode::sensorCb, this, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3));

        pose_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/pf_pose", 10);
        odom_pub_ = nh.advertise<nav_msgs::Odometry>("/pf_odom", 10);
        particles_pub_ = nh.advertise<geometry_msgs::PoseArray>("/particle_cloud", 10);
    }

private:
    void sensorCb(const nav_msgs::Odometry::ConstPtr &odom,
                  const sensor_msgs::Imu::ConstPtr &imu,
                  const sensor_msgs::JointState::ConstPtr &joint_state)
    {
        double dt = 0.0;
        if (!last_stamp_.isZero())
        {
            dt = (odom->header.stamp - last_stamp_).toSec();
            if (dt > 1e-6)
            {
                pf_->setDt(dt);
            }
        }
        last_stamp_ = odom->header.stamp;
        if (dt <= 1e-6)
            return;

        Eigen::Vector2d u_odom;
        u_odom << odom->twist.twist.linear.x, odom->twist.twist.angular.z;
        tf2::Quaternion q_imu;
        tf2::fromMsg(imu->orientation, q_imu);
        double roll_imu, pitch_imu, yaw_imu;
        tf2::Matrix3x3(q_imu).getRPY(roll_imu, pitch_imu, yaw_imu);

        if (is_first_measurement_)
        {
            current_kinematic_yaw_ = yaw_imu;
            is_first_measurement_ = false;
        }

        double left_wheel_vel = 0.0, right_wheel_vel = 0.0;
        bool velocities_calculated = false;
        int left_idx = -1, right_idx = -1;
        for (size_t i = 0; i < joint_state->name.size(); ++i)
        {
            if (joint_state->name[i] == "wheel_left_joint")
                left_idx = i;
            else if (joint_state->name[i] == "wheel_right_joint")
                right_idx = i;
        }
        if (left_idx != -1 && right_idx != -1)
        {
            double dt_js = (joint_state->header.stamp - last_joint_state_stamp_).toSec();
            if (!last_joint_state_stamp_.isZero() && dt_js > 1e-6)
            {
                if (prev_joint_pos_.count("left") && prev_joint_pos_.count("right"))
                {
                    left_wheel_vel = (joint_state->position[left_idx] - prev_joint_pos_["left"]) / dt_js;
                    right_wheel_vel = (joint_state->position[right_idx] - prev_joint_pos_["right"]) / dt_js;
                    velocities_calculated = true;
                }
            }
            prev_joint_pos_["left"] = joint_state->position[left_idx];
            prev_joint_pos_["right"] = joint_state->position[right_idx];
            last_joint_state_stamp_ = joint_state->header.stamp;
        }

        pf_->predict(u_odom);

        if (velocities_calculated)
        {
            double v_wheels = wheel_radius_ * (right_wheel_vel + left_wheel_vel) / 2.0;
            double w_wheels = wheel_radius_ * (right_wheel_vel - left_wheel_vel) / wheel_base_;
            double delta_s = v_wheels * dt;
            double delta_theta = w_wheels * dt;
            current_kinematic_x_ += delta_s * std::cos(current_kinematic_yaw_ + delta_theta / 2.0);
            current_kinematic_y_ += delta_s * std::sin(current_kinematic_yaw_ + delta_theta / 2.0);
            current_kinematic_yaw_ = angles::normalize_angle(current_kinematic_yaw_ + delta_theta);
            Eigen::VectorXd z(6);
            z(0) = current_kinematic_x_;
            z(1) = current_kinematic_y_;
            z(2) = yaw_imu;
            z(3) = v_wheels * std::cos(yaw_imu);
            z(4) = v_wheels * std::sin(yaw_imu);
            z(5) = w_wheels;
            pf_->update(z);
        }
        else
        {
            ROS_WARN_THROTTLE(1.0, "Could not calculate wheel velocities. Skipping update step.");
        }
        publishResults();
    }

    void publishResults()
    {
        geometry_msgs::PoseWithCovarianceStamped pose_msg;
        toPoseMsg(pf_->state(), pf_->cov(), last_stamp_, "map", pose_msg);
        pose_pub_.publish(pose_msg);

        nav_msgs::Odometry odom_msg;
        odom_msg.header = pose_msg.header;
        odom_msg.child_frame_id = "base_link";
        odom_msg.pose = pose_msg.pose;
        odom_msg.twist.twist.linear.x = pf_->state()(3);
        odom_msg.twist.twist.linear.y = pf_->state()(4);
        odom_msg.twist.twist.angular.z = pf_->state()(5);
        odom_pub_.publish(odom_msg);

        geometry_msgs::PoseArray particles_msg;
        particles_msg.header.stamp = last_stamp_;
        particles_msg.header.frame_id = "map";
        particles_msg.poses.resize(pf_->getNumParticles());
        for (int i = 0; i < pf_->getNumParticles(); ++i)
        {
            const auto &p_state = pf_->getParticleState(i);
            particles_msg.poses[i].position.x = p_state(0);
            particles_msg.poses[i].position.y = p_state(1);
            tf2::Quaternion q;
            q.setRPY(0, 0, p_state(2));
            particles_msg.poses[i].orientation = tf2::toMsg(q);
        }
        particles_pub_.publish(particles_msg);
    }

    std::unique_ptr<ParticleFilter> pf_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    message_filters::Subscriber<sensor_msgs::JointState> joint_state_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu, sensor_msgs::JointState>> sync_;
    ros::Publisher pose_pub_;
    ros::Publisher odom_pub_;
    ros::Publisher particles_pub_;
    ros::Time last_stamp_;
    double wheel_radius_, wheel_base_;
    double current_kinematic_x_, current_kinematic_y_, current_kinematic_yaw_;
    bool is_first_measurement_;
    std::map<std::string, double> prev_joint_pos_;
    ros::Time last_joint_state_stamp_;
};

// ============================ 4. main ================================
int main(int argc, char **argv)
{
    ros::init(argc, argv, "particle_filter_node");
    ros::NodeHandle nh("~");
    FilterNode node(nh);
    ros::spin();
    return 0;
}