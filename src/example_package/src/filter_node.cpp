#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "KalmanFilter.h"

class FilterNode
{
public:
    FilterNode(ros::NodeHandle &nh)
    {
        // Member-Initialisierung wie vorher, aber korrekt als Member
        states = Eigen::VectorXd::Zero(6);
        states(0) = 0.5; // Startwert x
        states(1) = 0.5; // Startwert y
        // states(2)...states(5) sind 0

        input = Eigen::VectorXd::Zero(2);
        measured = Eigen::VectorXd::Zero(6);
        measured(0) = 0.5;
        measured(1) = 0.5;

        vel_robot = {0.0, 0.0, 0.0};

        odom_sub_.subscribe(nh, "/odom", 10);
        imu_sub_.subscribe(nh, "/imu", 10);

        sync_.reset(new message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu>(odom_sub_, imu_sub_, 10));
        sync_->registerCallback(boost::bind(&FilterNode::sensorCallback, this, _1, _2));

        pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/KF", 10);
    }

private:
    ros::Time last_time_;
    KalmanFilter kf;

    Eigen::VectorXd states;   // 6x1
    Eigen::VectorXd input;    // 2x1
    Eigen::VectorXd measured; // 6x1

    struct r
    {
        double x;
        double y;
        double phi;
    };
    r vel_robot;
    r pos_robot;

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu>> sync_;
    ros::Publisher pub_;

    void sensorCallback(const nav_msgs::Odometry::ConstPtr &odom_msg, const sensor_msgs::Imu::ConstPtr &imu_msg)
    {
        //---- Zeitunterschied ----
        ros::Time current_time = odom_msg->header.stamp;
        double dt = 0.0;
        if (!last_time_.isZero())
        {
            ros::Duration delta = current_time - last_time_;
            dt = delta.toSec();
            ROS_INFO_STREAM("Zeit seit letzter Nachricht: " << dt << " Sekunden");
        }
        else
        {
            last_time_ = current_time;
            return; // Erster Callback: noch kein Delta!
        }
        last_time_ = current_time;

        //---- Eingangsvektor ----
        input(0) = odom_msg->twist.twist.linear.x;
        input(1) = odom_msg->twist.twist.angular.z;

        //std::cout << "input: " << input << std::endl;

        //---- Measurement Vektor aus IMU ----
        // IMU-Integration
        vel_robot.x = imu_msg->linear_acceleration.x * dt;
        //vel_robot.y = imu_msg->linear_acceleration.y * dt;
        vel_robot.y = 0.0;
        vel_robot.phi = imu_msg->angular_velocity.z;
        

        
        tf2::Quaternion q;
        tf2::fromMsg(imu_msg->orientation, q);
        double roll, pitch, yaw;

        // In Weltkoordinaten umrechnen (aus Roboter-KS)
        measured(3) = vel_robot.x * std::cos(yaw); //- vel_robot.y * std::sin(yaw);
        measured(4) = vel_robot.x * std::sin(yaw); // + vel_robot.y * std::cos(yaw);
        measured(5) = vel_robot.phi;

        
        // Positionsintegration (numerisch)
        measured(0) = measured(3) * dt;
        measured(1) = measured(4) * dt;
        measured(2) = yaw;

        //---- KalmanFilter-Schritt (std::pair) ----
        auto [filtered_state, filtered_cov] = kf.algorithm(states, kf.getCovariance(), input, measured, dt);
        states = filtered_state;

        //std::cout << "Filtered States: " << states(3) << std::endl;
        //---- Publisher (wie gehabt, hier beispielhaft) ----
        geometry_msgs::PoseWithCovarianceStamped out_msg;
        out_msg.header.stamp = current_time;
        out_msg.header.frame_id = "odom";
        out_msg.pose.pose.position.x = states(0);
        out_msg.pose.pose.position.y = states(1);
        out_msg.pose.pose.position.z = 0;

        ROS_INFO_STREAM("dt: " << dt);

        ROS_INFO_STREAM("input: " << input.transpose());
        ROS_INFO_STREAM("measured: " << measured.transpose());


        tf2::Quaternion q_out;
        q_out.setRPY(0, 0, states(2));
        out_msg.pose.pose.orientation = tf2::toMsg(q_out);

        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                out_msg.pose.covariance[i * 6 + j] = filtered_cov(i, j);

        pub_.publish(out_msg);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "filter_node");
    ros::NodeHandle nh("~");
    FilterNode node(nh);
    ros::spin();
    return 0;
}
