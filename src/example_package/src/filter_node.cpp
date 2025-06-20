#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "KalmanFilter.h"   // <<--- Hier bindest du DEINE Klasse ein

class FilterNode
{
public:
    FilterNode(ros::NodeHandle &nh)
        : kf_(0.05)   // dt = 0.05 (anpassen wie in deinem Konstruktor)
    {
        // --- Startpose via Parameter oder fest im Code ---
        double start_x, start_y, start_yaw;
        //nh.param("start_x", start_x, 0.5);
        //nh.param("start_y", start_y, 0.5);
        //nh.param("start_yaw", start_yaw, 0.0);

        Eigen::VectorXd mu0(6);
        mu0 << start_x, start_y, start_yaw, 0.5, 0.5, 0; // [x, y, theta, v_x, v_y, omega]
        Eigen::MatrixXd Sigma0 = Eigen::MatrixXd::Identity(6, 6) * 0.1;

        kf_.initialize(mu0, Sigma0);

        odom_sub_.subscribe(nh, "/odom", 10);
        imu_sub_.subscribe(nh, "/imu", 10);

        typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::Imu> MySyncPolicy;
        sync_.reset(new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), odom_sub_, imu_sub_));
        sync_->registerCallback(boost::bind(&FilterNode::sensorCallback, this, _1, _2));

        pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/prediction", 10);
    }

private:
    void sensorCallback(const nav_msgs::Odometry::ConstPtr &odom_msg, const sensor_msgs::Imu::ConstPtr &imu_msg)
    {
        // --- Prediction mit Odom: NUR linear.x und angular.z ---
        double v = odom_msg->twist.twist.linear.x;      // vorwärts
        double omega = odom_msg->twist.twist.angular.z; // Drehrate

        Eigen::VectorXd u(2);
        u << v, omega;

        kf_.predict(u); // deine Methode erwartet 2D-u

        // --- Measurement Update mit IMU ---
        // Extrahiere yaw aus IMU-Orientation
        tf2::Quaternion q_imu;
        tf2::fromMsg(imu_msg->orientation, q_imu);
        double roll, pitch, yaw_meas;
        tf2::Matrix3x3(q_imu).getRPY(roll, pitch, yaw_meas);

        // Nutze weitere Infos aus IMU nach Bedarf, z.B. acc, aber für updateMeasurement reicht Messvektor wie bisher
        Eigen::VectorXd z(5);
        // Hier nehmen wir die aktuelle Filter-Pose und ersetzen yaw durch IMU yaw
        Eigen::VectorXd mu = kf_.getState();
        z(0) = mu(0);            // x
        z(1) = mu(1);            // y
        z(2) = yaw_meas;         // theta (IMU)
        z(3) = mu(3);            // v_x
        z(4) = mu(4);            // v_y

        //kf_.updateMeasurement(z, true); // velocity_is_robot_frame = false

        // --- Publish ---
        Eigen::VectorXd mu_post = kf_.getState();
        Eigen::MatrixXd Sigma_post = kf_.getCovariance();

        geometry_msgs::PoseWithCovarianceStamped pred_msg;
        pred_msg.header.stamp = odom_msg->header.stamp;
        pred_msg.header.frame_id = "odom";
        pred_msg.pose.pose.position.x = mu_post(0);
        pred_msg.pose.pose.position.y = mu_post(1);
        pred_msg.pose.pose.position.z = 0.0;
        tf2::Quaternion q_out;
        q_out.setRPY(0, 0, mu_post(2));
        pred_msg.pose.pose.orientation = tf2::toMsg(q_out);

        // 6x6 Kovarianz-Matrix ins ROS-Format schreiben
        for(int i=0; i<36; ++i) pred_msg.pose.covariance[i] = 0.0;
        pred_msg.pose.covariance[0] = Sigma_post(0,0);
        pred_msg.pose.covariance[1] = Sigma_post(0,1);
        pred_msg.pose.covariance[5] = Sigma_post(0,2);
        pred_msg.pose.covariance[6] = Sigma_post(1,0);
        pred_msg.pose.covariance[7] = Sigma_post(1,1);
        pred_msg.pose.covariance[11]= Sigma_post(1,2);
        pred_msg.pose.covariance[30]= Sigma_post(2,0);
        pred_msg.pose.covariance[31]= Sigma_post(2,1);
        pred_msg.pose.covariance[35]= Sigma_post(2,2);

        pub_.publish(pred_msg);
    }

    // --- Member ---
    KalmanFilter kf_;

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::Imu>>> sync_;

    ros::Publisher pub_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "filter_node");
    ros::NodeHandle nh("~"); // privater Namespace für Parameter
    FilterNode node(nh);
    ros::spin();
    return 0;
}
