//// ============================================================================
////  filter_node.cpp   –  TurtleBot3 KF / EKF Pose Fusion
//// ============================================================================
//
//#include <ros/ros.h>
//#include <eigen3/Eigen/Dense>
//
//#include <nav_msgs/Odometry.h>
//#include <sensor_msgs/Imu.h>
//#include <sensor_msgs/JointState.h>
//#include <geometry_msgs/PoseWithCovarianceStamped.h>
//
//#include <tf2/LinearMath/Quaternion.h>
//#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
//
//#include <message_filters/subscriber.h>
//#include <message_filters/sync_policies/approximate_time.h>
//
//#include "KalmanFilter.h"
//#include "ExtendedKalmanFilter.h"
//
///* ========================================================================== */
//class FilterNode
//{
//    using SyncPol =
//        message_filters::sync_policies::ApproximateTime<
//            nav_msgs::Odometry, sensor_msgs::Imu, sensor_msgs::JointState>;
//    using Sync = message_filters::Synchronizer<SyncPol>;
//
//public:
//    explicit FilterNode(ros::NodeHandle &nh)
//    {
//        /* ---------- Parameter ---------------------------------------- */
//        nh.param("wheel_radius", wheel_radius_, 0.033);
//        nh.param("wheel_base", wheel_base_, 0.160);
//        nh.param<std::string>("left_wheel", left_joint_, "wheel_left_joint");
//        nh.param<std::string>("right_wheel", right_joint_, "wheel_right_joint");
//
//        nh.param("start_x", x_offset_, 0.5);
//        nh.param("start_y", y_offset_, 0.5);
//
//        /* ---------- ROS I/O ------------------------------------------ */
//        odom_sub_.subscribe(nh, "/odom", 20);
//        imu_sub_.subscribe(nh, "/imu", 50);
//        joint_sub_.subscribe(nh, "/joint_states", 50);
//
//        sync_.reset(new Sync(SyncPol(60), odom_sub_, imu_sub_, joint_sub_));
//        sync_->registerCallback(
//            boost::bind(&FilterNode::cb, this, _1, _2, _3));
//
//        pub_kf_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/KF", 10);
//        pub_ekf_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/EKF", 10);
//
//        /* ---------- KF-Rausch­matrix (6 × 6) ------------------------- */
//        Eigen::Matrix<double, 6, 6> R_kf = Eigen::Matrix<double, 6, 6>::Zero();
//        R_kf.diagonal() << 0.04 * 0.04, 0.04 * 0.04, 0.03 * 0.03,
//            0.03 * 0.03, 0.03 * 0.03, 0.012 * 0.012;
//        kf_.setMeasurementNoise(R_kf);
//        kf_.setProcessNoiseStd(0.08, 0.04);
//
//        /* ---------- EKF-Rausch­matrix (6 × 6) ------------------------ */
//        Eigen::Matrix<double, 6, 6> R_ekf = Eigen::Matrix<double, 6, 6>::Zero();
//        R_ekf.diagonal() << 0.05 * 0.05, 0.05 * 0.05, 0.03 * 0.03,
//            0.02 * 0.02, 0.02 * 0.02, 0.012 * 0.012;
//        ekf_.setMeasurementNoise(R_ekf);
//        ekf_.setProcessNoiseStd(0.08, 0.04);
//
//        /* ---------- Puffer anlegen ----------------------------------- */
//        kf_state_.setZero(6);
//        ekf_state_.setZero(6);
//        input_.setZero(2);
//        z_kf_.setZero(6);
//        z_ekf_.setZero(6);
//
//        x_fk_ = y_fk_ = 0.0;
//        last_pos_l_ = last_pos_r_ = 0.0;
//        pos_init_ = false;
//        last_stamp_ = ros::Time(0);
//    }
//
//private:
//    /* -------------------------------------------------------------------------- */
//    void cb(const nav_msgs::Odometry::ConstPtr &odom,
//            const sensor_msgs::Imu::ConstPtr &imu,
//            const sensor_msgs::JointState::ConstPtr &js)
//    {
//        /* ---------- Δt ------------------------------------------------ */
//        if (last_stamp_.isZero())
//        {
//            last_stamp_ = odom->header.stamp;
//            return;
//        }
//        double dt = (odom->header.stamp - last_stamp_).toSec();
//        last_stamp_ = odom->header.stamp;
//        if (dt < 1e-4)
//            return;
//        dt = std::min(dt, 0.05);
//
//        /* ---------- Joint-Positionen --------------------------------- */
//        double pos_l = 0.0, pos_r = 0.0;
//        for (size_t i = 0; i < js->name.size(); ++i)
//        {
//            if (js->name[i] == left_joint_)
//                pos_l = js->position[i];
//            if (js->name[i] == right_joint_)
//                pos_r = js->position[i];
//        }
//
//        /* ---------- Joint-Geschwindigkeiten -------------------------- */
//        double wl = 0.0, wr = 0.0;
//        if (pos_init_)
//        {
//            wl = (pos_l - last_pos_l_) / dt;
//            wr = (pos_r - last_pos_r_) / dt;
//        }
//        last_pos_l_ = pos_l;
//        last_pos_r_ = pos_r;
//        pos_init_ = true;
//
//        /* ---------- Vorwärts-Kinematik ------------------------------- */
//        double v_body = 0.5 * wheel_radius_ * (wr + wl);
//        double w_cmd = (wheel_radius_ / wheel_base_) * (wr - wl);
//        input_ << v_body, w_cmd;
//
//        /* ---------- IMU-Yaw ----------------------------------------- */
//        tf2::Quaternion q;
//        tf2::fromMsg(imu->orientation, q);
//        double roll, pitch, yaw;
//        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
//        
//
//        /* ---------- Welt-v ------------------------------------------ */
//        double vx_w = v_body * std::cos(yaw);
//        double vy_w = v_body * std::sin(yaw);
//
//        /* ---------- DR-Pose ----------------------------------------- */
//        x_fk_ += vx_w * dt;
//        y_fk_ += vy_w * dt;
//
//        /* ---------- Mess­vektoren ----------------------------------- */
//        z_kf_ << x_fk_, y_fk_, yaw, vx_w, vy_w, w_cmd; // KF 6-D
//        z_ekf_ = yaw, ;                                // EKF 6-D identisch
//
//        /* ---------- KF-Update -------------------------------------- */
//        auto [kx, kP] = kf_.step(kf_state_, kf_.cov(), input_, z_kf_, dt);
//        kf_state_ = kx;
//
//        /* ---------- EKF-Update ------------------------------------- */
//        auto [ex, eP] = ekf_.step(ekf_state_, ekf_.cov(), input_, z_ekf_, dt);
//        ekf_state_ = ex;
//
//        publish(pub_kf_, kf_state_, kP, odom->header.stamp);
//        publish(pub_ekf_, ekf_state_, eP, odom->header.stamp);
//    }
//
//    /* -------------------------- Publish -------------------------------------- */
//    // ---------------------- Publish helper ----------------------------
//    void publish(ros::Publisher &pub,
//                 const Eigen::VectorXd &x,
//                 const Eigen::MatrixXd &P,
//                 const ros::Time &stamp)
//    {
//        geometry_msgs::PoseWithCovarianceStamped msg;
//        msg.header.stamp = stamp;
//        msg.header.frame_id = "odom";
//
//        msg.pose.pose.position.x = x(0) + x_offset_;
//        msg.pose.pose.position.y = x(1) + y_offset_;
//
//        tf2::Quaternion q_out;
//        q_out.setRPY(0, 0, x(2));
//        msg.pose.pose.orientation = tf2::toMsg(q_out);
//
//        /* -------- nur Pose-Kovarianz kopieren ---------------------- */
//        // Ppose =  [ x  y  θ ]
//        //          [ v  v  ω ]  <-- NICHT in Pose!
//        /* ---------- Pose-Kovarianz kopieren -------------------------------- */
//        double min_var = 1e-6; // ~1 mm² bzw. 0.002°²
//
//        /* ---------- Position-Kovarianz ---------------------------- */
//        msg.pose.covariance.fill(0.0);
//
//        msg.pose.covariance[0] = std::max(P(0, 0), min_var); // σ²x
//        msg.pose.covariance[1] = P(0, 1);                    // cov(x,y)
//        msg.pose.covariance[6] = P(1, 0);
//        msg.pose.covariance[7] = std::max(P(1, 1), min_var); // σ²y
//
//        /* Z-Achse klein halten -> flacher Puck, nicht Stab           */
//        msg.pose.covariance[14] = min_var; // σ²z  ~ 0
//
//        /* ---------- Orientierung-Kovarianz ------------------------ */
//        msg.pose.covariance[21] = 1e3;                        // roll
//        msg.pose.covariance[28] = 1e3;                        // pitch
//        msg.pose.covariance[35] = std::max(P(2, 2), min_var); // yaw (rotZ)
//
//        pub.publish(msg);
//    }
//
//    /* --------------------------- Members ------------------------------------- */
//    KalmanFilter kf_;
//    Eigen::VectorXd kf_state_;
//    ExtendedKalmanFilter ekf_;
//    Eigen::VectorXd ekf_state_;
//    Eigen::VectorXd input_, z_kf_, z_ekf_;
//
//    double wheel_radius_, wheel_base_;
//    std::string left_joint_, right_joint_;
//    double x_offset_, y_offset_;
//
//    double x_fk_, y_fk_;
//    double last_pos_l_, last_pos_r_;
//    bool pos_init_;
//
//    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
//    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
//    message_filters::Subscriber<sensor_msgs::JointState> joint_sub_;
//    std::shared_ptr<Sync> sync_;
//    ros::Publisher pub_kf_, pub_ekf_;
//    ros::Time last_stamp_;
//};
//
///* ======================================================================== */
//int main(int argc, char **argv)
//{
//    ros::init(argc, argv, "filter_node");
//    ros::NodeHandle nh("~");
//    FilterNode node(nh);
//    ros::spin();
//    return 0;
//}
//