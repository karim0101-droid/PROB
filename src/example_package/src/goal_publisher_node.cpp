#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <vector>
#include <string>
#include <cmath> // For M_PI

struct RobotGoal
{
    double x;
    double y;
    double yaw_deg;
};

class GoalPublisher
{
public:
    typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

    GoalPublisher(ros::NodeHandle &nh) : nh_(nh),
                                         current_goal_idx_(-1),
                                         ac_("move_base", true) // Create the action client, true = spin a thread
    {
        goals_.push_back({3.5, 0.5, 90.0});
        goals_.push_back({4.0, 2.5, 180.0});
        goals_.push_back({3.0, 3.5, 255.0});
        goals_.push_back({2.5, 1.5, 270.0});

        ROS_INFO("GoalPublisher node started.");

        ROS_INFO("Waiting for the move_base action server to start...");
        ac_.waitForServer();
        ROS_INFO("move_base action server started.");
    }

    void startPublishingSequence()
    {
        // This is called once after initial 10-second delay in main()
        publishNextGoal();
    }

private:
    void publishNextGoal()
    {
        if (current_goal_idx_ + 1 < goals_.size())
        {
            current_goal_idx_++;
            const RobotGoal &current_robot_goal = goals_[current_goal_idx_];

            move_base_msgs::MoveBaseGoal goal;
            goal.target_pose.header.stamp = ros::Time::now();
            goal.target_pose.header.frame_id = "map";

            goal.target_pose.pose.position.x = current_robot_goal.x;
            goal.target_pose.pose.position.y = current_robot_goal.y;
            goal.target_pose.pose.position.z = 0.0;

            tf2::Quaternion q;
            double yaw_rad = current_robot_goal.yaw_deg * M_PI / 180.0;
            q.setRPY(0, 0, yaw_rad);
            goal.target_pose.pose.orientation = tf2::toMsg(q);

            ROS_INFO("Sending Goal %zu: x=%.2f, y=%.2f, yaw=%.1f deg",
                     current_goal_idx_ + 1, current_robot_goal.x, current_robot_goal.y, current_robot_goal.yaw_deg);

            ac_.sendGoal(goal,
                         boost::bind(&GoalPublisher::doneCb, this, _1, _2),
                         boost::bind(&GoalPublisher::activeCb, this),
                         boost::bind(&GoalPublisher::feedbackCb, this, _1));

            ROS_INFO("Waiting for goal %zu to complete...", current_goal_idx_ + 1);
        }
        else
        {
            ROS_INFO("All goals in sequence completed. Shutting down GoalPublisher.");
            ac_.cancelAllGoals();
            ros::shutdown();
        }
    }

    // Called once when the goal completes
    void doneCb(const actionlib::SimpleClientGoalState &state,
                const move_base_msgs::MoveBaseResultConstPtr &result)
    {
        ROS_INFO("Goal %zu finished with status: %s", current_goal_idx_ + 1, state.toString().c_str());

        // Create a one-shot timer to publish the next goal after a short delay
        // This allows the current doneCb to return and actionlib to process state changes.
        next_goal_timer_ = nh_.createTimer(ros::Duration(1.0), // 1 second delay
                                           &GoalPublisher::nextGoalTimerCallback,
                                           this,
                                           true); // true means one-shot timer
    }

    // Callback for the one-shot timer
    void nextGoalTimerCallback(const ros::TimerEvent &event)
    {
        publishNextGoal(); // Now, publish the next goal
    }

    // Called once when the goal becomes active
    void activeCb()
    {
        ROS_INFO("Goal %zu just went active.", current_goal_idx_ + 1);
    }

    // Called every time feedback is received for the goal
    void feedbackCb(const move_base_msgs::MoveBaseFeedbackConstPtr &feedback)
    {
        // ROS_INFO_THROTTLE(10, "Received feedback for goal %zu. Current pose: x=%.2f, y=%.2f, yaw_rad=%.2f",
        //          current_goal_idx_ + 1, feedback->base_position.pose.position.x,
        //          feedback->base_position.pose.position.y,
        //          tf2::getYaw(feedback->base_position.pose.orientation));
    }

    ros::NodeHandle nh_;
    MoveBaseClient ac_;
    std::vector<RobotGoal> goals_;
    ssize_t current_goal_idx_;
    ros::Timer next_goal_timer_; // New: Timer for scheduling next goal
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "goal_publisher_node");
    ros::NodeHandle nh;

    GoalPublisher publisher(nh);

    // Initial wait for 10 seconds before publishing the first goal
    ROS_INFO("Waiting 10 seconds before publishing the first goal...");
    ros::Duration(10.0).sleep();

    // Start publishing the goals
    publisher.startPublishingSequence();

    ros::spin();

    return 0;
}