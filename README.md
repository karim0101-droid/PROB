# 🧭 Filter Evaluation in TurtleBot3 Maze
Package: example_package
Workspace: testproject

This project implements and evaluates three state estimation filters for a mobile robot navigating a maze-like environment in Gazebo. Each filter estimates the robot's pose and velocity based on odometry and sensor data:

## 📚 Filter Overview
### 🔹 Linear Kalman Filter (KF)
Model: Assumes linear system dynamics and Gaussian noise.

Strengths: Simple, fast, and effective for systems that are approximately linear.

Limitations: Inaccurate when facing nonlinear motion or non-Gaussian noise.

### 🔸 Extended Kalman Filter (EKF)
Model: Handles nonlinear dynamics by linearizing around the current estimate.

Strengths: Suitable for real-world robots with nonlinear motion and sensors.

Limitations: Accuracy depends on how close the system behaves linearly around the current estimate.

### 🟢 Particle Filter (PF)
Model: Represents the belief state using a set of weighted samples (particles).

Strengths: Works with non-Gaussian, highly nonlinear systems. Robust to ambiguous observations.

Limitations: Computationally expensive, especially with many particles.

### 🚀 Launch the Simulation
```
cd ~/testproject
catkin_make
source devel/setup.bash
roslaunch example_package simulation.launch
````
This will:
✅ Start Gazebo with a maze
✅ Spawn a TurtleBot3 Burger
✅ Run AMCL + your custom filters
✅ Start goal_publisher_node
✅ Open RViz layout (from my_cool_project/config/)


#### 📈 Run: Bag + Plot
After launching the simulation, run:
```
bash
Kopieren
Bearbeiten
cd ~/testproject/src/example_package/scripts
./record_and_plot.sh
```
This script will:
Wait a few seconds
Record ROS topics for ~85 seconds
Generate trajectory, velocity, and yaw plots
Output goes to:
📁 example_package/plots/


##### 📊 Plots Generated
Each filter is compared to /odom ground-truth:

📍 Trajectory (x-y) + RMSE

📉 Linear velocities (vx, vy)

🔁 Angular velocity (ω)

🔄 Yaw over time

📈 Deviation over time

🎯 End-pose error

All plots are exported as .png (high DPI).


