# In your testproject/src/example_package/CMakeLists.txt

cmake_minimum_required(VERSION 3.0.2)
project(example_package)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf
  tf2_ros
  tf2_geometry_msgs
  message_filters
  angles
  # --- NEU ---
  actionlib
  move_base_msgs
)

add_compile_options(-std=c++17)

catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES example_package
  # CATKIN_DEPENDS roscpp rospy std_msgs
  # DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package contains header files in include/
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add source files for each filter class and the main node
add_library(linear_kf src/linear_kf.cpp)
add_library(extended_kf src/extended_kf.cpp)
add_library(particle_filter src/particle_filter.cpp)

# Define your main executable, including all source files and libraries
add_executable(filter_node src/filter_node.cpp)
target_link_libraries(filter_node
  linear_kf
  extended_kf
  particle_filter
  ${catkin_LIBRARIES}
)

# --- NEW: Goal Publisher Node ---
add_executable(goal_publisher_node src/goal_publisher_node.cpp)
target_link_libraries(goal_publisher_node
  ${catkin_LIBRARIES}
)
