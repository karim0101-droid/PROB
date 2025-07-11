#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Ground-Truth, Linear KF, EKF und PF from a ROS Bag.
Requires:  Python ≥3.8,  matplotlib,  pandas,  rospy / rosbag
"""
import argparse
import os
import sys
import rosbag
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tf.transformations import euler_from_quaternion # For extracting yaw from quaternion

def topic_to_pose_df(bag, topic):
    """
    Returns DataFrame with time, x, y, yaw for the given Pose topic.
    """
    records = []
    for t, msg, _ in bag.read_messages(topics=[topic]):
        sec = msg.header.stamp.to_sec()
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        records.append((sec, x, y, yaw))
    if not records:
        print(f"WARNING: Topic {topic} not found in bag or empty.")
        return pd.DataFrame(columns=['t', 'x', 'y', 'yaw'])
    df = pd.DataFrame(records, columns=['t', 'x', 'y', 'yaw'])
    return df

def topic_to_velocity_df(bag, topic):
    """
    Returns DataFrame with time, vx, vy, omega for the given Odometry topic (e.g., /odom)
    or a custom Velocity topic.
    
    IMPORTANT: This function assumes that 'vx' refers to the forward linear velocity
    (typically geometry_msgs/Twist.linear.x) and 'vy' refers to the lateral linear velocity
    (typically geometry_msgs/Twist.linear.y). If your robot's coordinate system or
    odometry publisher has a different convention (e.g., y is forward), you might
    need to adjust the assignment of msg.twist.twist.linear.x and .y.
    """
    records = []
    for t, msg, _ in bag.read_messages(topics=[topic]):
        sec   = msg.header.stamp.to_sec()
        # ASSUMPTION: linear.x is forward (vx), linear.y is lateral (vy)
        # If your robot defines linear.y as forward and linear.x as lateral,
        # UNCOMMENT and use these swapped assignments instead:
        # vx    = msg.twist.twist.linear.y
        # vy    = msg.twist.twist.linear.x
        vx    = msg.twist.twist.linear.x
        vy    = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z
        records.append((sec, vx, vy, omega))
    if not records:
        print(f"WARNING: Topic {topic} not found in bag or empty.")
        return pd.DataFrame(columns=['t', 'vx', 'vy', 'omega'])
    df = pd.DataFrame(records, columns=['t', 'vx', 'vy', 'omega'])
    return df

def calculate_rmse_position(df_estimated, df_ground_truth):
    """
    Calculates the RMSE of position (x, y) between two DataFrames.
    """
    merged_df = pd.merge_asof(
        df_ground_truth.sort_values('t'),
        df_estimated.sort_values('t'),
        on='t',
        direction='nearest',
        tolerance=pd.Timedelta('10ms'),
        suffixes=('_gt', '_est')
    )
    merged_df.dropna(inplace=True)

    if merged_df.empty:
        print("WARNING: No matching timestamps found for position RMSE.")
        return np.nan

    error_x_sq = (merged_df['x_gt'] - merged_df['x_est'])**2
    error_y_sq = (merged_df['y_gt'] - merged_df['y_est'])**2
    rmse = np.sqrt(np.mean(error_x_sq + error_y_sq))
    return rmse

def calculate_rmse_velocity_linear(df_estimated, df_ground_truth):
    """
    Calculates the RMSE of linear velocity (vx, vy) between two DataFrames.
    """
    merged_df = pd.merge_asof(
        df_ground_truth.sort_values('t'),
        df_estimated.sort_values('t'),
        on='t',
        direction='nearest',
        tolerance=pd.Timedelta('10ms'),
        suffixes=('_gt', '_est')
    )
    merged_df.dropna(inplace=True)

    if merged_df.empty:
        print("WARNING: No matching timestamps found for linear velocity RMSE.")
        return np.nan

    error_vx_sq = (merged_df['vx_gt'] - merged_df['vx_est'])**2
    error_vy_sq = (merged_df['vy_gt'] - merged_df['vy_est'])**2
    rmse = np.sqrt(np.mean(error_vx_sq + error_vy_sq))
    return rmse

def calculate_rmse_velocity_angular(df_estimated, df_ground_truth):
    """
    Calculates the RMSE of angular velocity (omega) between two DataFrames.
    """
    merged_df = pd.merge_asof(
        df_ground_truth.sort_values('t'),
        df_estimated.sort_values('t'),
        on='t',
        direction='nearest',
        tolerance=pd.Timedelta('10ms'),
        suffixes=('_gt', '_est')
    )
    merged_df.dropna(inplace=True)

    if merged_df.empty:
        print("WARNING: No matching timestamps found for angular velocity RMSE.")
        return np.nan

    error_omega_sq = (merged_df['omega_gt'] - merged_df['omega_est'])**2
    rmse = np.sqrt(np.mean(error_omega_sq))
    return rmse

def calculate_rmse_yaw(df_estimated, df_ground_truth):
    """
    Calculates the RMSE of yaw angle between two DataFrames.
    Yaw angles are normalized before comparison.
    """
    merged_df = pd.merge_asof(
        df_ground_truth.sort_values('t'),
        df_estimated.sort_values('t'),
        on='t',
        direction='nearest',
        tolerance=pd.Timedelta('10ms'),
        suffixes=('_gt', '_est')
    )
    merged_df.dropna(inplace=True)

    if merged_df.empty:
        print("WARNING: No matching timestamps found for yaw RMSE.")
        return np.nan

    # Normalize angles before calculating difference
    diff_yaw = merged_df['yaw_gt'] - merged_df['yaw_est']
    # Normalize difference to be within -pi to pi
    diff_yaw = np.arctan2(np.sin(diff_yaw), np.cos(diff_yaw))
    
    rmse = np.sqrt(np.mean(diff_yaw**2))
    return rmse

def calculate_end_pose_error(df_estimated, df_ground_truth):
    """
    Calculates the Euclidean distance between the last pose of df_estimated and df_ground_truth.
    """
    if df_estimated.empty or df_ground_truth.empty:
        return np.nan

    # Get the last valid pose from each DataFrame
    last_gt = df_ground_truth.iloc[-1]
    last_est = df_estimated.iloc[-1]

    error = np.sqrt((last_gt['x'] - last_est['x'])**2 + (last_gt['y'] - last_est['y'])**2)
    return error

def main():
    parser = argparse.ArgumentParser(
        description="Plot KF/EKF/PF vs. Ground-Truth from ROS Bag")
    parser.add_argument("bag", help="Path to the .bag file")
    parser.add_argument("--gt_pose",  default="/odom",
                        help="Ground-Truth Pose Topic (default: /odom)")
    parser.add_argument("--kf_pose",  default="/kf_prediction",
                        help="Linear KF Pose Topic")
    parser.add_argument("--ekf_pose", default="/ekf_prediction",
                        help="EKF Pose Topic")
    parser.add_argument("--pf_pose", default="/pf_prediction",
                        help="Particle Filter Pose Topic")
    parser.add_argument("--gt_vel",   default="/odom",
                        help="Ground-Truth Velocity Topic (default: /odom)")
    parser.add_argument("--kf_vel",   default="/kf_velocity_prediction",
                        help="Linear KF Velocity Topic")
    parser.add_argument("--ekf_vel",  default="/ekf_velocity_prediction",
                        help="EKF Velocity Topic")
    parser.add_argument("--pf_vel", default="/pf_velocity_prediction",
                        help="Particle Filter Velocity Topic")
    parser.add_argument("--out_prefix", default="plots/filter_comparison",
                        help="Prefix for output images (PNG)")
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        sys.exit(f"Bag file {args.bag} does not exist.")

    print("Opening Bag file…")
    with rosbag.Bag(args.bag) as bag:
        print("Reading Ground-Truth Poses…")
        df_gt_pose  = topic_to_pose_df(bag, args.gt_pose)
        print("Reading KF Poses…")
        df_kf_pose  = topic_to_pose_df(bag, args.kf_pose)
        print("Reading EKF Poses…")
        df_ekf_pose = topic_to_pose_df(bag, args.ekf_pose)
        print("Reading PF Poses…")
        df_pf_pose  = topic_to_pose_df(bag, args.pf_pose)

        print("Reading Ground-Truth Velocities…")
        df_gt_vel = topic_to_velocity_df(bag, args.gt_vel)
        print("Reading KF Velocities…")
        df_kf_vel = topic_to_velocity_df(bag, args.kf_vel)
        print("Reading EKF Velocities…")
        df_ekf_vel = topic_to_velocity_df(bag, args.ekf_vel)
        print("Reading PF Velocities…")
        df_pf_vel = topic_to_velocity_df(bag, args.pf_vel)


    # Convert time columns to timedelta for merge_asof tolerance
    # Pose DataFrames
    df_gt_pose_td  = df_gt_pose.copy()
    df_kf_pose_td  = df_kf_pose.copy()
    df_ekf_pose_td = df_ekf_pose.copy()
    df_pf_pose_td  = df_pf_pose.copy()
    
    df_gt_pose_td['t'] = pd.to_timedelta(df_gt_pose_td['t'], unit='s')
    df_kf_pose_td['t'] = pd.to_timedelta(df_kf_pose_td['t'], unit='s')
    df_ekf_pose_td['t'] = pd.to_timedelta(df_ekf_pose_td['t'], unit='s')
    df_pf_pose_td['t'] = pd.to_timedelta(df_pf_pose_td['t'], unit='s')

    # Velocity DataFrames
    df_gt_vel_td = df_gt_vel.copy()
    df_kf_vel_td = df_kf_vel.copy()
    df_ekf_vel_td = df_ekf_vel.copy()
    df_pf_vel_td = df_pf_vel.copy()

    df_gt_vel_td['t'] = pd.to_timedelta(df_gt_vel_td['t'], unit='s')
    df_kf_vel_td['t'] = pd.to_timedelta(df_kf_vel_td['t'], unit='s')
    df_ekf_vel_td['t'] = pd.to_timedelta(df_ekf_vel_td['t'], unit='s')
    df_pf_vel_td['t'] = pd.to_timedelta(df_pf_vel_td['t'], unit='s')


    # Calculate RMSE values for positions
    rmse_kf_pose_vs_odom  = calculate_rmse_position(df_kf_pose_td,  df_gt_pose_td)
    rmse_ekf_pose_vs_odom = calculate_rmse_position(df_ekf_pose_td, df_gt_pose_td)
    rmse_pf_pose_vs_odom  = calculate_rmse_position(df_pf_pose_td,  df_gt_pose_td)

    # Calculate RMSE values for velocities
    rmse_kf_vel_linear_vs_odom  = calculate_rmse_velocity_linear(df_kf_vel_td, df_gt_vel_td)
    rmse_ekf_vel_linear_vs_odom = calculate_rmse_velocity_linear(df_ekf_vel_td, df_gt_vel_td)
    rmse_pf_vel_linear_vs_odom  = calculate_rmse_velocity_linear(df_pf_vel_td, df_gt_vel_td)
    
    rmse_kf_vel_angular_vs_odom  = calculate_rmse_velocity_angular(df_kf_vel_td, df_gt_vel_td)
    rmse_ekf_vel_angular_vs_odom = calculate_rmse_velocity_angular(df_ekf_vel_td, df_gt_vel_td)
    rmse_pf_vel_angular_vs_odom  = calculate_rmse_velocity_angular(df_pf_vel_td, df_gt_vel_td)

    # Calculate RMSE values for yaw
    rmse_kf_yaw_vs_odom  = calculate_rmse_yaw(df_kf_pose_td, df_gt_pose_td)
    rmse_ekf_yaw_vs_odom = calculate_rmse_yaw(df_ekf_pose_td, df_gt_pose_td)
    rmse_pf_yaw_vs_odom  = calculate_rmse_yaw(df_pf_pose_td, df_gt_pose_td)

    # Calculate End Pose Errors
    end_error_kf_pose = calculate_end_pose_error(df_kf_pose, df_gt_pose)
    end_error_ekf_pose = calculate_end_pose_error(df_ekf_pose, df_gt_pose)
    end_error_pf_pose = calculate_end_pose_error(df_pf_pose, df_gt_pose)


    # Convert time columns back to float for plotting
    df_gt_pose['t'] = df_gt_pose_td['t'].dt.total_seconds()
    df_kf_pose['t'] = df_kf_pose_td['t'].dt.total_seconds()
    df_ekf_pose['t'] = df_ekf_pose_td['t'].dt.total_seconds()
    df_pf_pose['t'] = df_pf_pose_td['t'].dt.total_seconds()

    df_gt_vel['t'] = df_gt_vel_td['t'].dt.total_seconds()
    df_kf_vel['t'] = df_kf_vel_td['t'].dt.total_seconds()
    df_ekf_vel['t'] = df_ekf_vel_td['t'].dt.total_seconds()
    df_pf_vel['t'] = df_pf_vel_td['t'].dt.total_seconds()


    # --- Plotting Configuration ---
    # Set global font sizes for paper quality (increased by 150%)
    plt.rcParams.update({
        'font.size': 27,          # Increased from 18 to 27 (150%)
        'axes.labelsize': 30,     # Increased from 20 to 30 (150%)
        'xtick.labelsize': 27,    # Increased from 18 to 27 (150%)
        'ytick.labelsize': 27,    # Increased from 18 to 27 (150%)
        'legend.fontsize': 27,    # Increased from 18 to 27 (150%)
        'axes.titlesize': 33      # Increased from 22 to 33 (150%)
    })

    # Define a color palette that is color-blind friendly and highly distinguishable
    colors = {
        'reference': '#333333',  # Dark gray
        'kf': '#0072B2',         # Blue
        'ekf': '#D55E00',        # Orange
        'pf': '#009E73'          # Greenish-blue/Teal
    }
    
    # Line styles
    linestyles = {
        'reference': '--',
        'filter': '-'
    }
    
    # Line widths (increased by 150%)
    linewidths = {
        'reference': 5.25, # Increased from 3.5 to 5.25 (150%)
        'filter': 3.75     # Increased from 2.5 to 3.75 (150%)
    }

    # Text box styling for RMSE/Errors (unified format)
    # This dictionary will be used for all RMSE and End Pose Error text boxes
    bbox_props_annotation = dict(
        boxstyle='round,pad=0.5',  # Increased padding
        fc='white',                # White face color
        ec='#BBBBBB',              # Slightly lighter grey edge color
        alpha=0.9,                 # Slight transparency
        linewidth=1.5              # Thicker border
    )
    
    # Font size for all RMSE and End Pose Error text
    ANNOTATION_FONTSIZE = 24 # Explicitly set to 24 (150% of original 16)

    text_offset = 0.08 # Adjusted vertical offset for larger text


    # Smoothing window for deviations graph (adjust as needed)
    SMOOTHING_WINDOW_SIZE = 10 # For rolling mean, e.g., 10 samples


    # --- Plot 1: Trajectories (Positions) ---
    fig1, ax1 = plt.subplots(figsize=(20, 18)) # Larger figure size (increased by ~150%)
    ax1.plot(df_gt_pose.x, df_gt_pose.y, label="Odom (Reference)", linewidth=linewidths['reference'], color=colors['reference'], linestyle=linestyles['reference'])
    ax1.plot(df_kf_pose.x,  df_kf_pose.y,  label="Linear KF", linewidth=linewidths['filter'], color=colors['kf'], linestyle=linestyles['filter'])
    ax1.plot(df_ekf_pose.x, df_ekf_pose.y, label="Extended KF", linewidth=linewidths['filter'], color=colors['ekf'], linestyle=linestyles['filter'])
    ax1.plot(df_pf_pose.x,  df_pf_pose.y,  label="Particle Filter", linewidth=linewidths['filter'], color=colors['pf'], linestyle=linestyles['filter'])
    
    ax1.axis('equal')
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Trajectory Comparison (Position)")
    ax1.legend(loc='upper right') # Legend inside, top right
    ax1.grid(True)

    # Add RMSE values to the position plot (inside, upper left)
    rmse_text_x_pos = 0.02 # Left side
    rmse_text_y_pos = 0.98 # Top of the plot

    if not np.isnan(rmse_kf_pose_vs_odom):
        ax1.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE (LKF vs. Odom): {rmse_kf_pose_vs_odom:.4f} m',
                 transform=ax1.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        rmse_text_y_pos -= text_offset
    if not np.isnan(rmse_ekf_pose_vs_odom):
        ax1.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE (EKF vs. Odom): {rmse_ekf_pose_vs_odom:.4f} m',
                 transform=ax1.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        rmse_text_y_pos -= text_offset
    if not np.isnan(rmse_pf_pose_vs_odom):
        ax1.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE (PF vs. Odom): {rmse_pf_pose_vs_odom:.4f} m',
                 transform=ax1.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
    
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_trajectories.png", dpi=300) # Keep DPI for paper quality
    print(f"Plot saved as {args.out_prefix}_trajectories.png")
    plt.close(fig1) 


    # --- Plot 2: Linear Velocities (Vx) ---
    fig2, ax2 = plt.subplots(figsize=(20, 15)) # Larger figure size (increased by ~150%)
    # Check if a swap is needed. If the plotted Vx looks like lateral motion, swap.
    # The default is linear.x for vx.
    # To swap in topic_to_velocity_df:
    # vx    = msg.twist.twist.linear.y
    # vy    = msg.twist.twist.linear.x
    ax2.plot(df_gt_vel.t, df_gt_vel.vx, label="Odom $v_x$ (Reference)", linewidth=linewidths['reference'], color=colors['reference'], linestyle=linestyles['reference'])
    ax2.plot(df_kf_vel.t, df_kf_vel.vx, label="Linear KF $v_x$", linewidth=linewidths['filter'], color=colors['kf'], linestyle=linestyles['filter'])
    ax2.plot(df_ekf_vel.t, df_ekf_vel.vx, label="Extended KF $v_x$", linewidth=linewidths['filter'], color=colors['ekf'], linestyle=linestyles['filter'])
    ax2.plot(df_pf_vel.t, df_pf_vel.vx, label="Particle Filter $v_x$", linewidth=linewidths['filter'], color=colors['pf'], linestyle=linestyles['filter'])
    
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Linear Velocity [m/s]")
    ax2.set_title("Linear Velocities ($v_x$)")
    ax2.legend(loc='upper right') 
    ax2.grid(True)

    # Add RMSE values to the linear velocity plot
    rmse_text_x_pos = 0.02 
    rmse_text_y_pos = 0.98 
    if not np.isnan(rmse_kf_vel_linear_vs_odom):
        ax2.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Lin. Vel (LKF vs. Odom): {rmse_kf_vel_linear_vs_odom:.4f} m/s',
                 transform=ax2.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        rmse_text_y_pos -= text_offset
    if not np.isnan(rmse_ekf_vel_linear_vs_odom):
        ax2.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Lin. Vel (EKF vs. Odom): {rmse_ekf_vel_linear_vs_odom:.4f} m/s',
                 transform=ax2.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        rmse_text_y_pos -= text_offset
    if not np.isnan(rmse_pf_vel_linear_vs_odom):
        ax2.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Lin. Vel (PF vs. Odom): {rmse_pf_vel_linear_vs_odom:.4f} m/s',
                 transform=ax2.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_linear_velocities_vx.png", dpi=300)
    print(f"Plot saved as {args.out_prefix}_linear_velocities_vx.png")
    plt.close(fig2)

    # --- Plot 3: Linear Velocities (Vy) ---
    fig3, ax3 = plt.subplots(figsize=(20, 15)) # Larger figure size (increased by ~150%)
    # Check if a swap is needed. If the plotted Vy looks like forward motion, swap.
    # The default is linear.y for vy.
    # To swap in topic_to_velocity_df:
    # vx    = msg.twist.twist.linear.y
    # vy    = msg.twist.twist.linear.x
    ax3.plot(df_gt_vel.t, df_gt_vel.vy, label="Odom $v_y$ (Reference)", linewidth=linewidths['reference'], color=colors['reference'], linestyle=linestyles['reference'])
    ax3.plot(df_kf_vel.t, df_kf_vel.vy, label="Linear KF $v_y$", linewidth=linewidths['filter'], color=colors['kf'], linestyle=linestyles['filter'])
    ax3.plot(df_ekf_vel.t, df_ekf_vel.vy, label="Extended KF $v_y$", linewidth=linewidths['filter'], color=colors['ekf'], linestyle=linestyles['filter'])
    ax3.plot(df_pf_vel.t, df_pf_vel.vy, label="Particle Filter $v_y$", linewidth=linewidths['filter'], color=colors['pf'], linestyle=linestyles['filter'])
    
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Linear Velocity [m/s]")
    ax3.set_title("Linear Velocities ($v_y$)")
    ax3.legend(loc='upper right')
    ax3.grid(True)

    # Add RMSE values to the linear velocity y plot
    rmse_text_x_pos = 0.02
    rmse_text_y_pos = 0.98
    if not np.isnan(rmse_kf_vel_linear_vs_odom): # Using the overall linear RMSE here, as component-specific RMSE wasn't distinct.
        ax3.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Lin. Vel $v_y$ (LKF vs. Odom): {rmse_kf_vel_linear_vs_odom:.4f} m/s',
                 transform=ax3.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        rmse_text_y_pos -= text_offset
    if not np.isnan(rmse_ekf_vel_linear_vs_odom):
        ax3.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Lin. Vel $v_y$ (EKF vs. Odom): {rmse_ekf_vel_linear_vs_odom:.4f} m/s',
                 transform=ax3.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        rmse_text_y_pos -= text_offset
    if not np.isnan(rmse_pf_vel_linear_vs_odom):
        ax3.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Lin. Vel $v_y$ (PF vs. Odom): {rmse_pf_vel_linear_vs_odom:.4f} m/s',
                 transform=ax3.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_linear_velocities_vy.png", dpi=300)
    print(f"Plot saved as {args.out_prefix}_linear_velocities_vy.png")
    plt.close(fig3)


    # --- Plot 4: Angular Velocities (Omega) ---
    fig4, ax4 = plt.subplots(figsize=(20, 15)) # Larger figure size (increased by ~150%)
    ax4.plot(df_gt_vel.t, df_gt_vel.omega, label="Odom $\omega$ (Reference)", linewidth=linewidths['reference'], color=colors['reference'], linestyle=linestyles['reference'])
    ax4.plot(df_kf_vel.t, df_kf_vel.omega, label="Linear KF $\omega$", linewidth=linewidths['filter'], color=colors['kf'], linestyle=linestyles['filter'])
    ax4.plot(df_ekf_vel.t, df_ekf_vel.omega, label="Extended KF $\omega$", linewidth=linewidths['filter'], color=colors['ekf'], linestyle=linestyles['filter'])
    ax4.plot(df_pf_vel.t, df_pf_vel.omega, label="Particle Filter $\omega$", linewidth=linewidths['filter'], color=colors['pf'], linestyle=linestyles['filter'])
    
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Angular Velocity [rad/s]")
    ax4.set_title("Angular Velocities ($\omega$)")
    ax4.legend(loc='upper right')
    ax4.grid(True)

    # Add RMSE values to the angular velocity plot
    rmse_text_x_pos = 0.02
    rmse_text_y_pos = 0.98
    if not np.isnan(rmse_kf_vel_angular_vs_odom):
        ax4.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Ang. Vel (LKF vs. Odom): {rmse_kf_vel_angular_vs_odom:.4f} rad/s',
                 transform=ax4.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        rmse_text_y_pos -= text_offset
    if not np.isnan(rmse_ekf_vel_angular_vs_odom):
        ax4.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Ang. Vel (EKF vs. Odom): {rmse_ekf_vel_angular_vs_odom:.4f} rad/s',
                 transform=ax4.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        rmse_text_y_pos -= text_offset
    if not np.isnan(rmse_pf_vel_angular_vs_odom):
        ax4.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Ang. Vel (PF vs. Odom): {rmse_pf_vel_angular_vs_odom:.4f} rad/s',
                 transform=ax4.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)

    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_angular_velocities.png", dpi=300)
    print(f"Plot saved as {args.out_prefix}_angular_velocities.png")
    plt.close(fig4)


    # --- Plot 5: Absolute Position Deviations from Odom ---
    fig5, ax5 = plt.subplots(figsize=(20, 15)) # Larger figure size (increased by ~150%)
    # Merge all filter poses with GT pose to calculate deviations
    merged_kf_deviation = pd.merge_asof(df_gt_pose.sort_values('t'), df_kf_pose.sort_values('t'),
                                        on='t', direction='nearest', tolerance=pd.Timedelta('10ms').total_seconds(),
                                        suffixes=('_gt', '_kf'))
    merged_ekf_deviation = pd.merge_asof(df_gt_pose.sort_values('t'), df_ekf_pose.sort_values('t'),
                                         on='t', direction='nearest', tolerance=pd.Timedelta('10ms').total_seconds(),
                                         suffixes=('_gt', '_ekf'))
    merged_pf_deviation = pd.merge_asof(df_gt_pose.sort_values('t'), df_pf_pose.sort_values('t'),
                                        on='t', direction='nearest', tolerance=pd.Timedelta('10ms').total_seconds(),
                                        suffixes=('_gt', '_pf'))
    
    # Calculate absolute position deviation (Euclidean distance)
    if not merged_kf_deviation.empty:
        merged_kf_deviation['deviation'] = np.sqrt(
            (merged_kf_deviation['x_gt'] - merged_kf_deviation['x_kf'])**2 +
            (merged_kf_deviation['y_gt'] - merged_kf_deviation['y_kf'])**2
        )
        # Apply rolling mean for smoothing
        ax5.plot(merged_kf_deviation['t'], merged_kf_deviation['deviation'].rolling(window=SMOOTHING_WINDOW_SIZE, min_periods=1, center=True).mean(), 
                 label="Linear KF Deviation", linewidth=linewidths['filter'], color=colors['kf'], linestyle=linestyles['filter']) 
    
    if not merged_ekf_deviation.empty:
        merged_ekf_deviation['deviation'] = np.sqrt(
            (merged_ekf_deviation['x_gt'] - merged_ekf_deviation['x_ekf'])**2 +
            (merged_ekf_deviation['y_gt'] - merged_ekf_deviation['y_ekf'])**2
        )
        # Apply rolling mean for smoothing
        ax5.plot(merged_ekf_deviation['t'], merged_ekf_deviation['deviation'].rolling(window=SMOOTHING_WINDOW_SIZE, min_periods=1, center=True).mean(), 
                 label="Extended KF Deviation", linewidth=linewidths['filter'], color=colors['ekf'], linestyle=linestyles['filter'])

    if not merged_pf_deviation.empty:
        merged_pf_deviation['deviation'] = np.sqrt(
            (merged_pf_deviation['x_gt'] - merged_pf_deviation['x_pf'])**2 +
            (merged_pf_deviation['y_gt'] - merged_pf_deviation['y_pf'])**2
        )
        # Apply rolling mean for smoothing
        ax5.plot(merged_pf_deviation['t'], merged_pf_deviation['deviation'].rolling(window=SMOOTHING_WINDOW_SIZE, min_periods=1, center=True).mean(), 
                 label="Particle Filter Deviation", linewidth=linewidths['filter'], color=colors['pf'], linestyle=linestyles['filter'])


    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Euclidean Position Deviation [m]") 
    ax5.set_title("Absolute Position Deviations from Odom (Smoothed)") 
    ax5.legend(loc='upper right') 
    ax5.grid(True)

    # Add End Pose Error to the deviations plot
    end_error_text_x_pos = 0.02
    end_error_text_y_pos = 0.98
    # Use the unified bbox_props_annotation for consistent formatting
    if not np.isnan(end_error_kf_pose):
        ax5.text(end_error_text_x_pos, end_error_text_y_pos, f'End Pose Error (LKF): {end_error_kf_pose:.4f} m',
                 transform=ax5.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=dict(bbox_props_annotation, ec=colors['kf'], color=colors['kf'])) # Override edge/text color to match line
        end_error_text_y_pos -= text_offset
    if not np.isnan(end_error_ekf_pose):
        ax5.text(end_error_text_x_pos, end_error_text_y_pos, f'End Pose Error (EKF): {end_error_ekf_pose:.4f} m',
                 transform=ax5.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=dict(bbox_props_annotation, ec=colors['ekf'], color=colors['ekf'])) # Override edge/text color to match line
        end_error_text_y_pos -= text_offset
    if not np.isnan(end_error_pf_pose):
        ax5.text(end_error_text_x_pos, end_error_text_y_pos, f'End Pose Error (PF): {end_error_pf_pose:.4f} m',
                 transform=ax5.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=dict(bbox_props_annotation, ec=colors['pf'], color=colors['pf'])) # Override edge/text color to match line

    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_deviations.png", dpi=300)
    print(f"Plot saved as {args.out_prefix}_deviations.png")
    plt.close(fig5)

    # --- Plot 6: Yaw Angle over Time ---
    fig6, ax6 = plt.subplots(figsize=(20, 15)) # Larger figure size (increased by ~150%)
    ax6.plot(df_gt_pose.t, df_gt_pose.yaw, label="Odom Yaw (Reference)", linewidth=linewidths['reference'], color=colors['reference'], linestyle=linestyles['reference'])
    ax6.plot(df_kf_pose.t, df_kf_pose.yaw, label="Linear KF Yaw", linewidth=linewidths['filter'], color=colors['kf'], linestyle=linestyles['filter'])
    ax6.plot(df_ekf_pose.t, df_ekf_pose.yaw, label="Extended KF Yaw", linewidth=linewidths['filter'], color=colors['ekf'], linestyle=linestyles['filter'])
    ax6.plot(df_pf_pose.t, df_pf_pose.yaw, label="Particle Filter Yaw", linewidth=linewidths['filter'], color=colors['pf'], linestyle=linestyles['filter'])

    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Yaw Angle [rad]")
    ax6.set_title("Yaw Angle Comparison")
    ax6.legend(loc='upper right')
    ax6.grid(True)

    # Add RMSE values to the yaw plot
    rmse_text_x_pos = 0.02
    rmse_text_y_pos = 0.98
    if not np.isnan(rmse_kf_yaw_vs_odom):
        ax6.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Yaw (LKF vs. Odom): {rmse_kf_yaw_vs_odom:.4f} rad',
                 transform=ax6.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        rmse_text_y_pos -= text_offset
    if not np.isnan(rmse_ekf_yaw_vs_odom):
        ax6.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Yaw (EKF vs. Odom): {rmse_ekf_yaw_vs_odom:.4f} rad',
                 transform=ax6.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
        rmse_text_y_pos -= text_offset
    if not np.isnan(rmse_pf_yaw_vs_odom):
        ax6.text(rmse_text_x_pos, rmse_text_y_pos, f'RMSE Yaw (PF vs. Odom): {rmse_pf_yaw_vs_odom:.4f} rad',
                 transform=ax6.transAxes, fontsize=ANNOTATION_FONTSIZE, verticalalignment='top',
                 bbox=bbox_props_annotation)
    
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_yaw_angles.png", dpi=300)
    print(f"Plot saved as {args.out_prefix}_yaw_angles.png")
    plt.close(fig6)


if __name__ == "__main__":
    main()