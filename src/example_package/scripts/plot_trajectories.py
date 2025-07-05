#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Ground-Truth, Linear KF und EKF aus einer ROS-Bag.
Benötigt:  Python ≥3.8,  matplotlib,  pandas,  rospy / rosbag
"""
import argparse
import os
import sys
import rosbag
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def topic_to_pose_df(bag, topic):
    """
    Liefert DataFrame mit time, x, y für das gegebene Pose-Topic.
    """
    records = []
    for t, msg, _ in bag.read_messages(topics=[topic]):
        sec = msg.header.stamp.to_sec()
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        records.append((sec, x, y))
    if not records:
        print(f"WARNUNG: Topic {topic} nicht in Bag gefunden oder leer.")
        return pd.DataFrame(columns=['t', 'x', 'y'])
    df = pd.DataFrame(records, columns=['t', 'x', 'y'])
    return df

def topic_to_velocity_df(bag, topic):
    """
    Liefert DataFrame mit time, vx, vy, omega für das gegebene Odometry-Topic.
    """
    records = []
    for t, msg, _ in bag.read_messages(topics=[topic]):
        sec   = msg.header.stamp.to_sec()
        vx    = msg.twist.twist.linear.x
        vy    = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z
        records.append((sec, vx, vy, omega))
    if not records:
        print(f"WARNUNG: Topic {topic} nicht in Bag gefunden oder leer.")
        return pd.DataFrame(columns=['t', 'vx', 'vy', 'omega'])
    df = pd.DataFrame(records, columns=['t', 'vx', 'vy', 'omega'])
    return df

def calculate_rmse_position(df_estimated, df_ground_truth):
    """
    Berechnet den RMSE der Position (x, y) zwischen zwei DataFrames.
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
        print("WARNUNG: Keine übereinstimmenden Zeitstempel für Positions-RMSE gefunden.")
        return np.nan

    error_x_sq = (merged_df['x_gt'] - merged_df['x_est'])**2
    error_y_sq = (merged_df['y_gt'] - merged_df['y_est'])**2
    rmse = np.sqrt(np.mean(error_x_sq + error_y_sq))
    return rmse

def calculate_rmse_velocity_linear(df_estimated, df_ground_truth):
    """
    Berechnet den RMSE der linearen Geschwindigkeit (vx, vy) zwischen zwei DataFrames.
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
        print("WARNUNG: Keine übereinstimmenden Zeitstempel für lineare Geschwindigkeits-RMSE gefunden.")
        return np.nan

    error_vx_sq = (merged_df['vx_gt'] - merged_df['vx_est'])**2
    error_vy_sq = (merged_df['vy_gt'] - merged_df['vy_est'])**2
    rmse = np.sqrt(np.mean(error_vx_sq + error_vy_sq))
    return rmse

def calculate_rmse_velocity_angular(df_estimated, df_ground_truth):
    """
    Berechnet den RMSE der Winkelgeschwindigkeit (omega) zwischen zwei DataFrames.
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
        print("WARNUNG: Keine übereinstimmenden Zeitstempel für Winkelgeschwindigkeits-RMSE gefunden.")
        return np.nan

    error_omega_sq = (merged_df['omega_gt'] - merged_df['omega_est'])**2
    rmse = np.sqrt(np.mean(error_omega_sq))
    return rmse

def main():
    parser = argparse.ArgumentParser(
        description="Plot KF/EKF vs. Ground-Truth aus ROS-Bag")
    parser.add_argument("bag", help="Pfad zur .bag-Datei")
    parser.add_argument("--gt_pose",  default="/odom",
                        help="Ground-Truth Pose Topic (default: /odom)")
    parser.add_argument("--kf_pose",  default="/kf_prediction",
                        help="Linear KF Pose Topic")
    parser.add_argument("--ekf_pose", default="/ekf_prediction",
                        help="EKF Pose Topic")
    parser.add_argument("--gt_vel",   default="/odom", # Odom also has velocities
                        help="Ground-Truth Velocity Topic (default: /odom)")
    parser.add_argument("--kf_vel",   default="/kf_velocity_prediction", # NEW topic for KF velocities
                        help="Linear KF Velocity Topic")
    parser.add_argument("--ekf_vel",  default="/ekf_velocity_prediction", # NEW topic for EKF velocities
                        help="EKF Velocity Topic")
    parser.add_argument("--out", default="trajectories.png",
                        help="Ausgabebild (PNG)")
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        sys.exit(f"Bag-Datei {args.bag} existiert nicht.")

    print("Öffne Bag …")
    with rosbag.Bag(args.bag) as bag:
        print("Lese Ground-Truth Posen …")
        df_gt_pose  = topic_to_pose_df(bag, args.gt_pose)
        print("Lese KF Posen …")
        df_kf_pose  = topic_to_pose_df(bag, args.kf_pose)
        print("Lese EKF Posen …")
        df_ekf_pose = topic_to_pose_df(bag, args.ekf_pose)

        print("Lese Ground-Truth Geschwindigkeiten …")
        df_gt_vel = topic_to_velocity_df(bag, args.gt_vel)
        print("Lese KF Geschwindigkeiten …")
        df_kf_vel = topic_to_velocity_df(bag, args.kf_vel)
        print("Lese EKF Geschwindigkeiten …")
        df_ekf_vel = topic_to_velocity_df(bag, args.ekf_vel)


    # Convert time columns to timedelta for merge_asof tolerance
    # Pose DataFrames
    df_gt_pose['t'] = pd.to_timedelta(df_gt_pose['t'], unit='s')
    df_kf_pose['t'] = pd.to_timedelta(df_kf_pose['t'], unit='s')
    df_ekf_pose['t'] = pd.to_timedelta(df_ekf_pose['t'], unit='s')
    # Velocity DataFrames
    df_gt_vel['t'] = pd.to_timedelta(df_gt_vel['t'], unit='s')
    df_kf_vel['t'] = pd.to_timedelta(df_kf_vel['t'], unit='s')
    df_ekf_vel['t'] = pd.to_timedelta(df_ekf_vel['t'], unit='s')


    # Calculate RMSE values for positions
    rmse_kf_pose_vs_odom = calculate_rmse_position(df_kf_pose, df_gt_pose)
    rmse_ekf_pose_vs_odom = calculate_rmse_position(df_ekf_pose, df_gt_pose)

    # Calculate RMSE values for velocities
    rmse_kf_vel_linear_vs_odom = calculate_rmse_velocity_linear(df_kf_vel, df_gt_vel)
    rmse_ekf_vel_linear_vs_odom = calculate_rmse_velocity_linear(df_ekf_vel, df_gt_vel)
    rmse_kf_vel_angular_vs_odom = calculate_rmse_velocity_angular(df_kf_vel, df_gt_vel)
    rmse_ekf_vel_angular_vs_odom = calculate_rmse_velocity_angular(df_ekf_vel, df_gt_vel)


    # Convert time columns back to float for plotting
    df_gt_pose['t'] = df_gt_pose['t'].dt.total_seconds()
    df_kf_pose['t'] = df_kf_pose['t'].dt.total_seconds()
    df_ekf_pose['t'] = df_ekf_pose['t'].dt.total_seconds()
    df_gt_vel['t'] = df_gt_vel['t'].dt.total_seconds()
    df_kf_vel['t'] = df_kf_vel['t'].dt.total_seconds()
    df_ekf_vel['t'] = df_ekf_vel['t'].dt.total_seconds()

    # ---------- Plot ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12)) # Two subplots: one for pose, one for velocities

    # Plot 1: Trajectories (Positions)
    ax1.plot(df_gt_pose.x, df_gt_pose.y, label="Odom (Referenz)", linewidth=2, color='black', linestyle='--')
    ax1.plot(df_kf_pose.x,  df_kf_pose.y,  label="Linear KF", linewidth=1.5, color='blue')
    ax1.plot(df_ekf_pose.x, df_ekf_pose.y, label="Extended KF", linewidth=1.5, color='red')
    ax1.axis('equal')
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Trajektorien-Vergleich (Position)")
    ax1.legend()
    ax1.grid(True)

    # Add RMSE values to the position plot
    if not np.isnan(rmse_kf_pose_vs_odom):
        ax1.text(0.02, 0.98, f'RMSE (Linear KF vs. Odom): {rmse_kf_pose_vs_odom:.4f} m',
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='blue', alpha=0.1))
    if not np.isnan(rmse_ekf_pose_vs_odom):
        ax1.text(0.02, 0.93, f'RMSE (Extended KF vs. Odom): {rmse_ekf_pose_vs_odom:.4f} m',
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='red', alpha=0.1))

    # Plot 2: Velocities
    # Linear Velocity (vx)
    ax2.plot(df_gt_vel.t, df_gt_vel.vx, label="Odom $v_x$ (Referenz)", linewidth=2, color='black', linestyle='--')
    ax2.plot(df_kf_vel.t, df_kf_vel.vx, label="Linear KF $v_x$", linewidth=1.5, color='blue')
    ax2.plot(df_ekf_vel.t, df_ekf_vel.vx, label="Extended KF $v_x$", linewidth=1.5, color='red')
    
    # Angular Velocity (omega)
    ax2.plot(df_gt_vel.t, df_gt_vel.omega, label="Odom $\omega$ (Referenz)", linewidth=2, color='gray', linestyle=':')
    ax2.plot(df_kf_vel.t, df_kf_vel.omega, label="Linear KF $\omega$", linewidth=1.5, color='darkblue', linestyle='-')
    ax2.plot(df_ekf_vel.t, df_ekf_vel.omega, label="Extended KF $\omega$", linewidth=1.5, color='darkred', linestyle='-')

    ax2.set_xlabel("Zeit [s]")
    ax2.set_ylabel("Geschwindigkeit [m/s, rad/s]")
    ax2.set_title("Geschwindigkeits-Vergleich")
    ax2.legend(loc='upper right', fontsize='small') # Adjust legend location/size
    ax2.grid(True)

    # Add RMSE values to the velocity plot
    if not np.isnan(rmse_kf_vel_linear_vs_odom):
        ax2.text(0.02, 0.98, f'RMSE Linear Vel (LKF vs. Odom): {rmse_kf_vel_linear_vs_odom:.4f} m/s',
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='blue', alpha=0.1))
    if not np.isnan(rmse_ekf_vel_linear_vs_odom):
        ax2.text(0.02, 0.93, f'RMSE Linear Vel (EKF vs. Odom): {rmse_ekf_vel_linear_vs_odom:.4f} m/s',
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='red', alpha=0.1))
    if not np.isnan(rmse_kf_vel_angular_vs_odom):
        ax2.text(0.02, 0.88, f'RMSE Angular Vel (LKF vs. Odom): {rmse_kf_vel_angular_vs_odom:.4f} rad/s',
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='darkblue', alpha=0.1))
    if not np.isnan(rmse_ekf_vel_angular_vs_odom):
        ax2.text(0.02, 0.83, f'RMSE Angular Vel (EKF vs. Odom): {rmse_ekf_vel_angular_vs_odom:.4f} rad/s',
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='darkred', alpha=0.1))

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Plot gespeichert als {args.out}")

if __name__ == "__main__":
    main()

