#!/usr/bin/env bash
# ----------------[ USER-PARAMETER ]-----------------
BAG_NAME="run_$(date +%Y%m%d_%H%M%S).bag"
# record_and_plot.sh
GT_TOPIC="/odom"          # Referenz-Topic für Pose und Velocity
KF_POSE_TOPIC="/kf_prediction"
EKF_POSE_TOPIC="/ekf_prediction"
KF_VEL_TOPIC="/kf_velocity_prediction" # NEU
EKF_VEL_TOPIC="/ekf_velocity_prediction" # NEU
IMG_OUT="plots/${BAG_NAME%.bag}.png"
PY_SCRIPT="$(dirname "$0")/plot_trajectories.py"
# ---------------------------------------------------

mkdir -p plots          # Zielordner für PNGs

echo "▶  Starte rosbag-Aufnahme →  $BAG_NAME"
rosbag record  \
    $GT_TOPIC  \
    $KF_POSE_TOPIC  \
    $EKF_POSE_TOPIC \
    $KF_VEL_TOPIC \
    $EKF_VEL_TOPIC \
    -O "$BAG_NAME"

echo "▶  Aufnahme beendet – erzeuge Plot …"
python "$PY_SCRIPT"  "$BAG_NAME"  \
       --gt_pose  "$GT_TOPIC" \
       --kf_pose  "$KF_POSE_TOPIC" \
       --ekf_pose "$EKF_POSE_TOPIC" \
       --gt_vel   "$GT_TOPIC" \
       --kf_vel   "$KF_VEL_TOPIC" \
       --ekf_vel  "$EKF_VEL_TOPIC" \
       --out "$IMG_OUT"

echo "✅  Fertig!  Plot gespeichert unter $IMG_OUT"
