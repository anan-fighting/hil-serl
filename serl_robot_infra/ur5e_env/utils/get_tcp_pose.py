#!/usr/bin/env python3
"""
Utility script: read the current UR5e TCP pose and print it in rotvec
(UR native / axis-angle) format — ready to paste directly into config.py.

Usage:
    python get_tcp_pose.py --robot_ip 192.168.1.103

Use FreeDrive mode (press the button on the teach pendant or call
  rtde_control.teachMode() / freedriveMode()) to move the robot to
the desired configuration, then run this script to collect the pose.

Note: TARGET_POSE / RESET_POSE / ABS_POSE_LIMIT_* are stored as
  [x, y, z, rx, ry, rz]  where (rx, ry, rz) is the rotation vector
  (axis * angle, i.e. the UR/RTDE native representation).
"""

import argparse
import numpy as np
import rtde_receive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="192.168.1.103", help="UR5e IP address")
    args = parser.parse_args()

    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)

    tcp_rv = np.array(rtde_r.getActualTCPPose())   # [x,y,z,rx,ry,rz] rotvec
    joints = np.array(rtde_r.getActualQ())          # 6 joint angles (radians)

    print("\n=== UR5e Current TCP Pose ===")
    print(f"  Rotvec (UR native) [x,y,z,rx,ry,rz]: {np.round(tcp_rv, 6).tolist()}")
    print(f"  Joint angles (rad)                  : {np.round(joints, 6).tolist()}")
    print()
    print("Copy into your config.py as:")
    print(f"  TARGET_POSE = np.array({np.round(tcp_rv, 6).tolist()})")
    print()


if __name__ == "__main__":
    main()
