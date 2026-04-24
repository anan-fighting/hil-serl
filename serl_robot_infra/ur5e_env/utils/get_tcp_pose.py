#!/usr/bin/env python3
"""
Utility script: read the current UR5e TCP pose and print it in both
rotvec (UR native) and euler-XYZ formats.

Usage:
    python get_tcp_pose.py --robot_ip 192.168.1.100

Use FreeDrive mode (press the button on the teach pendant or call
  rtde_control.teachMode() / freedriveMode()) to move the robot to
the desired configuration, then run this script to collect the pose.
"""

import argparse
import numpy as np
import rtde_receive
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="192.168.1.100", help="UR5e IP address")
    args = parser.parse_args()

    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)

    tcp_rv = np.array(rtde_r.getActualTCPPose())   # [x,y,z,rx,ry,rz] rotvec
    joints = np.array(rtde_r.getActualQ())          # 6 joint angles (radians)

    euler = Rotation.from_rotvec(tcp_rv[3:]).as_euler("xyz")
    tcp_euler = np.concatenate([tcp_rv[:3], euler])

    print("\n=== UR5e Current TCP Pose ===")
    print(f"  Rotvec (UR native): {np.round(tcp_rv, 6).tolist()}")
    print(f"  Euler XYZ (rad)   : {np.round(tcp_euler, 6).tolist()}")
    print(f"  Joint angles (rad): {np.round(joints, 6).tolist()}")
    print()
    print("Copy the Euler XYZ line into your config.py as:")
    print(f"  TARGET_POSE = np.array({np.round(tcp_euler, 6).tolist()})")
    print()


if __name__ == "__main__":
    main()
