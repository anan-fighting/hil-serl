"""
UR5e RTDE utility functions: Euler/Quaternion conversions.
Mirrors franka_env/utils/rotations.py interface.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def euler_2_quat(euler: np.ndarray) -> np.ndarray:
    """Convert XYZ euler angles (radians) to quaternion [x, y, z, w]."""
    return Rotation.from_euler("xyz", euler).as_quat()


def quat_2_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to XYZ euler angles (radians)."""
    return Rotation.from_quat(quat).as_euler("xyz")


def rotvec_2_quat(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector (axis-angle) to quaternion [x, y, z, w]."""
    return Rotation.from_rotvec(rotvec).as_quat()


def quat_2_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to rotation vector (axis-angle)."""
    return Rotation.from_quat(quat).as_rotvec()


def pose_euler_to_rotvec(pose_euler: np.ndarray) -> np.ndarray:
    """
    Convert pose [x, y, z, rx, ry, rz (euler)] to UR pose [x, y, z, rx, ry, rz (rotvec)].
    """
    pos = pose_euler[:3]
    rotvec = Rotation.from_euler("xyz", pose_euler[3:]).as_rotvec()
    return np.concatenate([pos, rotvec])


def pose_rotvec_to_euler(pose_rotvec: np.ndarray) -> np.ndarray:
    """
    Convert UR pose [x, y, z, rx, ry, rz (rotvec)] to [x, y, z, rx, ry, rz (euler)].
    """
    pos = pose_rotvec[:3]
    euler = Rotation.from_rotvec(pose_rotvec[3:]).as_euler("xyz")
    return np.concatenate([pos, euler])


def pose_rotvec_to_quat(pose_rotvec: np.ndarray) -> np.ndarray:
    """
    Convert UR pose [x, y, z, rx, ry, rz (rotvec)] to [x, y, z, qx, qy, qz, qw].
    """
    pos = pose_rotvec[:3]
    quat = rotvec_2_quat(pose_rotvec[3:])
    return np.concatenate([pos, quat])


def pose_quat_to_rotvec(pose_quat: np.ndarray) -> np.ndarray:
    """
    Convert [x, y, z, qx, qy, qz, qw] to UR pose [x, y, z, rx, ry, rz (rotvec)].
    """
    pos = pose_quat[:3]
    rotvec = quat_2_rotvec(pose_quat[3:])
    return np.concatenate([pos, rotvec])
