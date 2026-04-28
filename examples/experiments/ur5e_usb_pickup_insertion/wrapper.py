"""
UR5e USB Pick-Up & Insertion task — task-specific Gym environment.

Reset procedure (mirrors Franka USBEnv):
  1. Open gripper → drop USB back onto the table.
  2. Move end-effector to just above TARGET_POSE (USB port).
  3. Move down to TARGET_POSE and close gripper to grasp USB.
  4. Lift up to RESET_POSE with optional XY / Rz randomisation.

Camera initialisation:
  Single wrist camera (wrist_1) — used for both policy and classifier.
"""

import copy
import time
from collections import OrderedDict
from typing import Optional

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from ur5e_env.envs.ur5e_env import UR5eEnv
from ur5e_env.utils.rotations import euler_2_quat, pose_rotvec_to_quat, pose_quat_to_rotvec


class UR5eUSBEnv(UR5eEnv):
    """
    UR5e environment for the USB pick-up and insertion task.

    The reset sequence mirrors the Franka USBEnv:
      open gripper → move to port → grasp → lift to reset pose
    """

    # ------------------------------------------------------------------ #
    # Camera init: single wrist camera, standard init
    # ------------------------------------------------------------------ #
    def init_cameras(self, name_serial_dict=None):
        if self.cap is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            self.cap[cam_name] = VideoCapture(RSCapture(name=cam_name, **kwargs))

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, joint_reset: bool = False, skip_grasp: bool = False, **kwargs):
        """
        Full reset for USB task.

        Args:
            skip_grasp: If True (or self.skip_grasp is True), skip the
                        descend-grasp-lift sequence and only move to RESET_POSE.
                        Useful for image preview / classifier data collection.
                        Can also be set via  env.unwrapped.skip_grasp = True
                        to work around gymnasium's fixed reset() signature.
        """
        # Allow setting via attribute (bypasses gymnasium wrapper signature)
        skip_grasp = skip_grasp or getattr(self, "skip_grasp", False)

        # Stop any ongoing servoL stream from the previous step() before issuing
        # interpolate_move / moveL / moveJ, otherwise the robot may "fly" to the
        # safety-box boundary when clip_safety_box acts on the buffered command.
        if not self.fake_env:
            try:
                self.rtde_c.servoStop(a=2.0)
                time.sleep(0.05)  # let the deceleration ramp finish
            except Exception:
                pass

        self._recover()

        if self.save_video:
            self._save_video_recording()

        if skip_grasp:
            # ── 仅预览模式：直接移动到 RESET_POSE，不做抓取 ──────────────
            reset_pose = self._RESET_POSE.copy()
            self.interpolate_move(reset_pose, timeout=2.0)
            print("[wrapper.py] Skip grasp enabled: moved to RESET_POSE without grasping.")
        else:
            # ── 完整重置（训练模式）────────────────────────────────────────
            # --- Step 1: open gripper to drop USB ---
            if self.gripper is not None:
                self.gripper.move(
                    self.config.GRIPPER_OPEN_MM,
                    self.config.GRIPPER_SPEED,
                    self.config.GRIPPER_FORCE,
                    block=True,
                )
                self._gripper_is_closed = False
                time.sleep(self.gripper_sleep)

            # --- Step 2: move to just above the USB port ---
            above_target = self._RESET_POSE.copy()
            above_target[2] = self._TARGET_POSE[2] + 0.05   # 5 cm above port
            self.interpolate_move(above_target, timeout=1.0)
            time.sleep(0.3)

            # --- Step 3: descend to USB port ---
            self.interpolate_move(self._TARGET_POSE, timeout=0.8)
            time.sleep(0.4)

            # --- Step 4: close gripper (grasp USB) ---
            if self.gripper is not None:
                self.gripper.move(
                    self.config.GRIPPER_CLOSE_MM,
                    self.config.GRIPPER_SPEED,
                    self.config.GRIPPER_FORCE,
                    block=True,
                )
                self._gripper_is_closed = True
                time.sleep(self.gripper_sleep)

            # --- Step 5: lift to reset pose (with optional randomisation) ---
            # ⚠️ 先沿 Z 轴竖直抬升到 RESET_POSE 高度，再平移到 XY 目标，
            #    避免夹住 USB 后斜向拔出导致碰撞。
            reset_pose = self._RESET_POSE.copy()
            if self.randomreset:
                reset_pose[:2] += np.random.uniform(
                    -self.random_xy_range, self.random_xy_range, (2,)
                )
                reset_pose[5] += np.random.uniform(
                    -self.random_rz_range, self.random_rz_range
                )
            # 第一段：仅抬高 Z 轴（保持当前 XY 和姿态不变）
            lift_pose = self._TARGET_POSE.copy()
            lift_pose[2] = reset_pose[2]          # 目标 Z 高度
            self.interpolate_move(lift_pose, timeout=1.0)
            time.sleep(0.1)
            # 第二段：平移到最终 reset_pose（XY + 旋转）
            self.interpolate_move(reset_pose, timeout=1.0)

        # Joint reset if triggered by cycle counter
        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            print("[UR5eUSBEnv] Joint reset …")
            self.rtde_c.moveJ(
                list(self.config.RESET_JOINTS.astype(float)),
                speed=0.3, acceleration=0.3,
            )

        self.curr_path_length = 0
        self.terminate = False
        self._update_currpos()
        obs = self._get_obs()
        return obs, {"succeed": False}

    # ------------------------------------------------------------------ #
    # Override go_to_reset (not used directly here, kept for compatibility)
    # ------------------------------------------------------------------ #
    def go_to_reset(self, joint_reset: bool = False):
        """Thin wrapper; full logic lives in reset()."""
        pass


# ---------------------------------------------------------------------------
# GripperPenaltyWrapper  (mirrors franka USBEnv version)
# ---------------------------------------------------------------------------
class GripperPenaltyWrapper(gym.Wrapper):
    """
    Add a small penalty term to `info["grasp_penalty"]` whenever the
    policy changes the gripper state (open↔close), discouraging excessive
    gripper toggling.  The penalty is NOT added to `reward` — it is logged
    separately so the learner can optionally incorporate it.
    """

    def __init__(self, env: gym.Env, penalty: float = -0.02):
        super().__init__(env)
        assert env.action_space.shape == (7,), (
            "GripperPenaltyWrapper expects action_space shape (7,)"
        )
        self.penalty = penalty
        self.last_gripper_pos: Optional[float] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # After SERLObsWrapper+ChunkingWrapper, obs["state"] is a flat np.ndarray.
        # gripper_pose is the first element of the state vector (see proprio_keys order).
        try:
            self.last_gripper_pos = float(np.asarray(obs["state"]).flat[0])
        except Exception:
            self.last_gripper_pos = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Use intervene_action if available (human override)
        act = info.get("intervene_action", action)

        try:
            current_gripper = float(np.asarray(obs["state"]).flat[0])
        except Exception:
            current_gripper = self.last_gripper_pos

        if (act[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            act[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = current_gripper
        return obs, reward, terminated, truncated, info
