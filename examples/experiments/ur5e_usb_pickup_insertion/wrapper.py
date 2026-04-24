"""
UR5e USB Pick-Up & Insertion task — task-specific Gym environment.

Reset procedure (mirrors Franka USBEnv):
  1. Open gripper → drop USB back onto the table.
  2. Move end-effector to just above TARGET_POSE (USB port).
  3. Move down to TARGET_POSE and close gripper to grasp USB.
  4. Lift up to RESET_POSE with optional XY / Rz randomisation.

Camera initialisation:
  "side_classifier" reuses the same physical capture object as
  "side_policy" (identical to the Franka USBEnv trick).
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
    # Camera init: side_classifier shares side_policy capture object
    # ------------------------------------------------------------------ #
    def init_cameras(self, name_serial_dict=None):
        if self.cap is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "side_classifier":
                # Share the same VideoCapture object as side_policy
                self.cap["side_classifier"] = self.cap["side_policy"]
            else:
                self.cap[cam_name] = VideoCapture(RSCapture(name=cam_name, **kwargs))

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, joint_reset: bool = False, **kwargs):
        """
        Full reset for USB task:
          1. Open gripper (drop USB).
          2. Move above USB port (TARGET_POSE + z offset).
          3. Move down to TARGET_POSE.
          4. Close gripper (pick up USB).
          5. Move to RESET_POSE (with optional randomisation).
        """
        self._recover()

        if self.save_video:
            self._save_video_recording()

        # --- Step 1: open gripper to drop USB ---
        if self.gripper is not None:
            self.gripper.move(
                0,
                int(self.config.GRIPPER_SPEED * 255),
                int(self.config.GRIPPER_FORCE * 255),
            )
            self._gripper_is_closed = False
            time.sleep(self.gripper_sleep)

        # --- Step 2: move to just above the USB port ---
        above_target = self._RESET_POSE.copy()          # euler [x,y,z,rx,ry,rz]
        above_target[2] = self._TARGET_POSE[2] + 0.05   # 5 cm above port
        self.interpolate_move(above_target, timeout=1.0)
        time.sleep(0.3)

        # --- Step 3: descend to USB port ---
        self.interpolate_move(self._TARGET_POSE, timeout=0.8)
        time.sleep(0.4)

        # --- Step 4: close gripper (grasp USB) ---
        if self.gripper is not None:
            self.gripper.move(
                255,
                int(self.config.GRIPPER_SPEED * 255),
                int(self.config.GRIPPER_FORCE * 255),
            )
            self._gripper_is_closed = True
            time.sleep(self.gripper_sleep)

        # --- Step 5: lift to reset pose (with optional randomisation) ---
        reset_pose = self._RESET_POSE.copy()
        if self.randomreset:
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            reset_pose[5] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
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
        # obs["state"] is shape (obs_horizon, state_dim) after ChunkingWrapper,
        # but here it may still be a flat dict — handle both.
        gp = obs["state"].get("gripper_pose", None)
        if gp is not None:
            self.last_gripper_pos = float(np.asarray(gp).flat[0])
        else:
            self.last_gripper_pos = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Use intervene_action if available (human override)
        act = info.get("intervene_action", action)

        gp = obs["state"].get("gripper_pose", None)
        current_gripper = float(np.asarray(gp).flat[0]) if gp is not None else self.last_gripper_pos

        if (act[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            act[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = current_gripper
        return obs, reward, terminated, truncated, info
