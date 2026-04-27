"""
Gym wrappers for UR5eEnv.

Mirrors franka_env/envs/wrappers.py, replacing SpaceMouseExpert
with KeyboardExpert for human intervention.
"""

import time
from typing import List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

from ur5e_env.keyboard.keyboard_expert import KeyboardExpert
from ur5e_env.envs.ur5e_env import UR5eEnv


# ---------------------------------------------------------------------------
# Observation wrappers
# ---------------------------------------------------------------------------

class Quat2EulerWrapper(gym.ObservationWrapper):
    """Convert tcp_pose from [x,y,z,qx,qy,qz,qw] to [x,y,z,rx,ry,rz] (euler)."""

    def __init__(self, env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, obs):
        tcp = obs["state"]["tcp_pose"]
        obs["state"]["tcp_pose"] = np.concatenate(
            [tcp[:3], R.from_quat(tcp[3:]).as_euler("xyz")]
        )
        return obs


# ---------------------------------------------------------------------------
# Reward wrappers
# ---------------------------------------------------------------------------

class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    Override reward with a trained image classifier.
    Identical interface to the Franka version.
    """

    def __init__(self, env: gym.Env, reward_classifier_func, target_hz=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        t0 = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or bool(rew)
        info["succeed"] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - t0)))
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


class HumanClassifierWrapper(gym.Wrapper):
    """Ask the human for reward via stdin."""

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0): "))
                    assert rew in (0, 1)
                    break
                except Exception:
                    continue
        info["succeed"] = bool(rew)
        return obs, rew, done, truncated, info


# ---------------------------------------------------------------------------
# Action wrappers
# ---------------------------------------------------------------------------

class GripperCloseEnv(gym.ActionWrapper):
    """
    Remove the gripper dimension from the action space.
    Useful for tasks where the gripper is always kept closed.
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new = np.zeros(7, dtype=np.float32)
        new[:6] = action
        return new

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class KeyboardIntervention(gym.ActionWrapper):
    """
    Human-in-the-loop intervention wrapper using a keyboard instead of SpaceMouse.

    When any motion key is held, the keyboard teleoperation action overrides
    the policy action.  The override action is stored as ``info["intervene_action"]``
    so the training loop can identify it as a human demonstration.

    Gripper keys:
        F (hold) → close gripper   (button[0] = 1)
        G (hold) → open  gripper   (button[1] = 1)

    Motion keys: see KeyboardExpert docstring.
    """

    def __init__(
        self,
        env,
        linear_speed: float = 1.0,
        angular_speed: float = 1.0,
        action_indices=None,
    ):
        super().__init__(env)

        self.gripper_enabled = self.action_space.shape == (7,)
        self.expert = KeyboardExpert(
            linear_speed=linear_speed,
            angular_speed=angular_speed,
        )
        self.action_indices = action_indices
        self._close_gripper = False
        self._open_gripper  = False

    def action(self, action: np.ndarray):
        """
        Returns (final_action, intervened: bool).
        Overrides gym.ActionWrapper.action() signature intentionally.
        """
        expert_a, buttons = self.expert.get_action() # 读键盘当前按下的键
        self._close_gripper, self._open_gripper = bool(buttons[0]), bool(buttons[1])

        intervened = np.linalg.norm(expert_a) > 0.001 # 有运动键被按住

        if self.gripper_enabled: 
            if self._close_gripper: # F 键
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self._open_gripper: # G 键
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros(1)
            expert_a = np.concatenate([expert_a, gripper_action])

        if self.action_indices is not None:
            filtered = np.zeros_like(expert_a)
            filtered[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered

        if intervened:
            return expert_a, True # ← 用键盘动作替换策略动作
        return action, False # ← 策略动作原样通过

    def step(self, action):
        # Propagate terminate flag from keyboard ESC
        if self.expert.terminate:
            self.env.unwrapped.terminate = True

        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action # ← 打上干预标记，训练循环识别
        info["close_gripper"] = self._close_gripper
        info["open_gripper"]  = self._open_gripper
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self.expert.terminate = False
        return self.env.reset(**kwargs)

    def close(self):
        self.expert.close()
        super().close()
