"""
Gym Interface for UR5e controlled via RTDE (no ROS required).

Design mirrors franka_env/envs/franka_env.py so that the same wrappers,
training scripts, and experiment configs can be reused with minimal changes.

Dependencies:
    pip install ur-rtde opencv-python gymnasium scipy numpy

Hardware:
    - UR5e with built-in wrist force/torque sensor (or external F/T)
    - RealSense cameras (same as Franka setup)
    - Robotiq 2F-85 gripper (or any gripper controlled via serial/RTDE)
    - 100 Mbit/s Ethernet connection to robot

RTDE real-time control uses `servoL` (Cartesian servo) which streams
end-effector poses at the control frequency.  This gives sub-millisecond
latency without ROS.
"""

import os
import copy
import queue
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

# UR RTDE Python bindings  (pip install ur-rtde)
import rtde_control
import rtde_receive

from ur5e_env.utils.rotations import (
    rotvec_2_quat,
    pose_rotvec_to_quat,
    pose_quat_to_rotvec,
)

# Reuse the RealSense camera wrappers from the Franka infra
from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture


# ---------------------------------------------------------------------------
# Image display helper (identical to FrankaEnv)
# ---------------------------------------------------------------------------
class ImageDisplayer(threading.Thread):
    def __init__(self, q: queue.Queue, name: str):
        super().__init__()
        self.queue = q
        self.daemon = True
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break
            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k],
                axis=1,
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
class DefaultUR5eEnvConfig:
    """
    Default configuration for UR5eEnv.
    Override these in your task-specific config class.
    """

    # UR5e network address
    ROBOT_IP: str = "192.168.1.103"

    # RTDE servo parameters
    SERVO_LOOKAHEAD_TIME: float = 0.1   # seconds (0.03 – 0.2)
    SERVO_GAIN: float = 300             # proportional gain (100 – 2000)
    SERVO_DT: float = 0.002            # UR5e controller time-step (500 Hz)

    # RealSense cameras  {name: {serial_number, dim, exposure}}
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": {
            "serial_number": "REPLACE_WITH_SERIAL",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }

    # Callable crops:  IMAGE_CROP = {"wrist_1": lambda img: img[y0:y1, x0:x1]}
    IMAGE_CROP: Dict = {}

    # Poses in [x, y, z, rx, ry, rz] as rotvec (axis-angle, UR native format, radians)
    TARGET_POSE:  np.ndarray = np.zeros(6)
    RESET_POSE:   np.ndarray = np.zeros(6)
    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])

    # Action scale: (translation_m, rotation_rad, gripper)
    ACTION_SCALE = (0.01, 0.06, 1.0)

    # Workspace safety bounding box (rotvec pose)
    ABS_POSE_LIMIT_LOW:  np.ndarray = np.zeros(6)
    ABS_POSE_LIMIT_HIGH: np.ndarray = np.zeros(6)

    # Reset randomisation
    RANDOM_RESET:    bool  = False
    RANDOM_XY_RANGE: float = 0.02   # metres
    RANDOM_RZ_RANGE: float = 0.05   # radians

    # Joint configuration for safe home reset
    RESET_JOINTS: np.ndarray = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

    # Gripper parameters
    # GRIPPER_TYPE: "robotiq" (Modbus RTU via serial port) or "none"
    GRIPPER_TYPE:  str   = "robotiq"
    GRIPPER_PORT:  str   = "/dev/ttyUSB0"  # serial port for Modbus RTU
    # GRIPPER_SPEED / FORCE: 0-255 (passed directly to HandEForRtu.move)
    GRIPPER_SPEED: int   = 150   # 0-255
    GRIPPER_FORCE: int   = 0     # 0-255
    GRIPPER_SLEEP: float = 0.6   # seconds to wait after gripper cmd
    # Fully open position in mm (HandEForRtu.FULL_POS = 50 mm)
    GRIPPER_OPEN_MM:  float = 50.0
    GRIPPER_CLOSE_MM: float = 0.0

    DISPLAY_IMAGE:      bool = True
    MAX_EPISODE_LENGTH: int  = 100

    # Joint reset period: perform a joint-space reset every N episodes (0 = never)
    JOINT_RESET_PERIOD: int = 0


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------
class UR5eEnv(gym.Env):
    """
    Gymnasium environment for UR5e controlled via RTDE.

    Observation space::

        {
          "state": {
            "tcp_pose"    : (7,)  xyz + quat
            "tcp_vel"     : (6,)  linear + angular velocity
            "gripper_pose": (1,)  normalised gripper width  [0=open, 1=closed]
            "tcp_force"   : (3,)  force  in TCP frame
            "tcp_torque"  : (3,)  torque in TCP frame
          },
          "images": {
            "<cam_name>": (128, 128, 3)
          }
        }

    Action space: Box(-1, 1, (7,))
        [dx, dy, dz, drx, dry, drz, dgripper]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        hz: int = 10,
        fake_env: bool = False,
        save_video: bool = False,
        config: DefaultUR5eEnvConfig = None,
    ):
        super().__init__()

        if config is None:
            config = DefaultUR5eEnvConfig()

        self.config = config
        self.hz = hz
        self.fake_env = fake_env
        self.save_video = save_video

        self.action_scale   = config.ACTION_SCALE
        self._TARGET_POSE   = config.TARGET_POSE.copy()
        self._RESET_POSE    = config.RESET_POSE.copy()
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD.copy()

        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image      = config.DISPLAY_IMAGE
        self.gripper_sleep      = config.GRIPPER_SLEEP
        self.randomreset        = config.RANDOM_RESET
        self.random_xy_range    = config.RANDOM_XY_RANGE
        self.random_rz_range    = config.RANDOM_RZ_RANGE
        self.joint_reset_cycle  = config.JOINT_RESET_PERIOD
        self.cycle_count        = 0

        # ------------------------------------------------------------------ #
        # Bounding box (euler-space)
        # ------------------------------------------------------------------ #
        self.xyz_bounding_box = spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )

        # ------------------------------------------------------------------ #
        # Gym spaces
        # ------------------------------------------------------------------ #
        self.action_space = spaces.Box(
            np.ones(7, dtype=np.float32) * -1,
            np.ones(7, dtype=np.float32),
        )

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "tcp_pose":     spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel":      spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": spaces.Box(-1, 1,    shape=(1,)),
                        "tcp_force":    spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque":   spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": spaces.Dict(
                    {
                        key: spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                        for key in config.REALSENSE_CAMERAS
                    }
                ),
            }
        )

        if save_video:
            self.recording_frames = []

        # ------------------------------------------------------------------ #
        # Internal state (populated by _update_currpos)
        # ------------------------------------------------------------------ #
        # currpos: [x, y, z, qx, qy, qz, qw]
        self.currpos        = np.zeros(7)
        self.currvel        = np.zeros(6)
        self.currforce      = np.zeros(3)
        self.currtorque     = np.zeros(3)
        self.curr_gripper_pos = np.array([0.0])   # 0 = open, 1 = closed
        self._gripper_is_closed = False           # software-side latch (used in _send_gripper_command)
        self.curr_path_length = 0

        # terminate flag (set by keyboard ESC listener inside wrappers)
        self.terminate = False

        if fake_env:
            return

        # ------------------------------------------------------------------ #
        # RTDE connection
        # ------------------------------------------------------------------ #
        print(f"[UR5eEnv] Connecting to robot at {config.ROBOT_IP} …")
        self.rtde_c = rtde_control.RTDEControlInterface(config.ROBOT_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(config.ROBOT_IP)
        print("[UR5eEnv] RTDE connected.")

        # Gripper (Robotiq Hand-E via Modbus RTU over serial)
        self.gripper = None
        if config.GRIPPER_TYPE == "robotiq":
            from ur5e_env.keyboard.robotiq.HandE import HandEForRtu
            print(f"[UR5eEnv] Initialising Robotiq gripper on {config.GRIPPER_PORT} …")
            self.gripper = HandEForRtu(config.GRIPPER_PORT, autoInit=True)
            # Open gripper at startup
            self.gripper.move(
                pos=config.GRIPPER_OPEN_MM,
                speed=config.GRIPPER_SPEED,
                force=config.GRIPPER_FORCE,
                block=True,
            )
            time.sleep(config.GRIPPER_SLEEP)
            print("[UR5eEnv] Robotiq gripper ready (open).")

        # ------------------------------------------------------------------ #
        # Cameras
        # ------------------------------------------------------------------ #
        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)

        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, f"UR5e-{config.ROBOT_IP}")
            self.displayer.start()

        # Initial state read
        self._update_currpos()
        print("[UR5eEnv] Ready.")

    # ====================================================================== #
    # Core gym methods
    # ====================================================================== #

    def step(self, action: np.ndarray):
        """Standard Gym step."""
        t0 = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta   = action[:3] * self.action_scale[0]
        rotvec_delta = action[3:6] * self.action_scale[1]
        gripper_action = action[6] * self.action_scale[2]

        # Build next pose in quat space then convert to rotvec for RTDE
        next_pos = self.currpos[:3] + xyz_delta
        next_quat = (
            Rotation.from_rotvec(rotvec_delta) * Rotation.from_quat(self.currpos[3:])
        ).as_quat()
        next_pose_quat = np.concatenate([next_pos, next_quat])
        next_pose_quat = self.clip_safety_box(next_pose_quat)

        # Send commands
        self._send_gripper_command(gripper_action)
        self._send_pos_command(next_pose_quat)

        self.curr_path_length += 1
        dt = time.time() - t0
        time.sleep(max(0, 1.0 / self.hz - dt))

        self._update_currpos()
        obs = self._get_obs()
        reward = self.compute_reward(obs)
        done = (
            self.curr_path_length >= self.max_episode_length
            or reward
            or self.terminate
        )
        return obs, int(reward), done, False, {"succeed": bool(reward)}

    def reset(self, joint_reset: bool = False, **kwargs):
        self._recover()

        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        if self.save_video:
            self._save_video_recording()

        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0
        self.terminate = False

        self._update_currpos()
        obs = self._get_obs()
        return obs, {"succeed": False}

    def compute_reward(self, obs) -> bool:
        """
        Default pose-threshold reward.
        Override in task-specific wrapper for classifier-based reward.
        """
        tcp_pose = obs["state"]["tcp_pose"]  # xyz + quat
        cur_rot    = Rotation.from_quat(tcp_pose[3:]).as_matrix()
        target_rot = Rotation.from_rotvec(self._TARGET_POSE[3:]).as_matrix()
        diff_rotvec = Rotation.from_matrix(cur_rot.T @ target_rot).as_rotvec()
        delta = np.abs(np.hstack([tcp_pose[:3] - self._TARGET_POSE[:3], diff_rotvec]))
        return bool(np.all(delta < self._REWARD_THRESHOLD))

    def close(self):
        if not self.fake_env:
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
            self.close_cameras()
            if self.display_image:
                self.img_queue.put(None)
                cv2.destroyAllWindows()
                self.displayer.join()

    # ====================================================================== #
    # Motion helpers
    # ====================================================================== #

    def clip_safety_box(self, pose_quat: np.ndarray) -> np.ndarray:
        """Clip xyz+quat pose to the configured workspace bounding box (limits in rotvec)."""
        pose_quat[:3] = np.clip(
            pose_quat[:3],
            self.xyz_bounding_box.low,
            self.xyz_bounding_box.high,
        )
        # Clip in rotvec space (component-wise, matches config ABS_POSE_LIMIT rotvec format)
        rotvec = Rotation.from_quat(pose_quat[3:]).as_rotvec()
        rotvec = np.clip(rotvec, self.rpy_bounding_box.low, self.rpy_bounding_box.high)
        pose_quat[3:] = Rotation.from_rotvec(rotvec).as_quat()
        return pose_quat

    def interpolate_move(self, goal_rotvec: np.ndarray, timeout: float = 2.0):
        """
        Move smoothly to `goal_rotvec` ([x,y,z, rx,ry,rz] in rotvec / UR native format)
        by linearly interpolating poses and streaming via servoL.
        """
        goal_quat = np.concatenate(
            [goal_rotvec[:3], rotvec_2_quat(goal_rotvec[3:])]
        )
        steps = max(2, int(timeout * self.hz))
        self._update_currpos()
        # Interpolate in rotvec space (SLERP-like)
        r_start = Rotation.from_quat(self.currpos[3:])
        r_end   = Rotation.from_quat(goal_quat[3:])
        ts = np.linspace(0, 1, steps)
        for t in ts:
            pos    = (1 - t) * self.currpos[:3] + t * goal_quat[:3]
            r_interp = Rotation.from_quat(
                Rotation.slerp(r_start, r_end, t) if False  # scipy ≥1.8
                else _slerp(r_start, r_end, t)
            ).as_rotvec()
            self.rtde_c.servoL(
                list(np.concatenate([pos, r_interp])),
                0.5, 0.3,
                self.config.SERVO_DT,
                self.config.SERVO_LOOKAHEAD_TIME,
                self.config.SERVO_GAIN,
            )
            time.sleep(1.0 / self.hz)
        self._update_currpos()

    def go_to_reset(self, joint_reset: bool = False):
        """
        Default reset: move straight back to RESET_POSE.
        Override in task-specific subclass for custom reset logic.
        """
        if joint_reset:
            print("[UR5eEnv] Performing joint reset …")
            self.rtde_c.moveJ(
                list(self.config.RESET_JOINTS),
                speed=0.5, acceleration=0.5
            )

        reset_pose = self._RESET_POSE.copy()
        if self.randomreset:
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            reset_pose[5] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
        self.interpolate_move(reset_pose, timeout=2.0)

    # ====================================================================== #
    # Low-level command helpers
    # ====================================================================== #

    def _send_pos_command(self, pose_quat: np.ndarray):
        """Send a Cartesian pose command via servoL (RTDE)."""
        pose_rv = pose_quat_to_rotvec(pose_quat)
        self.rtde_c.servoL(
            list(pose_rv.astype(float)),
            0.5,                              # velocity
            0.3,                              # acceleration
            self.config.SERVO_DT,             # dt
            self.config.SERVO_LOOKAHEAD_TIME, # lookahead_time
            self.config.SERVO_GAIN,           # gain
        )

    def _send_gripper_command(self, action: float, mode: str = "binary"):
        """
        Control the Robotiq gripper.

        action > +0.5  → open
        action < -0.5  → close
        """
        if self.gripper is None:
            return
        if mode == "binary":
            if action <= -0.5 and not self._gripper_is_closed:
                self.gripper.move(
                    self.config.GRIPPER_CLOSE_MM,
                    self.config.GRIPPER_SPEED,
                    self.config.GRIPPER_FORCE,
                    block=True,
                )
                self._gripper_is_closed = True
                time.sleep(self.gripper_sleep)
            elif action >= 0.5 and self._gripper_is_closed:
                self.gripper.move(
                    self.config.GRIPPER_OPEN_MM,
                    self.config.GRIPPER_SPEED,
                    self.config.GRIPPER_FORCE,
                    block=True,
                )
                self._gripper_is_closed = False
                time.sleep(self.gripper_sleep)

    def _update_currpos(self):
        """Read robot state from RTDE and cache it."""
        # TCP pose as rotvec → convert to xyz+quat
        tcp_rv  = np.array(self.rtde_r.getActualTCPPose())   # [x,y,z,rx,ry,rz]
        tcp_vel = np.array(self.rtde_r.getActualTCPSpeed())  # [vx,vy,vz,wx,wy,wz]

        self.currpos = pose_rotvec_to_quat(tcp_rv)           # [x,y,z,qx,qy,qz,qw]
        self.currvel = tcp_vel

        # Force/torque from wrist sensor (built-in UR5e)
        ft = np.array(self.rtde_r.getActualTCPForce())       # [Fx,Fy,Fz, Tx,Ty,Tz]
        self.currforce  = ft[:3]
        self.currtorque = ft[3:]

        # Gripper state: normalised to [0=open, 1=closed]
        if self.gripper is not None:
            pos_mm = self.gripper.position  # real position in mm from Modbus
            open_mm = max(self.config.GRIPPER_OPEN_MM, 1e-6)  # avoid div/0
            self.curr_gripper_pos = np.array([1.0 - float(pos_mm) / open_mm])
        else:
            self.curr_gripper_pos = np.array([float(self._gripper_is_closed)])

    def _recover(self):
        """Clear protective stops / errors."""
        try:
            if self.rtde_r.getSafetyMode() != 1:  # 1 = NORMAL
                self.rtde_c.unlockProtectiveStop()
                time.sleep(1.0)
        except Exception:
            pass

    def _get_obs(self) -> dict:
        images = self.get_im()
        state  = {
            "tcp_pose":     self.currpos.copy(),
            "tcp_vel":      self.currvel.copy(),
            "gripper_pose": self.curr_gripper_pos.copy(),
            "tcp_force":    self.currforce.copy(),
            "tcp_torque":   self.currtorque.copy(),
        }
        return copy.deepcopy(dict(images=images, state=state))

    # ====================================================================== #
    # Camera helpers (identical logic to FrankaEnv)
    # ====================================================================== #

    def init_cameras(self, name_serial_dict=None):
        if self.cap is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            self.cap[cam_name] = VideoCapture(RSCapture(name=cam_name, **kwargs))

    def close_cameras(self):
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"[UR5eEnv] Failed to close cameras: {e}")

    def get_im(self) -> Dict:
        images = {}
        display_images = {}
        full_res = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped,
                    self.observation_space["images"][key].shape[:2][::-1],
                )
                images[key]              = resized[..., ::-1]
                display_images[key]      = resized
                display_images[key + "_full"] = cropped
                full_res[key]            = copy.deepcopy(cropped)
            except queue.Empty:
                input(f"{key} camera frozen. Check connection, then press Enter …")
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        if self.save_video:
            self.recording_frames.append(full_res)
        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def _save_video_recording(self):
        try:
            if not self.recording_frames:
                return
            os.makedirs("./videos", exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for cam_key in self.recording_frames[0]:
                path = f"./videos/ur5e_{cam_key}_{ts}.mp4"
                h, w = self.recording_frames[0][cam_key].shape[:2]
                vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                for fd in self.recording_frames:
                    vw.write(fd[cam_key])
                vw.release()
                print(f"[UR5eEnv] Saved video {path}")
            self.recording_frames.clear()
        except Exception as e:
            print(f"[UR5eEnv] Failed to save video: {e}")


# ---------------------------------------------------------------------------
# SLERP helper (scipy < 1.8 doesn't have Rotation.slerp)
# ---------------------------------------------------------------------------
def _slerp(r0: Rotation, r1: Rotation, t: float) -> np.ndarray:
    """Return quaternion of spherical interpolation between r0 and r1 at t∈[0,1]."""
    q0 = r0.as_quat()
    q1 = r1.as_quat()
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = q0 + t * (q1 - q0)
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q).as_rotvec()
