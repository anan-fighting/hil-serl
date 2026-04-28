"""
UR5e USB Pick-Up & Insertion — Training Configuration.

相机布局（单腕部相机版本）：
  wrist_1        — 腕部相机（策略输入 + 奖励分类器共用同一物理相机）

修改清单（上机前必改）：
  [1] ROBOT_IP              — UR5e 的 IP 地址
  [2] REALSENSE_CAMERAS     — 腕部相机序列号
  [3] IMAGE_CROP            — 相机的图像裁剪范围（先跑 record_success_fail.py 预览）
  [4] TARGET_POSE           — USB 完全插入时的末端位姿（用 get_tcp_pose.py 采集）
  [5] RESET_POSE            — 每个 episode 开始时的末端位姿
  [6] ABS_POSE_LIMIT_*      — 安全探索边界
  [7] RESET_JOINTS          — 关节复位目标角度
"""

import os
import jax
import jax.numpy as jnp
import numpy as np

from ur5e_env.envs.ur5e_env import DefaultUR5eEnvConfig
from ur5e_env.envs.wrappers import (
    Quat2EulerWrapper,
    KeyboardIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.ur5e_usb_pickup_insertion.wrapper import UR5eUSBEnv, GripperPenaltyWrapper


# ============================================================
#  环境配置
# ============================================================
class EnvConfig(DefaultUR5eEnvConfig):

    # ------------------------------------------------------------------
    # [1] 机器人 IP
    # ------------------------------------------------------------------
    ROBOT_IP = "192.168.1.103"

    # ------------------------------------------------------------------
    # [2] 相机配置（单腕部相机）
    #   只保留 wrist_1，策略观测与奖励分类器共用同一物理相机。
    # ------------------------------------------------------------------
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "315122272182",
            "dim": (1280, 720),
            "exposure": 25000,
        },
    }

    # ------------------------------------------------------------------
    # [3] 图像裁剪（先运行 record_success_fail.py 预览后调整）
    # ------------------------------------------------------------------
    IMAGE_CROP = {
        "wrist_1": lambda img: img[50:-200, 200:-200],
    }

    # ------------------------------------------------------------------
    # [4] 关键位姿（旋转矢量 rotvec，UR 原生格式，单位：米 / 弧度）
    #   格式：[x, y, z, rx, ry, rz]，其中 (rx,ry,rz) = 旋转轴 × 旋转角度
    #   采集方法：进入 FreeDrive 移动到目标点，然后运行：
    #     conda activate hilserl
    #     cd serl_robot_infra/ur5e_env/utils
    #     python get_tcp_pose.py --robot_ip 192.168.1.103
    #   将输出的 "Rotvec (UR native)" 一行直接填入下方。
    # ------------------------------------------------------------------

    # USB 完全插入 USB 口时的末端位姿（rotvec，需用 get_tcp_pose.py 实测采集）
    # PS: UR示教器上TCP位姿为[0, 0, 190] mm
    TARGET_POSE = np.array([0.243049, -0.448224, 0.142155, 0.220175, 3.133029, -0.004121])

    # 每个 episode 开始时的末端位姿（rotvec，在 TARGET_POSE 基础上偏移）
    RESET_POSE = TARGET_POSE + np.array([-0.1, 0.1, 0.2, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # [5] 安全探索边界（rotvec 分量，与 TARGET_POSE 同格式）
    #
    # ⚠️ 重要：边界必须完整包含 RESET_POSE，否则第一次 step() 就会把
    #    机械臂从 RESET_POSE 强制夹到边界，导致"飞车"！
    #
    #   RESET_POSE 相对 TARGET_POSE 的偏移：[-0.1, +0.1, +0.2, 0, 0, 0]
    #   因此各轴下限/上限需覆盖上述偏移并留有余量。
    #
    #   当前设置（示意，可根据实际工作空间微调）：
    #     x: TARGET ± 0.12 m  → 覆盖 -0.1 偏移
    #     y: -0.08 / +0.12 m  → 覆盖 +0.1 偏移
    #     z: -0.05 / +0.22 m  → 覆盖 +0.2 偏移
    #     rot: ±0.1 rad        → 允许微量姿态调整
    # ------------------------------------------------------------------
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    ABS_POSE_LIMIT_LOW  = TARGET_POSE - np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])

    # ------------------------------------------------------------------
    # 动作缩放  (translation_m, rotation_rad, gripper)
    # ------------------------------------------------------------------
    ACTION_SCALE = (0.015, 0.1, 1.0)

    # 复位随机化
    RANDOM_RESET    = True
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1

    # ------------------------------------------------------------------
    # [6] 关节复位目标角度（弧度，6 个关节）
    # ------------------------------------------------------------------
    RESET_JOINTS = np.array([1.696414, -1.552883, 1.655629, -1.6759, -1.570164, 0.268404])

    # 夹爪
    GRIPPER_TYPE     = "robotiq"
    GRIPPER_PORT     = "/dev/ttyUSB0"   # Modbus RTU 串口
    GRIPPER_SPEED    = 150              # 0-255，建议 100-200
    GRIPPER_FORCE    = 0                # 0-255；0=最小夹持力（USB 插拔推荐）
    GRIPPER_OPEN_MM  = 50.0             # Hand-E 全开约 50 mm
    GRIPPER_CLOSE_MM = 0.0              # 全闭
    GRIPPER_SLEEP    = 0.6              # 等待夹爪动作完成（秒）

    MAX_EPISODE_LENGTH = 120
    DISPLAY_IMAGE      = True
    JOINT_RESET_PERIOD = 50

    # RTDE 伺服参数（通常无需修改）
    SERVO_LOOKAHEAD_TIME = 0.1
    SERVO_GAIN           = 300

    # 键盘遥操速度
    KEYBOARD_LINEAR_SPEED  = 1.0
    KEYBOARD_ANGULAR_SPEED = 1.0


# ============================================================
#  训练配置
# ============================================================
class TrainConfig(DefaultTrainingConfig):
    # 策略观测使用的图像键（单腕部相机）
    image_keys      = ["wrist_1"]
    # 奖励分类器使用的图像键（与策略观测相同，共用 wrist_1）
    classifier_keys = ["wrist_1"]
    # 本体感知键
    proprio_keys    = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]

    checkpoint_period  = 2000
    cta_ratio          = 2
    random_steps       = 0
    discount           = 0.98
    buffer_period      = 1000
    encoder_type       = "resnet-pretrained"
    # 夹爪由策略控制（learned gripper）
    setup_mode         = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = UR5eUSBEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        if not fake_env:
            env = KeyboardIntervention(
                env,
                linear_speed=EnvConfig.KEYBOARD_LINEAR_SPEED,
                angular_speed=EnvConfig.KEYBOARD_ANGULAR_SPEED,
            )
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # 分类器置信度 > 0.7  且  夹爪已闭合（gripper_pose > 0.4）
                return int(
                    sigmoid(classifier_fn(obs)) > 0.7
                    and obs["state"][0, 0] > 0.4
                )

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        # 夹爪频繁切换惩罚项（记录到 info["grasp_penalty"]）
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env

    def process_demos(self, demo):
        return demo
