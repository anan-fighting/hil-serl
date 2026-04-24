"""
UR5e USB Pick-Up & Insertion — Training Configuration.

相机布局（与 Franka USB 任务一致）：
  wrist_1        — 腕部相机 1（策略输入）
  wrist_2        — 腕部相机 2（策略输入）
  side_policy    — 侧方相机策略视图（策略输入）
  side_classifier — 侧方相机分类器视图（仅用于奖励分类器，与 side_policy 共用同一物理相机）

修改清单（上机前必改）：
  [1] ROBOT_IP              — UR5e 的 IP 地址
  [2] REALSENSE_CAMERAS     — 各相机序列号
  [3] IMAGE_CROP            — 各相机的图像裁剪范围（先跑 record_success_fail.py 预览）
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
    ROBOT_IP = "192.168.1.100"

    # ------------------------------------------------------------------
    # [2] 相机配置
    #   side_policy 与 side_classifier 的 serial_number 填写同一个序列号，
    #   wrapper.py 中会让它们共用同一 VideoCapture 对象（避免重复初始化）。
    # ------------------------------------------------------------------
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "REPLACE_WRIST1_SERIAL",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "wrist_2": {
            "serial_number": "REPLACE_WRIST2_SERIAL",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "side_policy": {
            "serial_number": "REPLACE_SIDE_SERIAL",
            "dim": (1280, 720),
            "exposure": 13000,
        },
        "side_classifier": {
            # 与 side_policy 填写相同序列号；wrapper 会共用 capture 对象
            "serial_number": "REPLACE_SIDE_SERIAL",
            "dim": (1280, 720),
            "exposure": 13000,
        },
    }

    # ------------------------------------------------------------------
    # [3] 图像裁剪（先运行 record_success_fail.py 预览后调整）
    # ------------------------------------------------------------------
    IMAGE_CROP = {
        "wrist_1":         lambda img: img[50:-200, 200:-200],
        "wrist_2":         lambda img: img[:-200,   200:-200],
        "side_policy":     lambda img: img[250:500,  350:650],
        "side_classifier": lambda img: img[270:398,  500:628],
    }

    # ------------------------------------------------------------------
    # [4] 关键位姿（Euler XYZ，单位：米 / 弧度）
    #   采集方法：
    #     conda activate hilserl
    #     cd serl_robot_infra/ur5e_env/utils
    #     python get_tcp_pose.py --robot_ip 192.168.1.100
    # ------------------------------------------------------------------

    # USB 完全插入 USB 口时的末端位姿
    TARGET_POSE = np.array([0.553, 0.177, 0.251, np.pi, 0.0, -np.pi / 2])

    # 每个 episode 开始时的末端位姿（比 TARGET_POSE 抬高 + 偏移一点）
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.03, 0.05, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # [5] 安全探索边界
    # ------------------------------------------------------------------
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.06, 0.05, 0.1, 0.1, 0.3])
    ABS_POSE_LIMIT_LOW  = TARGET_POSE - np.array([0.03, 0.01, 0.03, 0.1, 0.1, 0.3])

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
    RESET_JOINTS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

    # 夹爪
    GRIPPER_TYPE  = "robotiq"
    GRIPPER_SPEED = 0.5
    GRIPPER_FORCE = 0.5
    GRIPPER_SLEEP = 0.6

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
    # 策略观测使用的图像键
    image_keys      = ["side_policy", "wrist_1", "wrist_2"]
    # 奖励分类器使用的图像键（侧方相机裁剪更小的视图）
    classifier_keys = ["side_classifier"]
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
