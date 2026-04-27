# UR5e + 键盘 HIL-SERL 完整复现指南（USB Pick-Up & Insertion）

本文档描述如何在 **UR5e 机械臂**（自带末端力传感器）上使用 **RTDE** 直接控制（无需 ROS），并以**键盘**代替 SpaceMouse 进行人工干预，完整运行 **USB 拾取与插入（USB Pick-Up & Insertion）** 任务的 HIL-SERL 训练流程。

---

## 目录

1. [新增代码文件结构](#1-新增代码文件结构)
2. [环境安装](#2-环境安装)
3. [硬件准备与网络配置](#3-硬件准备与网络配置)
4. [采集关键位姿](#4-采集关键位姿)
5. [修改任务配置文件](#5-修改任务配置文件)
6. [训练奖励分类器](#6-训练奖励分类器)
7. [录制人工演示](#7-录制人工演示)
8. [策略训练（HIL-SERL）](#8-策略训练hil-serl)
9. [策略评估](#9-策略评估)
10. [键盘操控说明](#10-键盘操控说明)
11. [常见问题排查](#11-常见问题排查)
12. [自定义新任务](#12-自定义新任务)

---

## 1. 新增代码文件结构

以下是本次为 UR5e 新增的全部文件（**原 Franka 代码未做任何修改**）：

```
serl_robot_infra/
└── ur5e_env/                          ← 新增包（与 franka_env 并列）
    ├── __init__.py
    ├── setup.py
    ├── envs/
    │   ├── __init__.py
    │   ├── ur5e_env.py                ← 核心 Gym 环境（RTDE 直连，无 ROS）
    │   └── wrappers.py                ← Gym wrappers（含 KeyboardIntervention）
    ├── keyboard/
    │   ├── __init__.py
    │   └── keyboard_expert.py         ← 键盘遥操模块（替代 SpaceMouse）
    │   └── teleop_test.py             ← 键盘操控测试脚本
    └── utils/
        ├── __init__.py
        ├── rotations.py               ← Euler/Quat/Rotvec 转换工具
        └── get_tcp_pose.py            ← 读取当前 TCP 位姿的辅助脚本

examples/experiments/
└── ur5e_usb_pickup_insertion/         ← USB 拾取与插入任务
    ├── __init__.py
    ├── config.py                      ← 任务配置（IP、相机序列号、位姿等）
    ├── wrapper.py                     ← UR5eUSBEnv + GripperPenaltyWrapper
    ├── run_actor.sh                   ← 启动 Actor 节点
    └── run_learner.sh                 ← 启动 Learner 节点
```

> 如需适配自己的任务（非 USB 插入），复制 `ur5e_usb_pickup_insertion/` 目录并按
> [第 12 节](#12-自定义新任务) 说明修改即可。

---

## 2. 环境安装

### 2.1 Conda 环境

```bash
conda create -n hilserl python=3.10
conda activate hilserl
```

### 2.2 安装 JAX（GPU）

```bash
pip install --upgrade "jax[cuda12_pip]==0.4.35" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 2.3 安装 serl_launcher

```bash
cd serl_launcher
pip install -e .
pip install -r requirements.txt
```

### 2.4 安装 serl_robot_infra（含原 Franka 包）

```bash
cd serl_robot_infra
pip install -e .
```

### 2.5 安装 ur5e_env

```bash
cd serl_robot_infra/ur5e_env
pip install -e .
```

这一步会自动安装以下关键依赖：

| 包名 | 用途 |
|------|------|
| `ur-rtde` | UR5e RTDE Python 绑定（`rtde_control` / `rtde_receive`） |
| `pynput` | 键盘监听（替代 SpaceMouse） |
| `gymnasium` | Gym 环境框架 |
| `opencv-python` | 相机图像处理 |
| `scipy` | 旋转计算 |

> **注意**：`ur-rtde` 需要 **Ubuntu 20.04/22.04 x86_64**，并且安装时可能需要：
> ```bash
> sudo apt-get install -y build-essential cmake libboost-all-dev
> pip install ur-rtde
> ```

---

## 3. 硬件准备与网络配置

### 3.1 UR5e 网络设置

1. 在示教器：`设置 → 系统 → 网络` 中为 UR5e 分配固定 IP（例如 `192.168.1.103`）。
2. 将电脑网卡设置为同一网段（例如 `192.168.1.10/24`）。
3. 测试连通性：`ping 192.168.1.103`

### 3.2 开启 RTDE 远程控制

在 UR5e 示教器上：
1. `汉堡菜单 → 设置 → 系统 → 远程控制`
2. 打开**远程控制**开关（Remote Control Enable）
3. 确保机器人处于**远程模式**（状态灯为蓝色）

> ⚠️ **安全提示**：在切换到远程模式前，确保机械臂附近无人，急停按钮随时可达。

### 3.3 Robotiq 2F 夹爪（可选）

- 通过 RS-485 转 USB 连接到控制箱 Tool I/O，或直接接 UR 工具端口。
- UR5e 内置 RTDE 夹爪通信（通过 `ur-rtde` 的 `GripperSocketClient` 支持）。
- 如果使用其他夹爪类型，修改 `EnvConfig.GRIPPER_TYPE = "none"` 并重写 `_send_gripper_command()`。

### 3.4 RealSense 相机

- 安装腕部相机后，在 **RealSense Viewer** 中查看并记录序列号。
- 调整 `IMAGE_CROP` 裁剪参数（运行 `record_success_fail.py` 时可预览图像）。

---

## 4. 采集关键位姿

所有关键位姿（`TARGET_POSE`、`RESET_POSE`、`ABS_POSE_LIMIT_*`）均以 **旋转矢量 rotvec（UR 原生格式）** 存储：

```
[x, y, z, rx, ry, rz]
```

其中 `(rx, ry, rz)` 是旋转轴与旋转角度的乘积（axis-angle），即 `rtde_receive.getActualTCPPose()` 直接返回的格式，**无需任何转换**即可填入配置文件。

### 4.1 进入 FreeDrive 模式

```python
# 在 Python 中临时开启 FreeDrive（或用示教器面板的 FreeDrive 按钮）
import rtde_control
rtde_c = rtde_control.RTDEControlInterface("192.168.1.103")
rtde_c.teachMode()     # 进入 FreeDrive
# 手动移动机械臂到目标位置…
rtde_c.endTeachMode()  # 退出 FreeDrive
```

### 4.2 读取当前位姿

```bash
conda activate hilserl
cd serl_robot_infra/ur5e_env/utils
python get_tcp_pose.py --robot_ip 192.168.1.103
```

输出示例：
```
=== UR5e Current TCP Pose ===
  Rotvec (UR native) [x,y,z,rx,ry,rz]: [0.4, -0.1, 0.2, 3.1416, 0.0, 0.0]
  Joint angles (rad)                  : [-1.57, -1.57, 1.57, -1.57, -1.57, 0.0]

Copy into your config.py as:
  TARGET_POSE = np.array([0.4, -0.1, 0.2, 3.1416, 0.0, 0.0])
```

### 4.3 需要采集的关键位姿

| 配置字段 | 含义 | 采集方法 |
|----------|------|----------|
| `TARGET_POSE` | 任务成功时末端位姿 | 手动引导至目标点后读取 |
| `RESET_POSE` | 每个 episode 开始位姿 | 手动引导至复位点后读取 |
| `RESET_JOINTS` | 关节空间 home 位（用于关节复位） | 从 `get_tcp_pose.py` 输出的 Joint angles 填入 |
| `ABS_POSE_LIMIT_*` | 安全边界 | 在 `TARGET_POSE` 基础上加减偏移量 |

---

## 5. 修改任务配置文件

编辑 `examples/experiments/ur5e_pick_place/config.py`，按注释逐项修改：

```python
class EnvConfig(DefaultUR5eEnvConfig):
    ROBOT_IP = "192.168.1.103"       # ← 改为你的机器人 IP

    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "XXXXXXXXXXXXXXX",  # ← 改为你的相机序列号
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }

    IMAGE_CROP = {
        "wrist_1": lambda img: img[100:500, 300:1000],  # ← 根据实际调整
    }

    TARGET_POSE = np.array([...])   # ← 步骤 4 中采集
    RESET_POSE  = np.array([...])   # ← 步骤 4 中采集

    ABS_POSE_LIMIT_LOW  = TARGET_POSE - np.array([0.10, 0.10, 0.05, 0.2, 0.2, 0.5])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.10, 0.10, 0.15, 0.2, 0.2, 0.5])

    RESET_JOINTS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])  # ← 步骤 4 中采集
```

---

## 6. 训练奖励分类器

### 6.1 采集分类器数据

```bash
conda activate hilserl
cd examples
python record_success_fail.py --exp_name ur5e_usb_pickup_insertion --successes_needed 200
```

**操作方法**（键盘控制）：

| 按键 | 效果 |
|------|------|
| W/S/A/D/Q/E | 移动末端（平移） |
| I/K/J/L/U/O | 旋转末端 |
| F（按住） | 关闭夹爪 |
| G（按住） | 打开夹爪 |
| **Space（按住）** | 当前帧标记为**正样本（成功）** |
| ESC | 终止当前 episode |

- 默认所有帧为负样本（失败状态）
- **按住空格键**时录制的帧标记为正样本
- 脚本在收集到足够正样本（`--successes_needed`）后自动终止
- 数据保存至 `experiments/ur5e_usb_pickup_insertion/classifier_data/`

> **建议**：收集 2~3 倍的负样本以覆盖所有失败模式。

### 6.2 训练分类器

```bash
cd examples/experiments/ur5e_usb_pickup_insertion
python ../../train_reward_classifier.py --exp_name ur5e_usb_pickup_insertion
```

- 模型保存至 `experiments/ur5e_usb_pickup_insertion/classifier_ckpt/`

---

## 7. 录制人工演示

```bash
cd examples
python record_demos.py --exp_name ur5e_usb_pickup_insertion --successes_needed 20
```

- 使用**键盘**遥控机械臂完成 USB 拾取并插入（见[第 10 节](#10-键盘操控说明)）
- 奖励分类器判断成功 **或** episode 超时后，机械臂自动执行复位：打开夹爪 → 下移到 USB 口 → 夹住 USB → 抬回 RESET_POSE
- 收集到 20 条成功演示后自动结束
- 演示数据保存至 `experiments/ur5e_usb_pickup_insertion/demo_data/`

---

## 8. 策略训练（HIL-SERL）

### 8.1 修改训练脚本

**`run_actor.sh`**（无需修改，直接使用默认值）：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
python ../../train_rlpd.py \
    --exp_name=ur5e_usb_pickup_insertion \
    --checkpoint_path=/path/to/your/checkpoints/run1 \
    --actor
```

**`run_learner.sh`**（修改 `--demo_path` 和 `--checkpoint_path`）：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
python ../../train_rlpd.py \
    --exp_name=ur5e_usb_pickup_insertion \
    --checkpoint_path=/path/to/your/checkpoints/run1 \
    --demo_path=/path/to/demo_data/demos.pkl \
    --learner
```

### 8.2 启动训练（两个终端）

```bash
# 终端 1 ── Actor（在机器人旁，需要使用键盘）
cd examples/experiments/ur5e_usb_pickup_insertion
bash run_actor.sh

# 终端 2 ── Learner（可在另一台 GPU 机器上）
cd examples/experiments/ur5e_usb_pickup_insertion
bash run_learner.sh
```

若 Learner 在**另一台机器**上运行，还需在 Actor 端指定 Learner 的 IP：

```bash
bash run_actor.sh --ip <LEARNER_MACHINE_IP>
```

### 8.3 训练过程中的人工干预

键盘干预策略（参考 Franka 训练经验）：

| 训练阶段 | 建议干预频率 |
|----------|------------|
| 训练初期（前 100 episode） | 频繁介入，每 20~30 步引导一次，帮助策略获得奖励 |
| 中期策略有改善后 | 仅当策略重复错误行为时介入 |
| 后期接近收敛 | 几乎不介入，仅纠正边缘案例 |

干预动作自动存入 replay buffer 的 `intervene_action` 字段，被训练算法识别为人类示范。

---

## 9. 策略评估

```bash
cd examples/experiments/ur5e_usb_pickup_insertion
bash run_actor.sh \
    --eval_checkpoint_step=5000 \
    --eval_n_trajs=20
```

---

## 10. 键盘操控说明

```
┌─────────────────────────────────────────┐
│         Keyboard Teleoperation           │
├──────────────┬──────────────────────────┤
│  W / S       │  +X / -X  (前进/后退)    │
│  A / D       │  +Y / -Y  (左移/右移)    │
│  Q / E       │  +Z / -Z  (上升/下降)    │
│  I / K       │  +Rx / -Rx (绕 X 旋转)  │
│  J / L       │  +Ry / -Ry (绕 Y 旋转)  │
│  U / O       │  +Rz / -Rz (绕 Z 旋转)  │
│  F (按住)    │  关闭夹爪               │
│  G (按住)    │  打开夹爪               │
│  ESC         │  紧急停止               │
└──────────────┴──────────────────────────┘
```

**速度调整**：在 `config.py` 中修改：

```python
KEYBOARD_LINEAR_SPEED  = 1.0   # 平移速度倍率
KEYBOARD_ANGULAR_SPEED = 1.0   # 旋转速度倍率
```

> ⚠️ **注意**：键盘输入需要终端窗口保持焦点（在运行脚本的终端窗口内操作）。

---

## 11. 常见问题排查

### Q1：连接 UR5e 失败 `RTDEControlInterface failed to connect`

- 检查网络 ping 是否通：`ping 192.168.1.103`
- 确认 UR5e 开启了**远程控制**模式
- 确认没有其他 RTDE 客户端占用连接（UR5e 仅支持有限并发连接数）
- 检查防火墙：`sudo ufw allow 30002,30004/tcp`

### Q2：`servoL` 运动不平滑或抖动

- 降低 `ACTION_SCALE`，例如 `ACTION_SCALE = (0.005, 0.03, 1.0)`
- 增大 `SERVO_LOOKAHEAD_TIME`（0.1 → 0.15）
- 降低 `hz`（10 → 8）

### Q3：力传感器数据全为零

- UR5e 内置 F/T 传感器需要在示教器中**启用**：`设置 → 安装 → 力控制`
- 确认 RTDE 接收接口版本 ≥ 1.4

### Q4：键盘无响应

- 确保运行脚本的**终端窗口处于前台焦点**（鼠标点击终端窗口）
- 若在 SSH 远程连接中运行，需使用本地终端，或改用 `--display :0 python …` 传递 X11 显示

### Q5：相机图像冻结

- 检查 USB 连接是否稳定（建议使用 USB 3.0 直接接口，不经 Hub）
- 降低 `exposure` 值（从 40000 降到 10000）

### Q6：保护性停止（Protective Stop）

```bash
# 代码内自动调用，也可手动：
import rtde_control
rtde_c = rtde_control.RTDEControlInterface("192.168.1.103")
rtde_c.unlockProtectiveStop()
```

---

## 12. 自定义新任务

以 `ur5e_pick_place` 为模板，创建自己的任务只需：

### Step 1：复制任务目录

```bash
cp -r examples/experiments/ur5e_usb_pickup_insertion examples/experiments/ur5e_my_task
```

### Step 2：修改 `config.py`

- 更新 `ROBOT_IP`、相机序列号、所有关键位姿
- 修改 `TrainConfig.image_keys` / `classifier_keys` / `proprio_keys`
- 调整 `setup_mode`：
  - `"single-arm-learned-gripper"` — 策略控制夹爪
  - `"single-arm-fixed-gripper"` — 夹爪保持关闭（加 `GripperCloseEnv` wrapper）

### Step 3：修改 `wrapper.py`（可选）

重写 `go_to_reset()` 以实现任务专属的复位流程（如多步抓取复位）。

### Step 4：注册到 mappings.py

```python
# examples/experiments/mappings.py
from experiments.ur5e_my_task.config import TrainConfig as UR5eMyTaskTrainConfig

CONFIG_MAPPING = {
    ...
    "ur5e_usb_pickup_insertion": UR5eUSBPickupInsertionTrainConfig,
    "ur5e_my_task": UR5eMyTaskTrainConfig,
}
```

### Step 5：运行

```bash
# 分类器数据采集
python record_success_fail.py --exp_name ur5e_my_task --successes_needed 200

# 分类器训练
cd experiments/ur5e_my_task && python ../../train_reward_classifier.py --exp_name ur5e_my_task

# 演示录制
python record_demos.py --exp_name ur5e_my_task --successes_needed 20

# 策略训练
bash run_actor.sh
bash run_learner.sh
```

---

## 附录：RTDE vs ROS 控制对比

| 特性 | RTDE（本文档） | ROS（Franka 方案） |
|------|-------------|-----------------|
| 实时性 | ≤ 2ms 控制周期 | 依赖 ROS 网络延迟 |
| 依赖 | `ur-rtde` Python 包 | ROS + franka_ros |
| 部署复杂度 | 低（pip install） | 高（需编译 ROS 包） |
| 阻抗控制 | 通过 `servoL` 实现软顺应 | 专用阻抗控制器 |
| 力传感器 | 内置（通过 RTDE 读取） | 需单独 ROS 节点 |
