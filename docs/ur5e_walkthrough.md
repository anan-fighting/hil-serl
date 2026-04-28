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

```bash
(hilserl) zzw@zzw-ThinkBook:~/project/hil-serl/examples$ python record_success_fail.py --exp_name ur5e_usb_pickup_insertion --successes_needed 200
```
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

### 6.1 【目的】为什么需要奖励分类器？

强化学习需要一个"奖励信号"来告诉策略什么是成功、什么是失败。

USB 插入任务的成功状态（USB 完全插入孔位）**很难用简单的位置阈值判断**（因为不同角度、不同深度都可能看起来相似），所以我们用一个**基于图像的分类器**来判断：

> 分类器输入：腕部相机当前帧图像  
> 分类器输出：`成功（1）` 或 `失败（0）`

这个分类器在训练阶段实时运行，每一步给出奖励信号。

---

### 6.2 【操作】采集分类器数据

这一步的目标：**收集"USB 插好了"和"USB 没插好"的相机图像各若干帧**，用于训练分类器。

**你需要做的事**：用键盘遥控机械臂，让末端在 USB 口附近移动，同时手动标记哪些帧是"成功状态"。

#### 为什么必须用 `--skip_grasp` + 手动放入 USB？

`reset()` 的完整抓取流程（Step 5）是从 `TARGET_POSE` **线性插值**移动到 `RESET_POSE`，XYZ 三轴同时运动。如果此时 USB 还插在孔里，机械臂会**斜向拔出**，可能碰撞端口或损坏 USB。

`reset()` 在**训练/演示录制**时没有此问题，因为 Step 1 会先开夹爪放开 USB，USB 自由落下后再抬起。

但在**分类器数据采集**阶段，USB 始终被夹持插入，因此必须跳过自动抓取流程，改为手动放入。

---

#### 操作流程

**第一步**：加 `--skip_grasp` 运行脚本，机械臂移到 RESET_POSE（空手）：

```bash
conda activate hilserl
cd examples
python record_success_fail.py \
    --exp_name ur5e_usb_pickup_insertion \
    --successes_needed 200 \
    --skip_grasp
```

**第二步**：脚本进入采集循环后，**手动将 USB 放入夹爪之间，然后按住键盘 `F` 键关闭夹爪夹住 USB**。

> ⚠️ 注意：此时终端需要保持焦点（鼠标点击终端窗口），F 键才能被识别。

**第三步**：夹好 USB 后，用键盘操控机械臂将 USB 插入孔位：

- 按 `E` 键下降 + `W/S/A/D` 平移对准孔位并插入
- USB 完全插入后，**按一下 Space 空格键**，记录一帧正样本
- 可连续按几下空格，多采几帧
- 按 `Q` 键上升，**先竖直抬起**再平移，避免斜向拔出碰撞
- 如此反复：插入 → 按空格 → 竖直抬起 → 再插入 → 按空格……

> **注意**：空格是**按一下触发一次**。**只在 USB 确认插好时按**；其余所有帧自动记为负样本。

**第四步**：收集到 200 帧正样本后脚本自动停止，数据保存至：

```
examples/classifier_data/ur5e_usb_pickup_insertion_200_success_images_<时间戳>.pkl
examples/classifier_data/ur5e_usb_pickup_insertion_failure_images_<时间戳>.pkl
```

#### 采集技巧

| 建议 | 说明 |
|------|------|
| 正样本多样性 | 在略有偏差的成功位置多按几次，让分类器对轻微偏差也鲁棒 |
| 负样本覆盖全面 | 在不同失败姿态停留（太高、太偏、未插入），让负样本分布多样 |
| 负样本数量 | 建议负样本是正样本的 3~5 倍（脚本自动累积所有非空格帧） |
| 光照一致性 | 采集环境的光照应与后续训练时一致 |

---

### 6.3 训练分类器

```bash
cd examples
python train_reward_classifier.py --exp_name ur5e_usb_pickup_insertion
```

- 分类器模型保存至 `experiments/ur5e_usb_pickup_insertion/classifier_ckpt/`
- 训练完成后，该模型会在 `record_demos.py` 和 `train_rlpd.py` 中自动加载，实时判断任务是否成功并给出奖励

---

## 7. 录制人工演示

### 7.1 【目的】为什么需要人工演示？

HIL-SERL 使用 **RLPD（Reinforcement Learning from Prior Data）** 算法，该算法可以利用人工演示数据加速学习：

- 演示数据放入 **专用 demo replay buffer**，每次梯度更新有 50% 概率从中采样
- 从零强化学习可能需要数千次 episode，有了演示数据通常 **100~300 次 episode 内就能收敛**
- 演示数据同时为分类器提供了真实的成功轨迹分布参考

---

### 7.2 【操作】录制演示

```bash
cd examples
python record_demos.py --exp_name ur5e_usb_pickup_insertion --successes_needed 20
```

**这一步你需要完整手动完成整个任务**：机械臂从 RESET_POSE 出发，你用键盘遥控，将 USB 插入孔位，分类器判断成功后该条轨迹被保存，机械臂自动复位，进入下一条演示录制。

#### 录制流程（每条演示）

```
[机械臂在 RESET_POSE]
        ↓
  键盘遥控，引导末端靠近 USB 口
        ↓
  夹爪对准 → 按 E 下降 → USB 插入
        ↓
  分类器检测到成功（rew=1）→ done=True
        ↓
  本条轨迹所有帧自动保存
        ↓
  env.reset() 自动复位：开夹爪 → 下移 → 夹 USB → 抬回 RESET_POSE
        ↓
  继续录制下一条
```

- 只有**分类器判断成功的完整轨迹**才会被保存（`info["succeed"]=True`）
- 失败或超时的轨迹自动丢弃，重新开始
- 收集到 20 条成功演示后自动结束
- 数据保存至 `examples/demo_data/ur5e_usb_pickup_insertion_20_demos_<时间戳>.pkl`

> **建议**：每条演示尽量操作干净流畅，避免大幅摆动。20 条高质量演示优于 50 条质量差的演示。

---

## 8. 策略训练（HIL-SERL）

### 8.1 【整体架构】Actor + Learner 是什么？

HIL-SERL 将训练分为两个并行进程：

```
┌─────────────────────────────────┐        ┌────────────────────────────────┐
│           ACTOR 进程             │        │          LEARNER 进程           │
│  （必须连接机器人，在机器人旁）    │        │  （只需 CPU/GPU，可在另一台机器） │
│                                 │        │                                │
│  1. 用当前策略控制机械臂执行任务  │──数据──▶│  1. 从 replay buffer 采样      │
│  2. 收集 (obs, action, reward)  │        │  2. 计算梯度，更新网络参数       │
│  3. 存入 replay buffer          │◀─参数──│  3. 推送最新网络参数给 Actor    │
│  4. 键盘干预时记录人工动作       │        │  4. 同时利用 demo data 加速训练  │
└─────────────────────────────────┘        └────────────────────────────────┘
           ↑ agentlace 网络通信（默认 localhost，可跨机器）
```

**可以跑在同一台机器上**（使用 `--ip localhost`，默认就是），开两个终端即可。如果有独立 GPU 机器可以把 Learner 单独放过去。

---

### 8.2 修改启动脚本路径

编辑 `run_learner.sh`，将 `--demo_path` 改为第 7 步实际生成的文件：

```bash
# run_learner.sh 中修改这一行：
--demo_path=demo_data/ur5e_usb_pickup_insertion_20_demos_<时间戳>.pkl
```

如需自定义 checkpoint 保存路径，同时修改两个脚本的 `--checkpoint_path`：

```bash
--checkpoint_path=/home/zzw/project/hil-serl/checkpoints/run1
```

---

### 8.3 启动训练

**先启动 Learner，再启动 Actor**（Learner 要先建好参数服务器）：

```bash
# 终端 1 ── 先启动 Learner
cd examples/experiments/ur5e_usb_pickup_insertion
bash run_learner.sh

# 终端 2 ── 再启动 Actor（机器人旁，需要键盘焦点）
cd examples/experiments/ur5e_usb_pickup_insertion
bash run_actor.sh
```

若 Learner 在**另一台机器**上，在 Actor 端指定 IP：

```bash
bash run_actor.sh --ip <LEARNER_IP>
```

---

### 8.4 训练阶段的完整流程

```
[启动 Learner] → 加载 demo 数据，初始化 replay buffer，建立参数服务
        ↓
[启动 Actor]  → 连接机器人，加载分类器，从 Learner 拉取初始参数
        ↓
[Episode 开始] → env.reset() 机械臂回到 RESET_POSE
        ↓
[每一步 step]：
  ① Actor 用当前策略预测动作
  ② 机械臂执行
  ③ 分类器给出奖励（success=1 / fail=0）
  ④ 数据推送给 Learner 的 replay buffer
  ⑤ 如果你按键盘干预，干预动作替换策略动作，记为 intervene_action 存入 buffer
        ↓
[done=True（成功 or 超时）] → env.reset() 复位 → 下一个 episode
        ↓
[Learner 持续后台训练]：
  每 N 步更新一次网络 → 推送最新参数给 Actor
        ↓
[重复，直到策略收敛]
```

---

### 8.5 训练过程中的人工干预策略

训练阶段你坐在机器人旁边，随时可以用键盘干预。干预的动作会替代策略动作并存入 replay buffer，被算法当作高质量示范。

| 训练阶段 | 建议干预频率 | 目的 |
|----------|------------|------|
| 初期（前 50 episode） | 高频干预，每次任务基本靠你完成 | 让 replay buffer 里充满成功轨迹，确保分类器能给出奖励信号 |
| 中期（50~200 episode） | 策略出现明显错误时才介入 | 纠正策略的系统性错误，避免策略陷入局部最优 |
| 后期（200+ episode） | 几乎不介入 | 验证策略是否真正自主学会任务 |

> **核心原则**：初期不干预策略可能永远拿不到奖励（永远失败 → 永远学不到）；后期过度干预会让策略依赖人类，无法真正自主。

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
