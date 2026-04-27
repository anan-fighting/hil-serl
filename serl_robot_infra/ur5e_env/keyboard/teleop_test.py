#!/usr/bin/env python3
"""
独立键盘遥操测试脚本 —— 不依赖 Gym 环境，直接通过 RTDE 控制 UR5e。

功能：
  - 以 10 Hz 控制频率循环读取键盘输入
  - 将键盘动作叠加到当前 TCP 位姿并发送给机器人
  - 实时打印当前位姿、力/力矩、夹爪状态
  - ESC 退出

用法：
    conda activate hilserl
    cd serl_robot_infra
    python -m ur5e_env.keyboard.teleop_test --robot_ip 192.168.1.103

参数说明：
    --robot_ip       UR5e 的 IP 地址（默认 192.168.1.103）
    --hz             控制频率，Hz（默认 10）
    --linear_speed   平移速度缩放（默认 0.005，单位 m/step）
    --angular_speed  旋转速度缩放（默认 0.03，单位 rad/step）
    --gripper        启用 Robotiq 夹爪控制（默认不启用）
    --no_camera      跳过相机初始化（默认跳过，纯运动测试）
"""

import argparse
import time

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation

from ur5e_env.keyboard.keyboard_expert import KeyboardExpert
from ur5e_env.utils.rotations import pose_rotvec_to_quat, pose_quat_to_rotvec


# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="UR5e Keyboard Teleop Test")
    p.add_argument("--robot_ip",       default="192.168.1.103")
    p.add_argument("--hz",             type=float, default=10.0)
    p.add_argument("--linear_speed",   type=float, default=0.003,
                   help="平移速度（m/step），建议 0.003 ~ 0.01")
    p.add_argument("--angular_speed",  type=float, default=0.02,
                   help="旋转速度（rad/step），建议 0.01 ~ 0.05")
    p.add_argument("--gripper",        default=True,
                   help="启用 Robotiq 夹爪控制")
    p.add_argument("--gripper_port",   default="/dev/ttyUSB0",
                   help="Robotiq Modbus RTU 串口设备（默认 /dev/ttyUSB0）")
    p.add_argument("--servo_lookahead", type=float, default=0.1)
    p.add_argument("--servo_gain",      type=float, default=300)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
def print_status(tcp_rv, ft, gripper_closed):
    """在同一行刷新打印状态。"""
    pos = np.round(tcp_rv[:3], 4)
    rot = np.round(tcp_rv[3:], 4)
    f   = np.round(ft[:3], 2)
    t   = np.round(ft[3:], 2)
    g   = "CLOSED" if gripper_closed else "OPEN  "
    print(
        f"\r  pos=[{pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}]  "
        f"rot=[{rot[0]:+.4f},{rot[1]:+.4f},{rot[2]:+.4f}]  "
        f"F=[{f[0]:+6.2f},{f[1]:+6.2f},{f[2]:+6.2f}]N  "
        f"gripper={g}",
        end="", flush=True,
    )


# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    dt = 1.0 / args.hz

    # ── 连接机器人 ──────────────────────────────────────────
    print(f"[Teleop] 正在连接 UR5e @ {args.robot_ip} …")
    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    print("[Teleop] RTDE 连接成功。")

    # ── 夹爪 ────────────────────────────────────────────────
    gripper = None
    gripper_closed = False
    if args.gripper:
        try:
            from ur5e_env.keyboard.robotiq.HandE import HandEForRtu
            print(f"[Teleop] 正在初始化 Robotiq 夹爪 @ {args.gripper_port} …")
            gripper = HandEForRtu(args.gripper_port, autoInit=True)
            gripper.move(50.0, 150, 0, block=True)   # 打开至 50 mm
            time.sleep(0.5)
            print("[Teleop] Robotiq 夹爪已就绪（已打开）。")
        except Exception as e:
            print(f"[Teleop] 夹爪初始化失败（{e}），将跳过夹爪控制。")
            gripper = None

    # ── 键盘 ────────────────────────────────────────────────
    expert = KeyboardExpert(
        linear_speed=1.0,   # 速度缩放在下面手动应用，这里保持 1.0
        angular_speed=1.0,
    )

    # ── 获取初始位姿（rotvec，UR 原生格式）──────────────────
    curr_rv = np.array(rtde_r.getActualTCPPose(), dtype=np.float64)
    print(f"\n[Teleop] 当前位姿（rotvec）: {np.round(curr_rv, 4)}")
    print("[Teleop] 开始键盘遥操，ESC 退出。\n")

    # ── 主控制循环 ──────────────────────────────────────────
    try:
        while True:
            t0 = time.time()

            # 检查退出
            if expert.terminate:
                print("\n[Teleop] 检测到 ESC，退出。")
                break

            # 读取键盘动作  [dx, dy, dz, drx, dry, drz], buttons
            raw_action, buttons = expert.get_action()
            close_btn, open_btn = bool(buttons[0]), bool(buttons[1])

            # 叠加平移（世界坐标系，单位：m）
            delta_pos    = raw_action[:3] * args.linear_speed
            delta_rotvec = raw_action[3:] * args.angular_speed

            # 当前位姿 → 四元数空间做旋转叠加
            curr_quat = pose_rotvec_to_quat(curr_rv)           # [x,y,z,qx,qy,qz,qw]
            next_pos  = curr_quat[:3] + delta_pos
            next_quat_rot = (
                Rotation.from_rotvec(delta_rotvec)
                * Rotation.from_quat(curr_quat[3:])
            ).as_quat()
            next_rv = pose_quat_to_rotvec(
                np.concatenate([next_pos, next_quat_rot])
            )

            # 发送 servoL 命令（全部位置参数，ur-rtde 不支持关键字参数）
            if np.linalg.norm(raw_action) > 1e-6:
                rtde_c.servoL(
                    list(next_rv.astype(float)),
                    0.5,                     # velocity
                    0.3,                     # acceleration
                    0.002,                   # dt（UR 控制器 500 Hz）
                    args.servo_lookahead,    # lookahead_time
                    args.servo_gain,         # gain
                )

            # 夹爪控制
            if gripper is not None:
                if close_btn and not gripper_closed:
                    gripper.move(0.0, 150, 0, block=False)   # 关闭（0 mm）
                    gripper_closed = True
                elif open_btn and gripper_closed:
                    gripper.move(50.0, 150, 0, block=False)  # 打开（50 mm）
                    gripper_closed = False

            # 读取真实位姿（用于下一步叠加，避免漂移累积）
            curr_rv = np.array(rtde_r.getActualTCPPose(), dtype=np.float64)

            # 读取力/力矩
            ft = np.array(rtde_r.getActualTCPForce(), dtype=np.float64)

            # 打印状态（若有夹爪，显示真实位置 mm）
            if gripper is not None:
                try:
                    g_str = f"{gripper.position:.1f}mm"
                except Exception:
                    g_str = "CLOSED" if gripper_closed else "OPEN  "
            else:
                g_str = "N/A"
            pos = np.round(curr_rv[:3], 4)
            rot = np.round(curr_rv[3:], 4)
            f_  = np.round(ft[:3], 2)
            print(
                f"\r  pos=[{pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}]  "
                f"rot=[{rot[0]:+.4f},{rot[1]:+.4f},{rot[2]:+.4f}]  "
                f"F=[{f_[0]:+6.2f},{f_[1]:+6.2f},{f_[2]:+6.2f}]N  "
                f"gripper={g_str}   ",
                end="", flush=True,
            )

            # 控制频率等待
            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\n[Teleop] Ctrl+C，退出。")

    finally:
        print("\n[Teleop] 停止 servoL …")
        try:
            rtde_c.servoStop()
            rtde_c.stopScript()
        except Exception:
            pass
        expert.close()
        print("[Teleop] 已断开。")


if __name__ == "__main__":
    main()
