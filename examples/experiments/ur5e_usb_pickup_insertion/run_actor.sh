#!/bin/bash
# ============================================================
#  UR5e USB Pick-Up & Insertion — ACTOR
#  在连接机器人的机器上运行，训练时需坐在旁边随时键盘干预。
# ============================================================
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1

python ../../train_rlpd.py "$@" \
    --exp_name=ur5e_usb_pickup_insertion \
    --checkpoint_path=first_run \
    --actor
