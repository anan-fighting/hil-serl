#!/bin/bash
# ============================================================
#  UR5e USB Pick-Up & Insertion — LEARNER
#  可在另一台 GPU 机器上运行。
#  修改 --checkpoint_path 和 --demo_path 后再启动。
# ============================================================
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3

python ../../train_rlpd.py "$@" \
    --exp_name=ur5e_usb_pickup_insertion \
    --checkpoint_path=first_run \
    --demo_path=experiments/ur5e_usb_pickup_insertion/demo_data/demos.pkl \
    --learner
