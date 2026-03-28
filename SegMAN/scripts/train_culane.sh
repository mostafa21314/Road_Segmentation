#!/bin/bash

# ============================================================
# Train SegMAN-Small on CULane Rural Subset
# ============================================================
# Step 1: Run the data preparation script first (only needed once):
#   cd /home/g6/Mostafa/Road_Segmentation/SegMAN
#   python convert_culane_to_masks.py
#
# Step 2: Then run this training script:
#   cd /home/g6/Mostafa/Road_Segmentation/SegMAN/segmentation
#   bash ../scripts/train_culane.sh
# ============================================================

CONFIG="local_configs/segman/small/segman_s_culane.py"
WORK_DIR="work_dirs/segman_s_culane"

# Single GPU training (no distributed launch needed for 1 GPU)
python tools/train.py \
    ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --gpu-id 0 \
    --seed 15
