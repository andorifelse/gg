#!/bin/bash

set -e  # 出错就停止

echo "===== Start Training ====="
python train.py \
  -s /root/autodl-tmp/ramen \
  -r 1 \
  -m /root/autodl-tmp/output/ramen_v8 \
  --config_file config/gaussian_dataset/train.json \
  --eval \
  --use_wandb

echo "===== Start Rendering ====="
python render.py \
  -m /root/autodl-tmp/output/ramen_v8 \
  --eval \
  --skip_train \
  --iteration -1

echo "===== Compute Metrics ====="
python metrics.py -m /root/autodl-tmp/output/ramen_v8

echo "===== Segmentation Metrics ====="
python segmentation_metrics.py \
  -m /root/autodl-tmp/output/ramen_v8 \
  --split test \
  --method ours_30000

echo "===== All Done ====="
