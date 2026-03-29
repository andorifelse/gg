#!/bin/bash

set -e  # 出错就停止

#echo "===== Start Training teatime ====="
#python train.py \
#  -s /root/autodl-tmp/teatime \
#  -r 1 \
#  -m /root/autodl-tmp/output/teatime_ours \
#  --config_file config/gaussian_dataset/train.json \
#  --train_split \
#  --use_wandb

echo "===== Open Voc Segmentation====="
python render_lerf_mask.py -m /root/autodl-tmp/output/ramen_ours --skip_train
python script/eval_lerf_mask.py ramen
python render_lerf_mask.py -m /root/autodl-tmp/output/figurines_ours --skip_train
python script/eval_lerf_mask.py figurines
#python render_lerf_mask.py -m /root/autodl-tmp/output/teatime_ours --skip_train
#python script/eval_lerf_mask.py teatime

#echo "===== Start Rendering ====="
#python render.py \
#  -m /root/autodl-tmp/output/ramen_v8 \
#  --eval \
#  --skip_train \
#  --iteration -1
#
#echo "===== Compute Metrics ====="
#python metrics.py -m /root/autodl-tmp/output/ramen_v8
#
#echo "===== Segmentation Metrics ====="
#python segmentation_metrics.py \
#  -m /root/autodl-tmp/output/ramen_v8 \
#  --split test \
#  --method ours_30000

echo "===== All Done ====="
