#!/bin/bash

set -e  # 出错就停止

echo "===== Start Training teatime ====="
python train.py \
  -s /root/autodl-tmp/teatime \
  -r 1 \
  -m /root/autodl-tmp/output/teatime_v9 \
  --config_file config/gaussian_dataset/train.json \
  --train_split \
  --use_wandb

#echo "===== Open Voc Segmentation====="

#python render_lerf_mask_ours.py -m /root/autodl-tmp/output/ramen_ours_v3 --skip_train
#python render_lerf_mask_ours.py -m /root/autodl-tmp/output/figurines_ours --skip_train
#python render_lerf_mask_ours.py -m /root/autodl-tmp/output/teatime_ours --skip_train
#python script/eval_lerf_mask.py ramen
#python script/eval_lerf_mask.py figurines
#python script/eval_lerf_mask.py teatime


#python render_lerf_mask.py -m /root/autodl-tmp/official_output/ramen --skip_train
#python render_lerf_mask.py -m /root/autodl-tmp/official_output/teatime --skip_train
#python render_lerf_mask.py -m /root/autodl-tmp/official_output/figurines --skip_train
#python script/eval_lerf_mask_ours.py ramen
#python script/eval_lerf_mask.py teatime
#python script/eval_lerf_mask.py figurines


echo "===== Start Rendering ====="
python render.py \
  -m /root/autodl-tmp/output/teatime_ours \
  --eval \
  --skip_train \
  --iteration -1

echo "===== Compute Metrics ====="
python metrics.py -m /root/autodl-tmp/output/teatime_ours

echo "===== Segmentation Metrics ====="
python segmentation_metrics.py \
  -m /root/autodl-tmp/output/teatime_ours \
  --split test \
  --method ours_30000

echo "===== All Done ====="
