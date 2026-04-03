#!/bin/bash

set -e  # 出错就停止

#echo "===== Start Training ====="
#python train.py \
#  -s /root/autodl-tmp/teatime \
#  -r 1 \
#  -m /root/autodl-tmp/output/teatime_v9 \
#  --config_file config/gaussian_dataset/train.json \
#  --eval \
#  --use_wandb
#
#echo "===== Start Rendering ====="
#python render.py \
#  -m /root/autodl-tmp/output/teatime_v9 \
#  --eval \
#  --skip_train \
#  --iteration -1
#
#echo "===== Compute Metrics ====="
#python metrics.py -m /root/autodl-tmp/output/teatime_v9
#
#echo "===== Segmentation Metrics ====="
#python segmentation_metrics.py \
#  -m /root/autodl-tmp/output/teatime_v9 \
#  --split test \
#  --method ours_30000
#
#echo "===== All Done ====="


#echo "===== Open Voc Segmentation====="

# original training + original Voc Segmentation
#python render_lerf_mask.py -m /root/autodl-tmp/official_output/ramen --skip_train
#python render_lerf_mask.py -m /root/autodl-tmp/official_output/teatime --skip_train
#python render_lerf_mask.py -m /root/autodl-tmp/official_output/figurines --skip_train

# Our training + original Voc Segmentation
#python render_lerf_mask.py -m /root/autodl-tmp/output/ramen_ours_v3 --skip_train
#python render_lerf_mask.py -m /root/autodl-tmp/output/figurines_ours --skip_train
#python render_lerf_mask.py -m /root/autodl-tmp/output/teatime_ours --skip_train

# Our training + Our Voc Segmentation v1 (change render script)
#python render_lerf_mask_ours.py -m /root/autodl-tmp/output/ramen_ours_v3 --skip_train
#python render_lerf_mask_ours.py -m /root/autodl-tmp/output/figurines_ours --skip_train
#python render_lerf_mask_ours.py -m /root/autodl-tmp/output/teatime_ours --skip_train

# Our training + Our Voc Segmentation v2 (change detector)
#python render_lerf_mask_ours_v2.py   -m /root/autodl-tmp/output/ramen_ours_v3   --skip_train   --iteration -1   --florence_model_id /root/autodl-tmp/florence2_large_ft
#python render_lerf_mask_ours_v2.py   -m /root/autodl-tmp/output/figurines_ours --skip_train --iteration -1   --florence_model_id /root/autodl-tmp/florence2_large_ft
#python render_lerf_mask_ours_v2.py   -m /root/autodl-tmp/output/teatime_ours --skip_train --iteration -1   --florence_model_id /root/autodl-tmp/florence2_large_ft

# eval for Voc_seg
#python script/eval_lerf_mask_ours.py ramen
#python script/eval_lerf_mask.py teatime
#python script/eval_lerf_mask.py figurines
