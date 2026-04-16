#!/bin/bash
cd /home/pc/yjy/USIP-master
export PYTHONPATH=/home/pc/yjy/USIP-master:$PYTHONPATH

python evaluation/save_keypoints.py \
--gpu_ids 0 \
--dataset modelnet \
--dataroot ./mytestdatas \
--name 5000-64-k1k9-3d \
--checkpoints_dir ./modelnet/checkpoints \
--input_pc_num 5000 \
--node_num 64 \
--node_knn_k_1 32 \
--k 9
