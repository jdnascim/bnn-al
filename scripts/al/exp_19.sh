#!/bin/bash

train_set_size=$1
train_set_id=$2
gpu=$3
run_id=$4
dataset=$5

exp=19
imagepath=./data/CrisisMMD_v2.0/

cd ../../

python3 train_and_infer_al.py \
  --exp_id $exp \
  --exp_group al \
  --device $gpu \
  --event $dataset \
  --labeled_size $train_set_size \
  --set_id $train_set_id \
  --arch base_gnn_al \
  --run_id $run_id \
  --reduction autoenc \
  --autoenc base_ae \
  --event $dataset \
  --al unc \
  --al_iter 2 \
  --al_batch 20 \
  --al_isel balanced_random \
  --aug_unlbl_set
