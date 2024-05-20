#!/bin/bash

train_set_id=$1
gpu=$2
run_id=$3
dataset=$4

exp=18
imagepath=./data/CrisisMMD_v2.0/

cd ../../

python3 train_and_infer_al_isel.py \
  --exp_id $exp \
  --exp_group al_isel \
  --device $gpu \
  --event $dataset \
  --labeled_size 18 \
  --set_id $train_set_id \
  --arch bayesian_mlp \
  --run_id $run_id \
  --event $dataset \
  --al bald-degree \
  --al_iter 2 \
  --al_batch 16 \
  --al_isel degree \
  --al_random_pseudo_val \