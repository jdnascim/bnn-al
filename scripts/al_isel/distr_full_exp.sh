#!/bin/bash

exp=$1
gpu=$2
gpu_ix=$3
qtde_gpus=$4
id_run=$5
dataset=$6
start_offset=${7:-1}  # Default value is 0 if start_offset is not passed
qtde_experiments=${8:-10}  # Default value is 10 if qtde_experiments is not passed

ct=0
exp_count=0

train_sets_size=${#train_sets[@]}

for i in {0..9}; do
    if [ $(( i % $qtde_gpus )) -eq $gpu_ix ]; then
        exp_count=$((start_offset + ct));
        echo $exp_count of $qtde_experiments;
        ct=$((ct + 1));
        if [[ ! -f "../../results/${dataset}/al_is/${exp}/18_${i}_${id_run}.json" ]] || \
            [[ ! -f "../../results/${dataset}/al_is/${exp}/34_${i}_${id_run}.json" ]] || \
            [[ ! -f "../../results/${dataset}/al_is/${exp}/50_${i}_${id_run}.json" ]]; then
            bash exp_$exp.sh $i $gpu $id_run $dataset
        fi
    fi
done