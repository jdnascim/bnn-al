#!/bin/bash

cd data

# download the dataset
bash crisismmd_dataset.sh

cd CrisisMMD_v2.0

unzip crisismmd_datasplit_all.zip

cd ../../

# add text to the splits
python add_text.py