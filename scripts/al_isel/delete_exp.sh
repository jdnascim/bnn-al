#!/bin/bash

exp=$1

find /hahomes/jnascimento/exps/2024-bnn-al/results -type d -regex ".*/al_isel/${exp}.*" -exec rm -rf {} \; 2>/dev/null
