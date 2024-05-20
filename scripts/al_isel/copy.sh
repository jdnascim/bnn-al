#!/bin/bash

qtde_base=10
ct=0
for i in {1..3}; do
    new_ct=$((qtde_base+ct));
    cp exp_$i.sh exp_${new_ct}.sh;
    ((ct++));
    
    sed -i "s/exp=${i}/exp=${new_ct}/" exp_${new_ct}.sh;
    sed -i "s/al random/al unc-kmeans/" exp_${new_ct}.sh;
done