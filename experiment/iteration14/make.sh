#!/bin/bash
num_models=$(python -c 'from model import model_library ; print(len(model_library))')
num_models=$((num_models-1))
for n in `seq 0 $num_models` ; do
	echo "python vit_good.py -n ${n} -a 1 -y 980"
done 

