#!/bin/bash
# for i in `python -c 'import torch; print(torch.randperm(1000).numpy())' | tr -d '[]'` ; do
for i in 0 ; do
	for j in 1 2 4 8 16 32 64 ; do 
		echo "python nat.py -y $i -a $j "
	done 
done 

