#!/bin/bash
for y in `python -c 'import torch ; print(torch.randperm(1000).numpy()[:10])' | tr -d '[]'` ; do 
	for i in `cat  vit_i_like.txt` ; do
		echo "python vit.py -y $y -n $i -a 1"
	done 
done

