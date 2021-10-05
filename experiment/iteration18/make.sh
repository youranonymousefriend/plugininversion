#!/bin/bash
for i in `cat  bs_classes.txt` ; do
	for j in 1 2 4 8 16 32 64 ; do 
		echo "python invert.py -y $i -a $j "
	done 
done 

