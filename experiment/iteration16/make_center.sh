#!/bin/bash
for i in `cat classes_i_like.txt` ; do
	echo "python center.py -y $i"
done 

