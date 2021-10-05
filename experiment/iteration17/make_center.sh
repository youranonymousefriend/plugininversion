#!/bin/bash
for i in `cat center_perm.txt` ; do
	echo "python center.py -y $i"
	echo "python centerx.py -y $i"
done 

