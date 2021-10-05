#!/bin/bash
for i in `cat classes_i_like.txt` ; do
	echo "python zoom.py -y $i"
	echo "python zoom_bad.py -y $i"
done 

