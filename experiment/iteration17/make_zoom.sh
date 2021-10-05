#!/bin/bash
for i in `cat zoom_perm.txt` ; do
	echo "python zoom.py -y $i"
	echo "python zoomx.py -y $i"
done 

