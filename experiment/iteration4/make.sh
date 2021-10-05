#!/bin/bash 
for i in `seq 0 5 999` ; do 
	echo "python correct.py -y ${i} -x Correct${i}" ; 
done 
