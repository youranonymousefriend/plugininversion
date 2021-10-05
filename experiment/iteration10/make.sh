for a in 0.01 0.1 1 ; do 
	for t in `seq 0 64` ; do 
                #echo ${a} ${t}
		 python visualize.py -a ${a} -y ${t}
        done 
done 
