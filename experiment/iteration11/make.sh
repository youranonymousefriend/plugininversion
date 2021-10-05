for a in 1 2 4 8 16 32 ; do
	for t in `seq 0 5 49` ; do
		  echo "python optim_size.py -y ${t} -a ${a}"
    done
done 
