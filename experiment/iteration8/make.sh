for i in `cat classes.txt` ; do 
	echo "python nat.py -y ${i} -m normal_real_jitter_many";
done
