for a in 1 2 4 8 16 32 ; do 
    for i in `seq 0 5 45` ; do
      echo "python batch_size.py -y ${i} -m normal_real_jitter_many -a ${a}";
    done
done
