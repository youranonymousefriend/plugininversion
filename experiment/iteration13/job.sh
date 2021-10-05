#!/bin/bash 
base_file=$1
n_gpus=4
lines=$(cat "${base_file}" | wc -l)
sub_jobs=$(((lines+n_gpus-1)/n_gpus))

echo ${lines} ${sub_jobs}

job_dir=".${job_dir}"
mkdir -- ${job_dir}
mkdir -- ${job_dir}/scripts
mkdir -- ${job_dir}/outputs 

for i in `seq 0 $((sub_jobs-1))` ; do 
    echo -e "#!/bin/bash\n" > "${job_dir}/scripts/${i}.sh"
    for j in `seq 0 $((n_gpus-1))` ; do 
	line_no=$((i*n_gpus+j+1))
        if [[ ${line_no} -gt ${lines} ]] ; then 
            break
        fi 

	line="CUDA_VISIBLE_DEVICE=${j} `sed \"${line_no}q;d\" \"${base_file}\"` > ${job_dir}/outputs/${i}.txt & \n w${j}=\$!" 
        echo -e "${line}" >> "${job_dir}/scripts/${i}.sh"
    done 
    for j in `seq 0 $((n_gpus-1))` ; do 
	line_no=$((i*n_gpus+j+1))
        if [[ ${line_no} -gt ${lines} ]] ; then 
            break
        fi 
	line="wait \${w${j}}" 
        echo -e "${line}" >> "${job_dir}/scripts/${i}.sh"
    done 
done 


echo "Made ${sub_jobs}. Running now" 
for i in `seq 0 $((sub_jobs-1))` ; do 
    cp "${job_dir}/scripts/${i}.sh" ".current.sh"
    source ".current.sh" 
done 
