# CUDA_VISIBLE_DEVICE=0 python nat.py -y 0 -a 1 & 
# CUDA_VISIBLE_DEVICE=1 python nat.py -y 0 -a 2 & 
# CUDA_VISIBLE_DEVICE=2 python nat.py -y 0 -a 4 & 
# CUDA_VISIBLE_DEVICE=3 python nat.py -y 0 -a 8 & 
CUDA_VISIBLE_DEVICE=0 python nat.py -y 0 -a 16 & 
CUDA_VISIBLE_DEVICE=1 python nat.py -y 0 -a 32  & 
CUDA_VISIBLE_DEVICE=2 python nat.py -y 0 -a 64  & 
