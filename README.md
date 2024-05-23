# fsdp-qlora
llama3 70b training and inference

```

Thu May 23 05:36:58 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A6000               On  | 00000000:05:00.0 Off |                  Off |
| 33%   65C    P2             187W / 300W |  20795MiB / 49140MiB |     99%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX A6000               On  | 00000000:06:00.0 Off |                  Off |
| 30%   60C    P2             169W / 300W |  20795MiB / 49140MiB |     99%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA RTX A6000               On  | 00000000:07:00.0 Off |                  Off |
| 33%   65C    P2             185W / 300W |  20795MiB / 49140MiB |     97%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA RTX A6000               On  | 00000000:08:00.0 Off |                  Off |
| 31%   63C    P2             184W / 300W |  20795MiB / 49140MiB |     99%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      3584      C   /usr/bin/python3                          20742MiB |
|    1   N/A  N/A      3585      C   /usr/bin/python3                          20742MiB |
|    2   N/A  N/A      3586      C   /usr/bin/python3                          20742MiB |
|    3   N/A  N/A      3587      C   /usr/bin/python3                          20742MiB |
+---------------------------------------------------------------------------------------+




# https://huggingface.co/vincentoh/llama3_70b_no_robot_fsdp_qlora

# Install Pytorch for FSDP and FA/SDPA
pip install "torch==2.2.2" tensorboard

# Install Hugging Face libraries
pip install  --upgrade "transformers==4.40.0" "datasets==2.18.0" "accelerate==0.29.3"
pip install  --upgrade "evaluate==0.4.1" "bitsandbytes==0.43.1" "huggingface_hub==0.22.2" "trl==0.8.6"
pip install  --upgrade "peft==0.10.0" 


# creating train and test datasets
python no_robots2json.py

# train command line 8-GPU
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=8 ./run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml

```
