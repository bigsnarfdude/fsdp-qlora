# fsdp-qlora
llama3 70b training and inference

```

# https://huggingface.co/vincentoh/llama3_70b_no_robot_fsdp_qlora

# creating train and test datasets
python no_robots2json.py

# train command line 8-GPU
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=8 ./run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml

```
