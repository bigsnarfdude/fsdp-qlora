# fsdp-qlora
llama3 70b training and inference

```

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
