# need 138GB RAM

import torch
from peft import AutoPeftModelForCausalLM

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_llama-3-70b-hf-no-robot",safe_serialization=True, max_shard_size="2GB")
