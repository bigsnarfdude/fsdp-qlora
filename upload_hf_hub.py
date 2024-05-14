from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="merged_llama-3-70b-hf-no-robot",
    repo_id="vincentoh/llama3_70b_no_robot_fsdp_qlora",
    repo_type="model",
)
