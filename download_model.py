# download_model.py
import os
from dotenv import load_env
from huggingface_hub import snapshot_download

load_env()

snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    local_dir="/app/flux1-dev.safetensors",
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["*.md", "*.txt"],
)

print("Model downloaded successfully.")