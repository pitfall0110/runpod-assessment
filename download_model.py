from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="blackhole0110/FLUX.1-dev/tree/main",
    local_dir="/app/flux1-dev.safetensors",
    ignore_patterns=["*.md", "*.txt"],
)

print("Model downloaded successfully.")