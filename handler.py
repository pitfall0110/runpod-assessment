# handler.py
import os
import base64
from io import BytesIO

import runpod
import torch
from diffusers.pipelines.flux import FluxPipeline

MODEL_ID = "black-forest-labs/flux.1-dev"
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PIPE = None

def resolve_snapshot_path(model_id: str) -> str:
    """Resolve the local snapshot path for a cached model."""
    if "/" not in model_id:
        raise ValueError(f"model_id '{model_id}' must be in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    if os.path.isdir(snapshots_dir):
        versions = [
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if versions:
            versions.sort()
            return os.path.join(snapshots_dir, versions[0])

    raise RuntimeError(f"Cached model not found: {model_id}")

LOCAL_PATH = resolve_snapshot_path(MODEL_ID)

def load_pipe():
    # FLUX.1-dev requires a GPU and is typically run in fp16/bf16.
    # Use bf16 if your GPU supports it; otherwise fp16.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    pipe = FluxPipeline.from_pretrained(
        LOCAL_PATH,
        torch_dtype=dtype,
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Optional memory optimizations (may or may not be available depending on versions)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    return pipe


def pil_to_base64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    """
    Expected input:
    {
      "input": {
        "prompt": "a cat in space",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "seed": 123
      }
    }
    """
    global PIPE
    job_input = job.get("input", {}) or {}

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing `input.prompt`"}

    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    steps = int(job_input.get("num_inference_steps", 28))
    guidance = float(job_input.get("guidance_scale", 3.5))
    seed = job_input.get("seed", None)

    if PIPE is None:
        PIPE = load_pipe()

    generator = None
    if seed is not None and torch.cuda.is_available():
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # Generate
    out = PIPE(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )

    image = out.images[0]
    image_b64 = pil_to_base64(image)

    return {
        "status": "success",
        "prompt": prompt,
        "image_base64": image_b64,
        "width": width,
        "height": height,
        "seed": seed,
    }


runpod.serverless.start({"handler": handler})