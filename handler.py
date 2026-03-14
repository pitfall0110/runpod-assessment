# handler.py
import os
import base64
from io import BytesIO

import runpod
import torch
from diffusers.pipelines.flux import FluxPipeline


PIPE = None

def load_pipe():
    # FLUX.1-dev requires a GPU and is typically run in fp16/bf16.
    # Use bf16 if your GPU supports it; otherwise fp16.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    pipe = FluxPipeline.from_pretrained(
        "/app/flux1-dev",
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