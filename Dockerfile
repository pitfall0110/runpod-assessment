ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

FROM ${BASE_IMAGE} AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        "diffusers==0.31.0" \
        "transformers==4.45.2" \
        "accelerate>=0.26.0" \
        "sentencepiece" \
        "protobuf"

COPY handler.py .

ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}


RUN wget -q --header="Authorization: Bearer ${HF_TOKEN}" -O /app/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors

CMD ["python", "-u", "handler.py"]