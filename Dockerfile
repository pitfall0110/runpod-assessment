FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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

COPY download_model.py .
RUN python download_model.py

CMD ["python", "-u", "handler.py"]