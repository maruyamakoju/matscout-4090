# MatScout-4090 reproducible GPU image (Blackwell / cu128).
# Build:  docker build -t matscout:latest .
# Run  :  docker run --gpus all -e MP_API_KEY=$MP_API_KEY -v $PWD/outputs:/app/outputs matscout:latest run-campaign --campaign all
#
# Base carries CUDA 12.8 runtime so the cu128 PyTorch wheels find Blackwell kernels.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY models ./models

# Light core, then GPU/ML stack (cu128 torch + chgnet + mace), then data extras.
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -e . && \
    python3.11 -m pip install torch --index-url https://download.pytorch.org/whl/cu128 && \
    python3.11 -m pip install chgnet "mace-torch>=0.3.6" matminer

ENTRYPOINT ["matscout"]
CMD ["--help"]
