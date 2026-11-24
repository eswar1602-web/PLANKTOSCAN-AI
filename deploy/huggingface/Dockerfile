# PlanktoScan AI on Hugging Face Spaces (Docker SDK, free CPU hardware).
# Built by HF from the folder that deploy/huggingface/build_space.py assembles; not used for local development.
FROM python:3.12-slim

# OpenCV needs these shared libraries at import time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PLANKTOSCAN_HISTORY_DIR=/tmp/planktoscan-history \
    OMP_NUM_THREADS=2
WORKDIR /home/user/app

# CPU-only PyTorch (same version as requirements-lock.txt, without CUDA), then the pinned rest.
COPY --chown=user requirements.txt .
RUN pip install --user torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "-m", "api", "--host", "0.0.0.0", "--port", "7860"]
