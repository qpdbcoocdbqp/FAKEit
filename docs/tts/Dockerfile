FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ROOT_MODEL_DIR=/models \
    VOICE_DIR=/voices \
    PRECISION=int4 \
    CODEC_PRECISION=fp16 \
    THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Keep the PyTorch supplied by the base image, while installing API packages
# into an isolated environment so PEP 668 does not block pip.
RUN python -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements-api.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --prefer-binary -r requirements-api.txt
COPY . .
RUN mkdir -p /voices

EXPOSE 8024
CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8024"]
