# ─────────────────────────────────────────────────────────────
# Stage 1: Build  (CUDA 12.4 devel image)
# ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_STUB_DIR=/usr/local/cuda/targets/x86_64-linux/lib/stubs
ENV LIBRARY_PATH=${CUDA_STUB_DIR}:${LIBRARY_PATH}

WORKDIR /src

# Clone with submodules (ggml is bundled)
RUN git clone --recurse-submodules https://github.com/rodrigomatta/s2.cpp.git

# Build with CUDA support
RUN cd s2.cpp \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_CUDA=ON -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --parallel 6

# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime  (slim CUDA runtime image)
# ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary + tokenizer from build stage
COPY --from=builder /src/s2.cpp/build/s2       ./s2
COPY --from=builder /src/s2.cpp/tokenizer.json ./tokenizer.json

# Model and voice profile mounts
# Models: /app/models   (mount -v /your/path/models:/app/models)
# Voices: /app/voices   (mount -v /your/path/voices:/app/voices)
RUN mkdir -p models voices

EXPOSE 3030

# 環境變數可覆寫啟動參數
ENV MODEL_PATH=/app/models/s2-pro-q6_k.gguf
ENV HOST=0.0.0.0
ENV PORT=3030
ENV THREADS=0
ENV LOG_LEVEL=info

ENTRYPOINT ["/bin/sh", "-c", \
    "exec /app/s2 \
      --model      \"${MODEL_PATH}\" \
      --tokenizer  /app/tokenizer.json \
      --server \
      --host       \"${HOST}\" \
      --port       \"${PORT}\" \
      --gpu-layers \"${GPU_LAYERS}\" \
      --threads    \"${THREADS}\" \
      --log-level  \"${LOG_LEVEL}\" \
      --voice-dir  /app/voices \
      -c 0 \
      $EXTRA_ARGS"]
