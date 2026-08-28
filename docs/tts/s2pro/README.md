# s2.cpp Docker Inference Server

Local Docker deployment solution for Fish Audio S2 Pro TTS model, based on [s2.cpp](https://github.com/rodrigomatta/s2.cpp).

---

## Directory Structure

```
s2cpp-docker/
├── Dockerfile          # Multi-stage build (cuda:devel → cuda:runtime)
├── docker-compose.yml  # GPU configuration + volume mounts
└── voices/             # Directory for .s2voice voice profiles (optional)
```

---

## Quick Start

### 1. Prerequisites

```bash
# Verify GPU access from Docker
docker run --rm --gpus all nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 nvidia-smi

cd docs/tts/s2pro
docker build -t s2cpp:latest .
```

### 2. Start Server

```bash
docker compose up -d
```

---

## API Usage

Once started, the server listens on `http://localhost:3030`.

### Basic Synthesis

```bash
curl -X POST http://localhost:3030/generate \
  --form "text=Hello, this is a speech synthesis test." \
  -o output.wav
```

### Synthesis with Parameters

```bash
curl -X POST http://localhost:3030/generate \
  --form "text=The quick brown fox jumps over the lazy dog." \
  --form 'params={"max_new_tokens":512,"temperature":0.58,"top_p":0.88,"top_k":40}' \
  -o output.wav
```

### Voice Cloning

```bash
curl -X POST http://localhost:3030/generate \
  --form "reference=@reference.wav" \
  --form "reference_text=Transcript content of the reference audio." \
  --form "text=Synthesize this text using this voice." \
  -o output_cloned.wav
```

### Real-time Streaming (PCM16 Low Latency)

> Note: Live playback via `ffplay` requires `ffmpeg` installed on your host system (e.g., `sudo apt install ffmpeg` on Ubuntu/Debian or `brew install ffmpeg` on macOS).

```bash
curl -sN -X POST http://localhost:3030/generate \
  --form "voice=reference" \
  --form "text=Anthropic's learning platform, Claude Academy (academy.claude.com), has launched. Courses, instruction, and real-world use cases are integrated into a single portal, all free, and registration requires only an email address. This wasn't entirely a product from scratch: Anthropic mentioned on its blog that the company already had a dedicated team focused on educational content, researching how to enable the general public to use AI safely and effectively. Claude Academy is essentially a repackaged and publicly released version of this internal training methodology, targeting everyone from employees to the general public." \
  --form 'params={
    "stream": true,
    "chunked": true,
    "output_format": "pcm_s16le",
    "segment_sentences": true,
    "stream_start_buffer_ms": 1000,
    "max_new_tokens": 512
  }' \
| ffplay -autoexit -nodisp -infbuf -f s16le -ar 44100 -ac 1 -
```

### Using Saved Voice Profiles

```bash
# Enter container
docker exec -it s2cpp-server /bin/sh

# Encode reference audio to .s2voice format
/app/s2 \
  --model /app/models/s2-pro-q6_k.gguf \
  --tokenizer /app/tokenizer.json \
  --prompt-audio /app/voices/reference.wav \
  --prompt-text "Reference prompt transcript text here..." \
  --save-voice \
  --voice-dir /app/voices \
  --voice reference \
  --text "test" \
  --output /tmp/test.wav \
  -c 0

head -c 8 /app/voices/reference.s2voice | cat -v

# Place the .s2voice file into ./voices/ directory first
curl -X POST http://localhost:3030/generate \
  --form "voice=reference" \
  --form "text=Welcome to our stream! [excited] Massive sale today! [laughing] Don't miss out!" \
  -o reference_output.wav
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/s2-pro-q6_k.gguf` | Path to GGUF model file |
| `HOST` | `0.0.0.0` | Server host binding |
| `PORT` | `3030` | Server listening port |
| `GPU_LAYERS` | `-1` | Transformer layers on GPU (`-1` = auto, `0` = CPU only) |
| `THREADS` | `0` | CPU threads (`0` = auto) |
| `LOG_LEVEL` | `info` | Log verbosity: `error`, `warn`, `info`, or `debug` |
| `EXTRA_ARGS` | — | Extra CLI flags (e.g. `--codec-cpu`, `--codec-context-frames 128`) |

### Adjustments for Insufficient VRAM

When GPU VRAM is limited, you can set the following in `docker-compose.yml`:

```yaml
environment:
  GPU_LAYERS: 18                              # Offload partial transformer layers to GPU
  EXTRA_ARGS: --codec-cpu --codec-context-frames 128  # Force codec on CPU and reduce history buffer to save VRAM
```

VRAM tuning options:
- `--codec-cpu`: Force codec calculation on CPU even when the model runs on GPU.
- `--codec-context-frames <n>`: Reduce codec history context length (lower values consume less VRAM).
- `GPU_LAYERS <n>`: Limit the number of layers offloaded to the GPU.

---

## Useful Commands

```bash
# View logs
docker compose logs -f

# Stop server
docker compose down

# Enter container
docker compose exec s2cpp /bin/bash

# Rebuild image (after updating s2.cpp version)
docker compose build --no-cache
docker compose up -d
```
