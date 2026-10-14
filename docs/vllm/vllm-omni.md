# Qwen3-TTS with vLLM-Omni

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Docker image: `vllm/vllm-omni:v0.30.0`
- Deployment configuration: [qwen3-tts-8gb.yaml](qwen3-tts-8gb.yaml)

## Start the Server

Run from the repository root in Git Bash.

```bash
HF_CACHE_DIR="$HOME/.cache/huggingface"

MSYS_NO_PATHCONV=1 docker run -d --gpus all \
  --name qwen3-tts \
  --ipc=host \
  -p 8091:8091 \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$(pwd -W)/docs/vllm/qwen3-tts.yaml:/config/qwen3-tts.yaml:ro" \
  --entrypoint vllm \
  vllm/vllm-omni:v0.30.0 \
  serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --omni \
  --host 0.0.0.0 \
  --port 8091 \
  --deploy-config /config/qwen3-tts.yaml
```

## Check Readiness

After initialization, check the server:

```bash
curl --noproxy '*' --fail-with-body -i http://127.0.0.1:8091/health
curl --noproxy '*' --fail-with-body http://127.0.0.1:8091/v1/models
```

## Generate Test Audio

```bash
curl --noproxy '*' --fail-with-body --max-time 300 \
  http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello, this is a speech synthesis test.",
    "voice": "vivian",
    "language": "English",
    "response_format": "wav"
  }' \
  --output speech.wav
```

Output: `speech.wav`.
