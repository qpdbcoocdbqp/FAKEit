# Audio8 Torch TTS API

This service loads Audio8 with Torch. It uses CUDA + `bfloat16` when a GPU is
available and falls back to CPU + `float32` otherwise.

[Audio8/Audio8-TTS-Preview-0.1b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b)
[Audio8/Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)

## Start with Docker

The default model mount is the Hugging Face cache at
`~/.cache/huggingface/hub` (`root_model_dir`). The service automatically finds
the Audio8 snapshot below that directory. To mount a specific snapshot, set
`MODEL_DIR` to the container path and keep it read-only.

### 1. Build Docker Image

```bash
docker build -t audio8:latest .
```

### 2. Run with Docker Compose

Copy the example environment file and adjust if needed:
```bash
cp .env.example .env
```

You can switch the model via the `MODEL_NAME` environment variable:
- `Audio8/Audio8-TTS-Preview-0.1b`
- `Audio8/Audio8-TTS-Preview-0.6b` (default)

```bash
# Example: run with default 0.6b model
docker compose up -d

# Example: run with 0.1b model
MODEL_NAME=Audio8/Audio8-TTS-Preview-0.1b docker compose up -d
```

Registered voices are persisted in `./voices` by the compose file.

## API

Register a voice (the transcript must match the uploaded recording):

```bash
curl http://localhost:8024/api/voices/register \
  -F 'audio=@../resource/reference.wav' \
  -F 'text=Exact transcript of the recording.' \
  -F 'name=role_a'
```

Generate a WAV file:

```bash
curl -X POST http://localhost:8024/api/tts \
     -H "Content-Type: application/json" \
     -d '{"text":"I love anime and games. I am a Japanese girl who loves video games.","voice":"role_a"}' \
     -o speech.wav
```
