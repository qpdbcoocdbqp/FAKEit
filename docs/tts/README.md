# Audio8 Torch TTS API

This service loads Audio8 with Torch. It uses CUDA + `bfloat16` when a GPU is
available and falls back to CPU + `float32` otherwise.

## Start with Docker

The default model mount is the Hugging Face cache at
`~/.cache/huggingface/hub` (`root_model_dir`). The service automatically finds
the Audio8 snapshot below that directory. To mount a specific snapshot, set
`MODEL_DIR` to the container path and keep it read-only.

```bash
docker compose up --build
```

Registered voices are persisted in `./voices` by the compose file.

## API

Register a voice (the transcript must match the uploaded recording):

```bash
curl http://localhost:8024/register_voice \
  -F 'audio=@reference.wav' \
  -F 'text=Exact transcript of the recording.' \
  -F 'name=speaker_a'
```

Generate a WAV file:

```bash
curl http://localhost:8024/synthesize_to_wav \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Audio8.","voice":"speaker_a"}' \
  -o speech.wav
```

Aliases `/api/voices/register` and `/api/tts` are also available. `GET /health`
and `GET /voices` can be used for readiness and voice discovery.
