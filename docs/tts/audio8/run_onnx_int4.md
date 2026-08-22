# Quick Test

* download model
```bash
hf download Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4 
```

* setup

```bash
git clone https://github.com/Audio8-AI/Audio8_TTS.git
cd Audio8_TTS/onnx_runtime

uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

* run server and client
```bash
# server
export ARKTTS_MODEL_DIR="$HOME/.cache/huggingface/hub/models--Audio8--Audio8-TTS-Preview-0.6B-ONNX-INT4/snapshots/818569c6b832118ad68d61bbd873abe250fcd68a"
export ARKTTS_VOICES_DIR="./voices"
export ARKTTS_REGISTRATION_DIR="$HOME/.cache/huggingface/hub/models--Audio8--Audio8-TTS-Preview-0.6B-ONNX-INT4/snapshots/818569c6b832118ad68d61bbd873abe250fcd68a/registration"
export ARKTTS_PRECISION="int4"
export ARKTTS_CODEC_PRECISION="fp16"
export ARKTTS_THREADS="2"

uvicorn arktts_runtime.service:app \
    --app-dir "./" \
    --host "0.0.0.0" \
    --port "8024"

# register reference voice
curl http://localhost:8024/api/voices/register \
  -F 'audio=@/c/Users/siao/iloveit/FAKEit/docs/tts/reference.wav' \
  -F 'text=The exact transcript of the reference recording.' \
  -F 'name=speaker_a' \
  -F 'overwrite=false'

# client request
curl http://localhost:8024/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Every voice carries a story. With Audio8, that story can travel across languages and reach more people.","voice_name":"speaker_a","max_new_tokens":128}' \
  -o ./output.wav

```
