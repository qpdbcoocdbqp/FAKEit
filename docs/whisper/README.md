# Whisper

* Model download

```bash
hf download openai/whisper-large-v3-turbo \
--include "*.json" \
--include "*.safetensors" \
--include "*.txt"
```

* Dataset: [hf-internal-testing/librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)

* convert
```bash
ffmpeg/bin/ffmpeg -i ./audio.flac ./audio.mp3
ffmpeg/bin/ffplay ./audio.mp3
```

* Live stream

```bash
uv pip install soundcard numpy faster-whisper

python -m livestream
```

  * flow: 
    1. Computer Audio
    1. WASAPI Loopback
    1. VAD
    1. faster-whisper
    1. English Transcript
    1. Translation
    1. 中文字幕

  * Translation Local LLM：
    1. Whisper
    1. English
    1. Qwen / Gemma / Llama
    1. Traditional Chinese
