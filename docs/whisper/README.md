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
