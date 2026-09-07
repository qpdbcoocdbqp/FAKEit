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
uv pip install soundcard numpy faster-whisper llama-cpp-python
# ctranslate2 depend on cu12
uv pip install nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12

python -m docs.whisper.livestream
```

  * Flow: 
    1. Computer Audio (System audio output)
    1. WASAPI Loopback (Audio capture via `soundcard`)
    1. VAD & RMS Filter (Silero VAD / RMS energy silence detection)
    1. faster-whisper & Local Agreement (`FasterWhisperASR` + `HypothesisBuffer` streaming recognition & hallucination filtering)
    1. English Transcript (Real-time English subtitles output `[EN]`)
    1. Async Translation Queue (`SubtitleTranslator` async queue dispatcher)
    1. Local LLM Translation (`llama-cpp-python` inference engine)
    1. Traditional Chinese Subtitles (Real-time Traditional Chinese subtitles output `[ZH]`)

  * Translation Local LLM:
    1. English Text Input (English transcript produced by Whisper)
    1. Inference Engine: `llama-cpp-python` (Supports `n_gpu_layers` GPU acceleration)
    1. Translation Model: TranslateGemma (`translategemma-4b-it-GGUF`)
    1. Prompt Format: Chat completion (`source_lang_code: en`, `target_lang_code: zh-TW`)
    1. Output: Traditional Chinese (`zh-TW` subtitles)
