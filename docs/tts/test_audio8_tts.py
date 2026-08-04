"""
Test script for Audio8/Audio8-TTS-Preview-0.6b
https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b

Install deps first:
    pip install "torch>=2.5.0" "torchaudio>=2.5.0" \
        "transformers>=4.57.0,<5" "soundfile>=0.12" "safetensors>=0.4"

Notes:
- Model card lists two different repo ids ("Audio8/..." for the HF page,
  "AutoArk-AI/Audio8-TTS-Preview-0.6b" in the code sample). Try MODEL_ID
  below first; if it 404s, swap in the AutoArk-AI one.
- Needs a CUDA GPU for reasonable speed (bf16). Falls back to CPU float32
  automatically but will be slow.
- trust_remote_code=True is required (custom architecture) -- only run
  this against a model repo you trust.
"""

import sys
import time

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

MODEL_ID = "Audio8/Audio8-TTS-Preview-0.6b"
# Fallback if the above id isn't resolvable:
# MODEL_ID = "AutoArk-AI/Audio8-TTS-Preview-0.6b"

TEXT = "Hello, this is a test of the Audio8 text to speech model."
OUTPUT_PATH = "output.wav"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[info] device={device} dtype={dtype}")

    print(f"[info] loading processor/model: {MODEL_ID}")
    t0 = time.time()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = (
            AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=dtype)
            .eval()
            .to(device)
        )
    except Exception as e:
        print(f"[error] failed to load model '{MODEL_ID}': {e}")
        print("[hint] try MODEL_ID = 'AutoArk-AI/Audio8-TTS-Preview-0.6b' instead")
        sys.exit(1)
    print(f"[info] loaded in {time.time() - t0:.1f}s")

    # --- Generation without a reference voice (simplest smoke test) ---
    print("[info] running generation without reference audio...")
    inputs = processor(text=[TEXT], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            return_dict_in_generate=True,
        )
        waveforms, waveform_lengths = model.decode_audio(output.codes)
    print(f"[info] generated in {time.time() - t0:.1f}s")

    audio = waveforms[0, : int(waveform_lengths[0])].float().cpu().numpy()
    sf.write(OUTPUT_PATH, audio, model.config.codec_sample_rate)
    print(f"[done] wrote {OUTPUT_PATH} ({len(audio) / model.config.codec_sample_rate:.2f}s audio)")


if __name__ == "__main__":
    main()
