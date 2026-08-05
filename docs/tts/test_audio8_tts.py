"""
Test script for Audio8/Audio8-TTS-Preview-0.6b
https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b

Install deps first:
    pip install "torch>=2.5.0" "torchaudio>=2.5.0" \
        "transformers>=4.57.0,<5" "soundfile>=0.12" "safetensors>=0.4"

Notes:
- Needs a CUDA GPU for reasonable speed (bf16). Falls back to CPU float32
  automatically but will be slow.
- trust_remote_code=True is required (custom architecture) -- only run
  this against a model repo you trust.
"""

import time

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

MODEL_ID = "Audio8/Audio8-TTS-Preview-0.6b"

TEXT = "今天想和你分享一个好消息，Audio8 现在可以用更高效的方式生成自然流畅的语音。"
OUTPUT_PATH = "output.wav"

# Reference voice for zero-shot voice cloning.
REFERENCE_AUDIO = "docs/tts/reference_5s.wav"
# Must be the exact words spoken in REFERENCE_AUDIO.
REFERENCE_TEXT = "突然转错帐可能是某个系统整个当掉结果回头一查才发现写着乱扣友是"
# "AI而唯一该把关的人从头到尾没看过他那这个锅到底该谁扛"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[info] device={device} dtype={dtype}")

    # Keep debugging runs reproducible.
    torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)

    print(f"[info] loading processor/model: {MODEL_ID}")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=dtype)
        .eval()
        .to(device)
    )
    print(f"[info] loaded in {time.time() - t0:.1f}s")

    # --- Generation with a reference voice ---
    print(f"[info] running generation with reference audio: {REFERENCE_AUDIO}")
    inputs = processor(
        text=[TEXT],
        reference_audio=[REFERENCE_AUDIO],
        reference_text=[REFERENCE_TEXT],
        return_tensors="pt",
    )
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
        print(f"[debug] codes shape={tuple(output.codes.shape)}")
        waveforms, waveform_lengths = model.decode_audio(output.codes)
    print(f"[debug] waveform lengths={waveform_lengths.tolist()}")
    print(f"[info] generated in {time.time() - t0:.1f}s")

    audio = waveforms[0, : int(waveform_lengths[0])].float().cpu().numpy()
    sf.write(OUTPUT_PATH, audio, model.config.codec_sample_rate)
    print(f"[done] wrote {OUTPUT_PATH} ({len(audio) / model.config.codec_sample_rate:.2f}s audio)")


if __name__ == "__main__":
    main()
