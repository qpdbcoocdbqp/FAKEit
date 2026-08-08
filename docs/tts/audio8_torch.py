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

Voice registration flow:
  1. register_voice() encodes a reference WAV into codec codes via
     model.encode_audio() and saves them to <voices_dir>/<name>/codes.npy
     alongside a meta.json.  Only needed once per speaker.
  2. synthesize() loads the saved codes and passes them to the processor as
     reference_codes, bypassing the encoder entirely on subsequent runs.
"""

import json
import time
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

MODEL_ID = "Audio8/Audio8-TTS-Preview-0.6b"

TEXT = "最高の音質体験をしていただくために、本物をサポートしてください。良いアニメーション、良い音楽、忘れられない思い出。忘れられないことを願っています。",
OUTPUT_PATH = "output.wav"

# Reference voice for zero-shot voice cloning.
REFERENCE_AUDIO = "docs/tts/reference.wav"
# Must be the exact words spoken in REFERENCE_AUDIO.
REFERENCE_TEXT = "突然轉錯帳可能是某個系統整個當掉結果回頭一查才發現寫這段的是AI而唯一該把關的人從頭到尾沒看過他那這個鍋到底該誰扛"


# Where registered voices are stored.
VOICES_DIR = Path("./voices")
# Name used to save / load this speaker.
VOICE_NAME = "user_0"


# ---------------------------------------------------------------------------
# Voice registration
# ---------------------------------------------------------------------------

def register_voice(
    model: "AutoModel",
    processor: "AutoProcessor",
    name: str,
    audio_path: str | Path,
    reference_text: str,
    voices_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Encode a reference WAV and save the codec codes to *voices_dir/name/*.

    Returns the voice directory path.  Skips encoding if the voice already
    exists and *overwrite* is False.
    """
    voice_dir = voices_dir / name
    codes_path = voice_dir / "codes.npy"
    meta_path = voice_dir / "meta.json"

    if codes_path.is_file() and meta_path.is_file() and not overwrite:
        print(f"[register] voice '{name}' already exists, skipping (pass overwrite=True to re-encode)")
        return voice_dir

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"reference audio not found: {audio_path}")

    print(f"[register] encoding reference audio: {audio_path}")
    t0 = time.time()

    # Use the processor just to load + resample the audio tensor.
    proc_out = processor(
        text=["_"],                      # dummy text, not used for encoding
        reference_audio=[str(audio_path)],
        reference_text=[reference_text],
        return_tensors="pt",
    )
    audio_values = proc_out["reference_audio_values"].to(model.device)
    audio_lengths = proc_out["reference_audio_lengths"].to(model.device)

    with torch.inference_mode():
        codes, code_lengths = model.encode_audio(audio_values, audio_lengths)

    # codes: [B, num_codebooks, T]  — take the first (and only) batch item
    length = int(code_lengths[0].item())
    codes_np = codes[0, :, :length].cpu().numpy().astype(np.int64)   # [num_codebooks, T]

    voice_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(codes_path), codes_np)

    meta = {
        "name": name,
        "reference_text": reference_text,
        "shape": list(codes_np.shape),
        "source_audio": str(audio_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[register] saved {codes_np.shape[1]} frames → {voice_dir}  ({time.time() - t0:.1f}s)")
    return voice_dir


# ---------------------------------------------------------------------------
# Synthesis using a registered voice
# ---------------------------------------------------------------------------

def synthesize(
    model: "AutoModel",
    processor: "AutoProcessor",
    text: str,
    voice_name: str,
    voices_dir: Path,
    output_path: str | Path,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    seed: int = 1234,
) -> Path:
    """Generate speech from *text* using the registered voice codes."""
    voice_dir = voices_dir / voice_name
    codes_path = voice_dir / "codes.npy"
    meta_path = voice_dir / "meta.json"

    if not codes_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Voice '{voice_name}' not found in {voices_dir}. "
            "Call register_voice() first."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    reference_text = meta["reference_text"]

    print(f"[synth] voice='{voice_name}'  text='{text}'")
    t0 = time.time()

    inputs = processor(
        text=[text],
        reference_text=[reference_text],
        reference_codes=[str(codes_path)],   # processor accepts a .npy path directly
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if seed is not None:
        torch.manual_seed(seed)
        if model.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            return_dict_in_generate=True,
        )
        print(f"[synth] codes shape={tuple(output.codes.shape)}")
        waveforms, waveform_lengths = model.decode_audio(output.codes)

    print(f"[synth] generated in {time.time() - t0:.1f}s")

    audio = waveforms[0, : int(waveform_lengths[0])].float().cpu().numpy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, model.config.codec_sample_rate)
    duration = len(audio) / model.config.codec_sample_rate
    print(f"[synth] wrote {output_path} ({duration:.2f}s audio @ {model.config.codec_sample_rate} Hz)")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"[info] device={device}  dtype={dtype}")

print(f"[info] loading processor/model: {MODEL_ID}")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = (
    AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=dtype)
    .eval()
    .to(device)
)
print(f"[info] loaded model")

# Step 1 — register the reference voice (only runs encoding when needed)
register_voice(
    model=model,
    processor=processor,
    name=VOICE_NAME,
    audio_path=REFERENCE_AUDIO,
    reference_text=REFERENCE_TEXT,
    voices_dir=VOICES_DIR,
    overwrite=True,          # set True to force re-encoding
)

# Step 2 — synthesize using the saved codes
synthesize(
    model=model,
    processor=processor,
    text=TEXT,
    voice_name=VOICE_NAME,
    voices_dir=VOICES_DIR,
    output_path=OUTPUT_PATH,
    max_new_tokens=1024,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    seed=random.randint(0, 1000),
)



