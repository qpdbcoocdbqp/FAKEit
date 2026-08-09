"""Whisper ASR + Audio8 Torch GPU voice cloning.

Edit the configuration constants below, then run:

    python whisper_audio8_clone.py
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from openai import OpenAI
from transformers import AutoModel, AutoProcessor

# ------------------------------- configuration -----------------------------
REFERENCE_AUDIO_MP3 = Path("./docs/tts/resource/reference.mp3")
REFERENCE_AUDIO = Path("./docs/tts/resource/reference.wav")

TTS_TEXT = "這是一段 testing script，是用 reference.mp3 的聲音來合成的。"
OUTPUT_PATH = Path("output.wav")  # .wav or .mp3

WHISPER_API_BASE = os.environ.get("WHISPER_API_BASE", "http://localhost:18000/v1")
WHISPER_API_KEY = os.environ.get("WHISPER_API_KEY", "***")
WHISPER_MODEL = "openai/whisper-large-v3-turbo"

AUDIO8_MODEL = os.environ.get("AUDIO8_MODEL", "Audio8/Audio8-TTS-Preview-0.6b")

VOICES_DIR = Path("./docs/tts/voices")
VOICE_NAME = "role_01"
OVERWRITE_VOICE = True
MAX_NEW_TOKENS = 1024


def transcribe(audio_path: Path) -> str:
    client = OpenAI(base_url=WHISPER_API_BASE, api_key=WHISPER_API_KEY)
    with audio_path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL, file=audio_file, response_format="json"
        )
    text = result.get("text") if isinstance(result, dict) else result.text
    text = " ".join(str(text).strip().split())
    if not text:
        raise RuntimeError("Whisper can not recognize reference_text")
    return text

def register_torch_voice(model, processor, name: str, audio_path: Path,
                         reference_text: str, voices_dir: Path,
                         overwrite: bool) -> Path:
    voice_dir = voices_dir / name
    codes_path = voice_dir / "codes.npy"
    meta_path = voice_dir / "meta.json"
    if codes_path.is_file() and meta_path.is_file() and not overwrite:
        return voice_dir

    proc_out = processor(
        text=["_"], reference_audio=[str(audio_path)],
        reference_text=[reference_text], return_tensors="pt",
    )
    proc_out = {key: value.to(model.device) for key, value in proc_out.items()}
    with torch.inference_mode():
        codes, code_lengths = model.encode_audio(
            proc_out["reference_audio_values"], proc_out["reference_audio_lengths"]
        )
    length = int(code_lengths[0].item())
    codes_np = codes[0, :, :length].cpu().numpy().astype(np.int64)
    voice_dir.mkdir(parents=True, exist_ok=True)
    np.save(codes_path, codes_np)
    meta_path.write_text(
        json.dumps({"name": name, "reference_text": reference_text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return voice_dir


def synthesize_torch(model, processor, text: str, voice_name: str,
                     voices_dir: Path, output_path: Path
                     ) -> None:
    voice_dir = voices_dir / voice_name
    meta = json.loads((voice_dir / "meta.json").read_text(encoding="utf-8"))
    inputs = processor(
        text=[text], reference_text=[meta["reference_text"]],
        reference_codes=[str(voice_dir / "codes.npy")], return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=0.8,
            top_p=0.95, top_k=50, do_sample=True, return_dict_in_generate=True,
        )
        waveforms, lengths = model.decode_audio(generated.codes)
    audio = waveforms[0, :int(lengths[0])].float().cpu().numpy()
    wav_path = output_path if output_path.suffix.lower() == ".wav" else output_path.with_suffix(".wav")
    sf.write(str(wav_path), audio, 44100)
    if wav_path != output_path:
        try:
            subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), str(output_path)],
                           check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("輸出 MP3 需要 ffmpeg，或將 OUTPUT_PATH 改成 .wav") from exc
        wav_path.unlink()

# main

print("[1/3] Whisper ASR...")
reference_text = transcribe(REFERENCE_AUDIO_MP3)
print(f"reference_text: {reference_text}")

print("[2/3] 載入 Audio8 Torch 模型（cuda）並登錄 voice...")
processor = AutoProcessor.from_pretrained(AUDIO8_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(
    AUDIO8_MODEL, trust_remote_code=True, dtype=torch.bfloat16
).eval().to("cuda")
register_torch_voice(model, processor, VOICE_NAME, REFERENCE_AUDIO, reference_text, VOICES_DIR, OVERWRITE_VOICE)

print("[3/3] 使用 GPU 合成 TTS...")
synthesize_torch(
    model, processor, TTS_TEXT, VOICE_NAME, VOICES_DIR, OUTPUT_PATH
    )
print(f"完成：{OUTPUT_PATH.resolve()}")
