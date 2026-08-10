from __future__ import annotations

import json
import random
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor


def load_model(model_dir: str | Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, dtype=dtype
    ).eval().to(device)
    return model, processor


def register_voice(model, processor, name, audio_path, reference_text, voices_dir, overwrite=False):
    voice_dir = Path(voices_dir) / name
    codes_path, meta_path = voice_dir / "codes.npy", voice_dir / "meta.json"
    if codes_path.is_file() and meta_path.is_file() and not overwrite:
        raise FileExistsError(f"voice already exists: {name}")
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"reference audio not found: {audio_path}")
    proc = processor(text=["_"], reference_audio=[str(audio_path)],
                     reference_text=[reference_text], return_tensors="pt")
    audio = proc["reference_audio_values"].to(model.device)
    lengths = proc["reference_audio_lengths"].to(model.device)
    with torch.inference_mode():
        codes, code_lengths = model.encode_audio(audio, lengths)
    length = int(code_lengths[0].item())
    values = codes[0, :, :length].cpu().numpy().astype(np.int64)
    voice_dir.mkdir(parents=True, exist_ok=True)
    np.save(codes_path, values)
    meta = {"name": name, "reference_text": reference_text,
            "shape": list(values.shape), "source_audio": audio_path.name}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def synthesize_to_wav(model, processor, text, voice, voices_dir, output_path,
                      max_new_tokens=1024, temperature=0.7, top_p=0.9,
                      top_k=50, seed=42, sample_rate=24000):
    voice_dir = Path(voices_dir) / voice
    codes_path, meta_path = voice_dir / "codes.npy", voice_dir / "meta.json"
    if not codes_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Voice '{voice}' not found in {voices_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    inputs = processor(text=[text], reference_text=[meta["reference_text"]],
                       reference_codes=[str(codes_path)], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    torch.manual_seed(seed)
    if model.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k, do_sample=True,
            return_dict_in_generate=True)
        waveforms, lengths = model.decode_audio(output.codes)
    audio = waveforms[0, :int(lengths[0])].float().cpu().numpy()
    path = Path(output_path)
    sf.write(path, audio, int(getattr(model.config, "codec_sample_rate", sample_rate)))
    return path
