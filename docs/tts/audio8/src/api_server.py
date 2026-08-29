"""Audio8 TTS HTTP API.

The model and registered voices are deliberately kept outside the image. Mount
them into the container and point MODEL_DIR / VOICE_DIR at those mounts.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

try:
    from torch_backend import load_model, register_voice, synthesize_to_wav
except ImportError:
    from .torch_backend import load_model, register_voice, synthesize_to_wav


MODEL_NAME = os.getenv("MODEL_NAME", "Audio8/Audio8-TTS-Preview-0.6b")


def _model_dir() -> Path | str:
    configured = os.getenv("MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    root = Path(os.getenv("ROOT_MODEL_DIR", "~/.cache/huggingface/hub")).expanduser()
    if (root / "runtime_manifest.json").is_file():
        return root
    hf_dir_name = "models--" + MODEL_NAME.replace("/", "--")
    candidates = sorted(root.glob(f"{hf_dir_name}/snapshots/*"))
    if not candidates:
        return MODEL_NAME
    return candidates[-1]


MODEL_DIR = _model_dir()
VOICE_DIR = Path(os.getenv("VOICE_DIR", "/voices")).expanduser()
PRECISION = os.getenv("PRECISION", "int4")
CODEC_PRECISION = os.getenv("CODEC_PRECISION", "fp16")
THREADS = int(os.getenv("THREADS", "2"))

app = FastAPI(title="Audio8 TTS API", version="1.0.0")
model = None
processor = None
model_lock = asyncio.Lock()


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    voice_name: str = Field(
        min_length=1, alias="voice", description="Registered voice name"
    )
    max_new_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, gt=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=50, ge=1)
    seed: int = 42

    model_config = {"populate_by_name": True}


@app.on_event("startup")
def startup() -> None:
    global model, processor
    model, processor = load_model(MODEL_DIR)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_dir": str(MODEL_DIR), "voice_dir": str(VOICE_DIR)}


@app.get("/voices")
def voices() -> list[dict]:
    if model is None:
        raise HTTPException(503, "model is not ready")
    result = []
    for path in sorted(VOICE_DIR.glob("*/meta.json")):
        try:
            result.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return result


@app.post("/register_voice")
@app.post("/api/voices/register")
async def register_voice_endpoint(
    audio: Annotated[UploadFile, File(description="Reference WAV/audio file")],
    text: Annotated[str, Form(description="Exact transcript of the reference audio")],
    name: Annotated[str, Form()],
    overwrite: Annotated[bool, Form()] = False,
) -> dict:
    if model is None or processor is None:
        raise HTTPException(503, "model is not ready")
    suffix = Path(audio.filename or "reference.wav").suffix or ".wav"
    temporary = Path(tempfile.gettempdir()) / f"audio8-{uuid4().hex}{suffix}"
    try:
        temporary.write_bytes(await audio.read())
        async with model_lock:
            return await asyncio.to_thread(
                register_voice, model, processor, name, temporary, text, VOICE_DIR, overwrite
            )
    except (FileNotFoundError, ValueError, RuntimeError, FileExistsError) as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        temporary.unlink(missing_ok=True)


@app.post("/synthesize_to_wav")
@app.post("/api/tts")
async def synthesize_endpoint(request: SynthesizeRequest) -> FileResponse:
    if model is None or processor is None:
        raise HTTPException(503, "model is not ready")
    output = Path(tempfile.gettempdir()) / f"audio8-{uuid4().hex}.wav"
    try:
        async with model_lock:
            await asyncio.to_thread(
                synthesize_to_wav, model, processor, request.text, request.voice_name, VOICE_DIR, output,
                request.max_new_tokens, request.temperature, request.top_p,
                request.top_k, request.seed,
            )
        return FileResponse(
            output, media_type="audio/wav", filename="speech.wav",
            background=BackgroundTask(output.unlink, missing_ok=True),
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        output.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
