import json
import wave
from pathlib import Path
import numpy as np
from docs.tts.utils.runtime import ArkTtsRuntime
from docs.tts.utils.registration import VoiceRegistration

def _read_fingerprint(manifest_path: Path) -> str | None:
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    fingerprint = manifest.get("model_fingerprint")
    return str(fingerprint) if fingerprint else None

def _resolve_model_fingerprint(model_dir: Path) -> str:
    fingerprint = _read_fingerprint(model_dir / "registration_manifest.json")
    if fingerprint:
        return fingerprint
    raise ValueError(
        "Confirm that registration_manifest.json "
        "contains the model_fingerprint field"
    )
 
def register_voice(
    model_dir: str | Path,
    voices_dir: str | Path,
    name: str,
    audio_path: str | Path,
    text: str,
    overwrite: bool = False,
) -> dict:
    """Read a piece of reference audio, encode it into codes using the codec encoder, and register it as a new voice."""
    model_dir = Path(model_dir).expanduser()
    voices_dir = Path(voices_dir).expanduser()
    audio_path = Path(audio_path).expanduser()
 
    fingerprint = _resolve_model_fingerprint(model_dir)
    registration = VoiceRegistration(model_dir, voices_dir, fingerprint)


    state = registration.status()
    if not state["available"]:
        raise RuntimeError(f"Voice registration is unavailable: {state['reason']}")
 
    data = audio_path.read_bytes()
    meta = registration.register(
        data=data,
        filename=audio_path.name,
        text=text,
        name=name,
        overwrite=overwrite,
    )
    return meta
 
def load_runtime(
    model_dir: str | Path,
    voices_dir: str | Path,
    precision: str | None = None,
    codec_precision: str | None = None,
    threads: int | None = None,
) -> ArkTtsRuntime:
    """Load models (slow / fast / codec decoder), tokenizer, and voice store."""
    return ArkTtsRuntime(
        model_dir=Path(model_dir),
        voices_dir=Path(voices_dir),
        precision=precision,
        codec_precision=codec_precision,
        threads=threads,
    )

def synthesize_to_wav(
    runtime: ArkTtsRuntime,
    text: str,
    voice: str,
    output_path: str | Path,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    seed: int = 42,
    sample_rate: int = 24000,
) -> Path:
    """Call the model to generate audio and save it as a wav file."""
    audio, _codes = runtime.synthesize(
        text=text,
        voice=voice,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )

    output_path = Path(output_path)
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return output_path


model = Path("~/.cache/huggingface/hub/models--Audio8--Audio8-TTS-Preview-0.6B-ONNX-INT4/snapshots/818569c6b832118ad68d61bbd873abe250fcd68a").expanduser()
regist_model = model / "registration"
reference_audio = Path("docs/tts/reference.wav").expanduser()
reference_text="突然转错帐可能是某个系统整个当掉结果回头一查才发现写這段code的是AI而唯一该把关的人从头到尾没看过他那这个锅到底该谁扛",

voices = Path("./voices")
voice = "user_0"
text = "今天想和你分享一个好消息，Audio8 现在可以用更高效的方式生成自然流畅的语音。"
output = "output.wav"

# Register a new voice using the reference audio and text
meta = register_voice(
    model_dir=regist_model,
    voices_dir=voices,
    name=voice,
    audio_path=reference_audio,
    text=reference_text,
    overwrite=True,
    )

print(f"Registered voice: {meta['name']}")
print(json.dumps(meta, ensure_ascii=False, indent=2))

# Load the runtime with the model and voices
runtime = load_runtime(
    model_dir=model,
    voices_dir=voices,
    precision="int4",
    codec_precision="fp16",
    threads=2,
)

# Generate audio from the text using the registered voice and save it to a wav file
output_path = synthesize_to_wav(
    runtime=runtime,
    text=text,
    voice="speaker_a",
    output_path=output,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    seed=42,
    sample_rate=44100,
)

print(f"Generated audio file: {output_path}")
