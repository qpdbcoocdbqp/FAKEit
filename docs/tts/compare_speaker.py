"""Compare speaker similarity between two audio files.

Extracts speaker embeddings and reports cosine similarity. Intended for
voice-cloning evaluation where the two clips may contain different text.

Usage:
    python compare_speaker.py
    python compare_speaker.py resource/reference.wav api_test_output.wav
    python compare_speaker.py -r C:/path/ref.wav -t C:/path/out.wav
    python compare_speaker.py -r ref.wav -t out.wav -o result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


DEFAULT_REFERENCE = Path("resource/reference.wav")
DEFAULT_HYPOTHESIS = Path("api_test_output.wav")
DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_CACHE_DIR = Path(".cache/speechbrain")
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioInfo:
    path: str
    duration_sec: float
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class CompareResult:
    reference: AudioInfo
    hypothesis: AudioInfo
    model: str
    cosine_similarity: float
    similarity_percent: float
    interpretation: str


def get_audio_info(path: Path) -> AudioInfo:
    resolved = path.resolve()
    info = sf.info(resolved)
    return AudioInfo(
        path=str(resolved),
        duration_sec=round(info.duration, 3),
        sample_rate=info.samplerate,
        channels=info.channels,
    )


def load_mono_16k(path: Path) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(wav.mean(axis=1))
    if sr != TARGET_SAMPLE_RATE:
        mono = torchaudio.functional.resample(
            mono.unsqueeze(0), sr, TARGET_SAMPLE_RATE
        ).squeeze(0)
    return mono


def load_verifier(model_id: str):
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy
    except ImportError as exc:
        raise RuntimeError(
            "speechbrain and torchaudio are required. Install with:\n"
            "  uv pip install -r requirements-compare.txt"
        ) from exc

    savedir = MODEL_CACHE_DIR / model_id.replace("/", "_")
    return SpeakerRecognition.from_hparams(
        source=model_id,
        savedir=savedir,
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": "cpu"},
    )


def compare(reference: Path, hypothesis: Path, model_id: str = DEFAULT_MODEL) -> CompareResult:
    ref_info = get_audio_info(reference)
    hyp_info = get_audio_info(hypothesis)

    verifier = load_verifier(model_id)
    ref = load_mono_16k(reference).unsqueeze(0)
    hyp = load_mono_16k(hypothesis).unsqueeze(0)
    with torch.inference_mode():
        score, _ = verifier.verify_batch(ref, hyp)
    sim = float(score)

    return CompareResult(
        reference=ref_info,
        hypothesis=hyp_info,
        model=model_id,
        cosine_similarity=round(sim, 6),
        similarity_percent=round(sim * 100, 2),
        interpretation=interpret(sim),
    )


def interpret(sim: float) -> str:
    pct = sim * 100
    if pct < 50:
        return "low - likely different speakers"
    if pct < 60:
        return "weak - some shared traits, but not a strong match"
    if pct < 75:
        return "moderate - typical range for voice-cloning TTS (Seed-TTS SIM ~63-79%)"
    if pct < 85:
        return "high - strong speaker match"
    return "very high - very close speaker identity (verify for overfitting or leakage)"


def print_human(result: CompareResult) -> None:
    print(f"Model     : {result.model}")
    print(f"Reference : {result.reference.path}")
    print(f"  duration={result.reference.duration_sec}s  "
          f"sr={result.reference.sample_rate}  channels={result.reference.channels}")
    print(f"Hypothesis: {result.hypothesis.path}")
    print(f"  duration={result.hypothesis.duration_sec}s  "
          f"sr={result.hypothesis.sample_rate}  channels={result.hypothesis.channels}")
    print()
    print(f"Cosine similarity : {result.cosine_similarity:.6f}")
    print(f"Similarity (SIM%) : {result.similarity_percent:.2f}%")
    print(f"Interpretation    : {result.interpretation}")


def resolve_paths(
    reference: Path | None,
    hypothesis: Path | None,
    positional: list[Path],
) -> tuple[Path, Path]:
    ref = reference
    hyp = hypothesis

    if positional:
        if len(positional) > 2:
            raise ValueError("at most two positional wav paths are allowed")
        if ref is None and len(positional) >= 1:
            ref = positional[0]
        if hyp is None and len(positional) >= 2:
            hyp = positional[1]
        if len(positional) == 1 and hyp is None:
            raise ValueError(
                "provide two positional wav paths, or use -r and -t together"
            )

    if ref is None:
        ref = DEFAULT_REFERENCE
    if hyp is None:
        hyp = DEFAULT_HYPOTHESIS
    return ref, hyp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare speaker similarity between two audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  compare_speaker.py\n"
            "  compare_speaker.py ref.wav out.wav\n"
            "  compare_speaker.py -r C:/voices/ref.wav -t ./speech.wav\n"
            "  compare_speaker.py -r ref.wav -t out.wav -o result.json --json"
        ),
    )
    parser.add_argument(
        "wav_paths",
        nargs="*",
        type=Path,
        metavar="WAV",
        help="reference.wav hypothesis.wav (optional positional shorthand)",
    )
    parser.add_argument(
        "-r", "--reference",
        type=Path,
        default=None,
        help=f"reference speaker wav (default: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "-t", "--hypothesis",
        type=Path,
        default=None,
        help=f"wav to compare against reference (default: {DEFAULT_HYPOTHESIS})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="save JSON result to this file",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face speaker model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--json", action="store_true", help="print JSON result")
    args = parser.parse_args()

    try:
        reference, hypothesis = resolve_paths(
            args.reference, args.hypothesis, args.wav_paths
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    for label, path in (("Reference", reference), ("Hypothesis", hypothesis)):
        if not path.is_file():
            print(f"{label} audio not found: {path.resolve()}", file=sys.stderr)
            return 2

    try:
        result = compare(reference, hypothesis, model_id=args.model)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(asdict(result), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        if not args.json:
            print_human(result)
            print(f"\nSaved: {args.output.resolve()}")
    elif args.json:
        print(payload)
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
