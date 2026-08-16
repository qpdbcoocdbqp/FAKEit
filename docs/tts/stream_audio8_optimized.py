# -*- coding: utf-8 -*-
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
  2. stream_synthesize() loads the saved codes and passes them to the processor as
     reference_codes, bypassing the encoder entirely on subsequent runs.
"""

import json
import os
import time
import types
import threading
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor


def _suppress_alsa_errors() -> None:
    """Redirect noisy C-level ALSA underrun prints from stderr in WSL2/Linux."""
    try:
        import ctypes
        asound = ctypes.cdll.LoadLibrary("libasound.so.2")
        c_handler_type = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
        )
        def _dummy_handler(file, line, function, err, fmt):
            pass
        _suppress_alsa_errors._c_handler = c_handler_type(_dummy_handler)
        asound.snd_lib_error_set_handler(_suppress_alsa_errors._c_handler)
    except Exception:
        pass


_suppress_alsa_errors()


_FAST_PROFILE = None
_FAST_PROFILE_ENABLED = os.environ.get("AUDIO8_FAST_PROFILE", "0") == "1"


def _reset_fast_profile(device) -> None:
    global _FAST_PROFILE
    if not _FAST_PROFILE_ENABLED:
        _FAST_PROFILE = None
        return
    _FAST_PROFILE = {
        "cuda": device.type == "cuda",
        "fast_step": [[] for _ in range(10)],
        "filter": [[] for _ in range(9)],
        "sample": [[] for _ in range(9)],
    }


def _profile_start():
    if _FAST_PROFILE is not None and _FAST_PROFILE["cuda"]:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    return time.perf_counter()


def _profile_end(bucket, position, start):
    if _FAST_PROFILE is None:
        return
    if _FAST_PROFILE["cuda"]:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        _FAST_PROFILE[bucket][position].append((start, end))
    else:
        _FAST_PROFILE[bucket][position].append(
            time.perf_counter() - start
        )


def _report_fast_profile() -> None:
    if _FAST_PROFILE is None:
        return

    def elapsed(item):
        if _FAST_PROFILE["cuda"]:
            start, end = item
            return start.elapsed_time(end) / 1000.0
        return item

    def report(bucket, label):
        totals = [sum(elapsed(item) for item in values)
                  for values in _FAST_PROFILE[bucket]]
        total = sum(totals)
        calls = sum(len(values) for values in _FAST_PROFILE[bucket])
        if not calls:
            return
        average = total / calls
        detail = ", ".join(
            f"p{index}={value / max(1, len(_FAST_PROFILE[bucket][index])) * 1000.0:.2f}ms"
            for index, value in enumerate(totals)
            if _FAST_PROFILE[bucket][index]
        )
        print(
            f"[fast-profile] {label} total={total:.3f}s "
            f"calls={calls} avg={average * 1000.0:.2f}ms  {detail}"
        )

    report("fast_step", "fast_step")
    report("filter", "topk_topp")
    report("sample", "sample_embed")

MODEL_ID = "Audio8/Audio8-TTS-Preview-0.6b"

# TEXT = "最高の音質体験をしていただくために、本物をサポートしてください。良いアニメーション、良い音楽、忘れられない思い出。忘れられないことを願っています。"
TEXT = "We may use artificial intelligence (AI) tools to support parts of the hiring process, such as reviewing applications, analyzing resumes, or assessing responses and identifying potential inconsistencies or verification signals in application materials based on available information. These tools assist our recruitment team but do not replace human judgment. Final hiring decisions are ultimately made by humans. If you would like more information about how your data is processed, please contact us."

OUTPUT_PATH = "output.wav"

# Reference voice for zero-shot voice cloning.
REFERENCE_AUDIO = "docs/tts/resource/reference.wav"
# Must be the exact words spoken in REFERENCE_AUDIO.
REFERENCE_TEXT = "突然轉錯帳可能是某個系統整個當掉結果回頭一查才發現寫這段的是AI而唯一該把關的人從頭到尾沒看過他那這個鍋到底該誰扛"


# Where registered voices are stored.
VOICES_DIR = Path("./docs/tts/voices")
# Name used to save / load this speaker.
VOICE_NAME = "user_0"

# Persistent on-disk directory for Inductor's own compile cache (FX graph /
# Triton / autotuning). Overrides the default /tmp/torchinductor_<user>,
# which on WSL2 often lives on tmpfs and gets wiped on reboot -- pointing
# it here means a same-shape rerun on this machine can skip re-compiling
# even without the explicit save/load step below.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path("./docs/tts/inductor_cache").resolve()))

# Portable snapshot of torch.compile's caches (Dynamo guards + AOTAutograd +
# Inductor + Triton + autotuning artifacts), saved/loaded explicitly via
# torch.compiler.{save,load}_cache_artifacts(). This is what actually lets a
# brand-new process skip the cold-compile cost paid during warmup_model().
COMPILE_CACHE_PATH = Path("./docs/tts/compile_cache/audio8_compile_cache.bin")


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
# Streaming synthesis (generate + play as codes come out)
# ---------------------------------------------------------------------------
#
# Audio8's custom `generate()` does not accept HuggingFace's `streamer`
# kwarg. We mirror its generation loop and decode/play incrementally.
#
# Playback smoothness relies on:
#   - fixed codec_frame_size sample mapping (no empirical hop drift)
#   - overlap re-decode + crossfade at chunk boundaries
#   - prefetch buffer + background playback thread (avoids underruns)


def _crossfade_pair(prev: np.ndarray, nxt: np.ndarray, n: int) -> np.ndarray:
    """Crossfade two adjacent audio regions of equal length."""
    if n <= 0:
        return np.empty(0, dtype=np.float32)
    n = min(n, prev.size, nxt.size)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return prev[-n:] * (1.0 - x) + nxt[:n] * x


def _optimized_processed_scores(
    self,
    input_ids: torch.Tensor,
    scores: torch.Tensor,
    processors,
    top_k: int,
    top_p: float,
    temperature: float,
) -> torch.Tensor:
    """Equivalent logits filtering with top-k partial selection.

    The remote model sorts the complete 4096-entry vocabulary for every
    generated codebook.  Since entries after top_k are discarded anyway,
    selecting top_k first avoids that full sort while preserving the same
    top-p filtering order over the retained candidates.
    """
    scores = processors(input_ids, scores)
    vocab_size = scores.shape[-1]
    k = min(max(int(top_k), 1), vocab_size)
    top_scores, top_indices = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)
    cumulative = torch.cumsum(torch.softmax(top_scores, dim=-1), dim=-1)
    positions = torch.arange(k, device=scores.device)
    threshold = torch.tensor(top_p, dtype=cumulative.dtype, device=scores.device)
    remove = cumulative > threshold
    remove[..., 0] = False
    remove |= positions >= k

    kept_scores = top_scores.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(scores, float("-inf"))
    filtered.scatter_(dim=-1, index=top_indices, src=kept_scores)
    temperature_value = torch.tensor(
        temperature, dtype=scores.dtype, device=scores.device
    ).clamp_min(1e-5)
    return filtered / temperature_value


def _optimized_generate_codebooks(
    self,
    slow_hidden: torch.Tensor,
    semantic: torch.Tensor,
    processors,
    top_k: int,
    top_p: float,
    temperature: float,
    do_sample: bool,
    generator=None,
) -> torch.Tensor:
    """Workspace override for Audio8's remote _generate_codebooks()."""
    hidden = self.fast_project_in(slow_hidden)
    fast_positions = torch.arange(
        self.config.num_codebooks, device=hidden.device, dtype=torch.long
    )

    # Position 0 populates the fast KV cache; its logits are intentionally
    # unused by the original model and must still be computed.
    fast_start = _profile_start()
    self._fast_step(hidden, fast_positions[0:1])
    _profile_end("fast_step", 0, fast_start)

    current = (semantic - self.config.semantic_begin_id).clamp(
        0, self.config.codebook_size - 1
    )
    codebooks = [current]
    fast_history = torch.empty(
        (current.shape[0], self.config.num_codebooks),
        dtype=current.dtype,
        device=current.device,
    )
    fast_history[:, 0] = current
    hidden = self.fast_embeddings(current)[:, None]

    for position in range(1, self.config.num_codebooks):
        fast_start = _profile_start()
        scores = self._fast_step(hidden, fast_positions[position:position + 1])
        _profile_end("fast_step", position, fast_start)

        filter_start = _profile_start()
        scores = _optimized_processed_scores(
            self, fast_history[:, :position], scores, processors,
            top_k, top_p, temperature,
        )
        _profile_end("filter", position - 1, filter_start)

        sample_start = _profile_start()
        current = (
            self._sample(scores, generator=generator)
            if do_sample
            else scores.argmax(dim=-1)
        )
        codebooks.append(current)
        fast_history[:, position] = current
        hidden = self.fast_embeddings(current)[:, None]
        _profile_end("sample", position - 1, sample_start)

    return torch.stack(codebooks, dim=1)


def _install_optimized_codebook_generation(model) -> None:
    model._generate_codebooks = types.MethodType(
        _optimized_generate_codebooks, model
    )
    print("[info] installed optimized _generate_codebooks(topk)")


def _enable_fast_sdpa(model) -> None:
    """Enable fused SDPA in Audio8's fast codebook transformer layers."""
    fast_layers = getattr(model, "fast_layers", None)
    if fast_layers is None:
        print("[info] fast SDPA unavailable: model has no fast_layers")
        return
    enabled = 0
    for layer in fast_layers:
        attention = getattr(layer, "attention", None)
        if attention is not None and hasattr(attention, "use_sdpa"):
            attention.use_sdpa = True
            enabled += 1
    print(f"[info] enabled fast-layer SDPA ({enabled} layers)")


def _compile_fast_step(model) -> None:
    """Compile the small, repeatedly-called fast Transformer step."""
    def tensor_fast_step(self, hidden, cache_position):
        key_mask = torch.ones(
            (hidden.shape[0], self.config.num_codebooks),
            device=hidden.device,
            dtype=torch.bool,
        )
        rope = self.fast_freqs_cis[cache_position]
        mask = self._causal_mask(
            key_mask, cache_position, self.config.num_codebooks
        )
        for layer in self.fast_layers:
            hidden = layer(hidden, rope, mask, cache_position)
        return self.fast_output(self.fast_norm(hidden))[:, -1]

    original_fast_step = types.MethodType(tensor_fast_step, model)
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        model._fast_step = original_fast_step
        print("[info] torch.compile unavailable; using eager fast_step")
        return
    try:
        compiled_fast_step = compiler(
            original_fast_step,
            mode="default",
            fullgraph=False,
            dynamic=False,
        )

        compile_state = {"active": True}

        def safe_fast_step(*args, **kwargs):
            if compile_state["active"]:
                try:
                    return compiled_fast_step(*args, **kwargs)
                except Exception as exc:
                    compile_state["active"] = False
                    model._fast_step = original_fast_step
                    print(
                        f"[info] fast_step compile runtime fallback; "
                        f"using eager ({exc})"
                    )
                    return original_fast_step(*args, **kwargs)
            return original_fast_step(*args, **kwargs)

        model._fast_step = safe_fast_step
        print("[info] enabled torch.compile for fast_step")
    except Exception as exc:
        model._fast_step = original_fast_step
        print(f"[info] fast_step compile unavailable; using eager ({exc})")


def _compile_slow_step(model) -> None:
    """Compile the autoregressive semantic / slow Transformer step."""
    if not hasattr(torch, "compile"):
        return
    try:
        orig_slow_step = model._slow_step
        compiled_slow_step = torch.compile(
            orig_slow_step,
            mode="default",
            fullgraph=False,
            dynamic=True,
        )
        compile_state = {"active": True}

        def safe_slow_step(*args, **kwargs):
            if compile_state["active"]:
                try:
                    return compiled_slow_step(*args, **kwargs)
                except Exception as exc:
                    compile_state["active"] = False
                    model._slow_step = orig_slow_step
                    print(
                        f"[info] slow_step compile runtime fallback; "
                        f"using eager ({exc})"
                    )
                    return orig_slow_step(*args, **kwargs)
            return orig_slow_step(*args, **kwargs)

        model._slow_step = safe_slow_step
        print("[info] enabled torch.compile for _slow_step")
    except Exception as exc:
        print(f"[info] _slow_step compile unavailable; using eager ({exc})")


def _compile_codec(model) -> None:
    """Compile audio codec decoder (vocoder)."""
    if not hasattr(torch, "compile"):
        return
    if hasattr(model, "decode_audio"):
        try:
            orig_decode = model.decode_audio
            compiled_decode = torch.compile(
                orig_decode,
                mode="default",
                fullgraph=False,
                dynamic=True,
            )
            compile_state = {"active": True}

            def safe_decode(*args, **kwargs):
                if compile_state["active"]:
                    try:
                        return compiled_decode(*args, **kwargs)
                    except Exception as exc:
                        compile_state["active"] = False
                        model.decode_audio = orig_decode
                        print(
                            f"[info] decode_audio compile runtime fallback; "
                            f"using eager ({exc})"
                        )
                        return orig_decode(*args, **kwargs)
                return orig_decode(*args, **kwargs)

            model.decode_audio = safe_decode
            print("[info] enabled torch.compile for decode_audio (vocoder)")
        except Exception as exc:
            print(f"[info] decode_audio compile unavailable; using eager ({exc})")


def compile_audio8_model(model) -> None:
    """Compile performance-critical model components with torch.compile."""
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 64

    # fast_step is called 8-9 times per frame with fixed shapes, giving immediate
    # speedups with near-instant JIT compilation.
    _compile_fast_step(model)

    # slow_step contains complex 32-layer dynamic KV caches and symbolic mask tensors.
    # Its PyTorch FlashAttention SDPA is already extremely fast (RTF < 0.3).
    # Compiling it requires long symbolic tracing (sym_node) on first run.
    #
    # WARNING: measured on this workload, enabling this made things ~5x
    # SLOWER (RTF 1.0 -> 4.9), not faster. The AR loop feeds a fresh
    # `torch.tensor([prompt_width + step])` every iteration (see
    # _generate_streaming), so the growing cache position looks like a
    # new compile-time constant on (what appears to be) many steps
    # instead of a stable symbolic dim -- causing Dynamo to recompile
    # repeatedly across the generation loop rather than compiling once
    # and reusing the graph. Left here as an opt-in experiment only;
    # do not enable by default for this AR shape.
    if os.environ.get("AUDIO8_COMPILE_SLOW", "0") == "1":
        # Reduces (but does not eliminate) the decode_audio graph break
        # from `int(valid.sum().item())` by letting Dynamo capture the
        # scalar symbolically instead of guard-and-break. This is a
        # secondary issue -- the recompile storm above is the dominant
        # cost -- so don't expect this alone to fix RTF.
        torch._dynamo.config.capture_scalar_outputs = True
        _compile_slow_step(model)
        _compile_codec(model)


def _warmup_decode_audio_shapes(
    model,
    chunk_frames: int,
    overlap_frames: int,
) -> None:
    """Explicitly exercise model.decode_audio() on every distinct window
    length that _CodeStreamer actually produces during real streaming:
      - the first chunk (length == chunk_frames, no overlap yet)
      - a steady-state chunk (length == chunk_frames + overlap_frames)
      - a ragged tail chunk (some other, smaller length)

    Why this matters: torch.compile(dynamic=True) on decode_audio only
    generalizes a dimension to a true symbolic size after it has SEEN at
    least two different values for it. A warm-up that never produces more
    than one window shape (e.g. a very short throwaway sentence that ends
    before a full chunk_frames worth of audio is generated) leaves that
    generalization to happen on the first differently-shaped window of the
    REAL request instead -- which is exactly the mid-stream recompile that
    caused the playback stutter. Calling decode_audio directly with dummy
    codes sidesteps relying on how many frames a warm-up sentence happens
    to produce.
    """
    num_codebooks = int(model.config.num_codebooks)
    lengths = sorted({
        max(1, chunk_frames),
        max(1, chunk_frames + overlap_frames),
        max(1, chunk_frames - max(1, overlap_frames // 2)),
    })
    for length in lengths:
        dummy_codes = torch.zeros(
            (1, num_codebooks, length), dtype=torch.long, device=model.device
        )
        try:
            with torch.inference_mode():
                model.decode_audio(dummy_codes)
        except Exception as exc:
            print(f"[info] decode_audio warm-up shape={length} failed (continuing): {exc}")


def warmup_model(
    model,
    processor,
    voices_dir: Path,
    voice_name: str,
    max_new_tokens: int = 24,
    chunk_frames: int = 16,
    overlap_frames: int = 8,
) -> None:
    """Run one short, silent, throwaway generation to pay any torch.compile
    first-call compilation cost up front, so it never gets folded into the
    RTF measurement (or the audible playback) of the real request.

    Also explicitly warms decode_audio on every window shape the real
    streaming pipeline will use (see _warmup_decode_audio_shapes) -- a
    short warm-up sentence alone isn't reliable for this, since it may not
    generate enough frames to ever hit more than one chunk shape.

    Safe to call even when nothing is compiled -- it just costs a small
    eager forward pass in that case.
    """
    print("[info] warm-up pass (discarded output, primes torch.compile)...")
    t0 = time.time()
    try:
        stream_synthesize(
            model=model,
            processor=processor,
            text="Hello.",
            voice_name=voice_name,
            voices_dir=voices_dir,
            max_new_tokens=max_new_tokens,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
            play=False,
            save_to=None,
        )
    except Exception as exc:
        print(f"[info] warm-up pass raised (continuing anyway): {exc}")
    _warmup_decode_audio_shapes(model, chunk_frames, overlap_frames)
    print(f"[info] warm-up done in {time.time() - t0:.1f}s")


def load_compile_cache(path: Path = COMPILE_CACHE_PATH) -> bool:
    """Pre-populate torch.compile's caches (Dynamo guards, AOTAutograd,
    Inductor, Triton, autotuning results) from a snapshot saved by
    save_compile_cache() in a previous run.

    Call this BEFORE compile_audio8_model()/warmup_model(). If a call's
    input shapes exactly match what was captured in the snapshot, that call
    can skip cold compilation almost entirely; shapes never seen before
    still compile normally (and can be folded into a future snapshot).
    Safe no-op if the API is unavailable in this torch build or no
    snapshot exists yet.
    """
    if not hasattr(torch, "compiler") or not hasattr(torch.compiler, "load_cache_artifacts"):
        print("[info] torch.compiler cache-artifacts API unavailable in this torch build")
        return False
    if not path.is_file():
        print(f"[info] no compile cache at {path} yet (first run will create one)")
        return False
    try:
        artifact_bytes = path.read_bytes()
        torch.compiler.load_cache_artifacts(artifact_bytes)
        print(f"[info] loaded compile cache from {path} ({len(artifact_bytes) / 1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"[info] failed to load compile cache, continuing cold ({exc})")
        return False


def save_compile_cache(path: Path = COMPILE_CACHE_PATH) -> None:
    """Snapshot torch.compile's current caches to disk.

    Call this AFTER at least one successful compiled forward pass (right
    after warmup_model() is the natural spot) so the next process launch
    can call load_compile_cache() and skip most of the cold-compile cost
    for any shape already exercised in this run.

    NOTE: snapshots are only valid for the same torch/CUDA/GPU combination
    they were captured on -- if you change hardware or upgrade torch,
    delete the old file and let it regenerate.
    """
    if not hasattr(torch, "compiler") or not hasattr(torch.compiler, "save_cache_artifacts"):
        return
    try:
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            print("[info] nothing to save yet (no compiled artifacts produced)")
            return
        artifact_bytes, _cache_info = artifacts
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact_bytes)
        print(f"[info] saved compile cache to {path} ({len(artifact_bytes) / 1e6:.1f} MB)")
    except Exception as exc:
        print(f"[info] failed to save compile cache: {exc}")


class _PlaybackBuffer:
    """Low-overhead PCM queue with seamless playback and graceful completion.

    Guarantees that all streamed audio chunks are physically played to the
    very end before the stream is closed and output files are written.

    Also tolerates real ALSA/PortAudio xruns (hardware buffer underruns),
    which are common on WSL2/WSLg's virtual audio device: if PortAudio's
    own internal recovery (AlsaRestart) fails and the stream silently dies,
    a watchdog notices and recreates the OutputStream so playback of the
    still-buffered audio can resume, instead of the rest of the audio
    going permanently silent.
    """

    def __init__(
        self,
        sample_rate: int,
        prefetch_seconds: float = 1.0,
        max_buffer_seconds: float = 8.0,
        blocksize: int = 2048,
        latency: float | str = 0.3,
        max_restart_attempts: int = 5,
    ):
        import collections

        self._sample_rate = sample_rate
        self._prefetch_samples = max(1, int(sample_rate * prefetch_seconds))
        self._max_buffer_samples = max(
            self._prefetch_samples,
            int(sample_rate * max_buffer_seconds),
        )
        self._buffer: collections.deque[np.ndarray] = collections.deque()
        self._pending_samples = 0
        self._total_samples_fed = 0
        self._started = False
        self._closed = False
        self._start_perf_time: float | None = None
        self._is_buffering = True
        self._lock = threading.Lock()
        self._stream = None
        self._blocksize = blocksize
        # A numeric latency (seconds) is a real request to PortAudio for a
        # bigger hardware buffer. The string "high" is only a hint and, on
        # WSL2/WSLg's virtual audio device, was observed not to be enough
        # to avoid xruns -- an explicit value gives more headroom against
        # GIL contention from the decode/generation threads.
        self._latency = latency
        self._max_restart_attempts = max(0, int(max_restart_attempts))
        self._restart_count = 0
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

    def buffered_seconds(self) -> float:
        with self._lock:
            return max(0, self._pending_samples) / self._sample_rate

    def feed(self, chunk: np.ndarray) -> None:
        if self._closed or chunk.size == 0:
            return
        chunk = np.ascontiguousarray(
            np.clip(chunk, -1.0, 1.0).astype(np.float32, copy=False)
        )

        while not self._closed:
            with self._lock:
                if self._pending_samples + chunk.size <= self._max_buffer_samples:
                    self._buffer.append(chunk)
                    self._pending_samples += chunk.size
                    self._total_samples_fed += chunk.size
                    if not self._started and self._pending_samples >= self._prefetch_samples:
                        self._start()
                    break
            time.sleep(0.002)

    def _audio_callback(self, outdata, frames, time_info, status):
        if status:
            # status carries flags like output_underflow. We can't fix an
            # xrun from inside the callback (that's PortAudio/ALSA's own
            # recovery path, which is what failed) -- just note it so the
            # watchdog's restart, if one follows, is explainable in logs.
            print(f"[stream] audio callback status: {status}")

        needed = frames
        out_ptr = 0

        with self._lock:
            if self._is_buffering:
                if self._pending_samples < self._prefetch_samples and not self._closed:
                    outdata.fill(0)
                    return
                self._is_buffering = False

            while needed > 0 and self._buffer:
                chunk = self._buffer[0]
                chunk_len = len(chunk)
                if chunk_len <= needed:
                    outdata[out_ptr : out_ptr + chunk_len, 0] = chunk
                    out_ptr += chunk_len
                    needed -= chunk_len
                    self._pending_samples = max(0, self._pending_samples - chunk_len)
                    self._buffer.popleft()
                else:
                    outdata[out_ptr : out_ptr + needed, 0] = chunk[:needed]
                    self._buffer[0] = chunk[needed:]
                    self._pending_samples = max(0, self._pending_samples - needed)
                    out_ptr += needed
                    needed = 0

            if needed > 0:
                outdata[out_ptr:, 0] = 0
                if not self._closed:
                    self._is_buffering = True

    def _open_stream(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._blocksize,
            latency=self._latency,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _start(self) -> None:
        if self._started:
            return
        try:
            self._open_stream()
            self._started = True
            self._start_perf_time = time.perf_counter()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name="audio8-playback-watchdog", daemon=True
            )
            self._watchdog_thread.start()
        except Exception as exc:
            print(f"[stream] warning: audio device start failed ({exc})")

    def _watchdog_loop(self) -> None:
        """Detect a PortAudio stream that died underneath us (e.g. because
        its own ALSA-xrun recovery failed) and recreate it so any audio
        still sitting in self._buffer isn't stranded and lost in silence.
        """
        while not self._watchdog_stop.wait(0.5):
            if self._closed:
                return
            stream = self._stream
            if stream is None:
                continue
            try:
                alive = stream.active
            except Exception:
                alive = False
            if alive:
                continue
            # Stream is no longer active but we didn't ask it to stop.
            with self._lock:
                still_have_audio = self._pending_samples > 0 or bool(self._buffer)
            if not still_have_audio:
                continue
            if self._restart_count >= self._max_restart_attempts:
                print(
                    "[stream] warning: audio stream died and max restart "
                    f"attempts ({self._max_restart_attempts}) exhausted; "
                    "remaining buffered audio will not be played."
                )
                return
            self._restart_count += 1
            print(
                f"[stream] audio stream died unexpectedly (likely an ALSA "
                f"xrun the driver couldn't recover from) -- restarting "
                f"(attempt {self._restart_count}/{self._max_restart_attempts})"
            )
            try:
                try:
                    stream.close()
                except Exception:
                    pass
                self._is_buffering = True  # re-prime prefetch on the new stream
                self._open_stream()
            except Exception as exc:
                print(f"[stream] warning: audio stream restart failed ({exc})")

    def close(self, drain_timeout: float | None = None) -> None:
        if self._closed:
            return
        # Set this BEFORE the drain wait, not after: _audio_callback uses
        # `not self._closed` to decide whether a short tail (less than a
        # full prefetch_seconds worth) is still allowed to play immediately
        # instead of waiting to re-fill the normal prefetch threshold. If
        # this flag isn't set until after the wait loop, a underrun near
        # the very end of the stream (common, since the last chunk is
        # often shorter than prefetch_seconds) makes the callback wait
        # forever for data that will never arrive -- which is exactly the
        # "timed out waiting for playback to drain" case.
        self._closed = True

        with self._lock:
            if not self._started and self._pending_samples > 0:
                self._start()

        # Wait until EVERY SINGLE audio sample in self._buffer is physically
        # consumed by the callback -- but bounded, in case the stream keeps
        # dying and the watchdog exhausts its restart attempts, so a broken
        # audio device can't hang the whole program forever.
        if drain_timeout is None:
            drain_timeout = max(30.0, self._max_buffer_samples / self._sample_rate * 4.0)
        deadline = time.perf_counter() + drain_timeout
        while True:
            with self._lock:
                done = (len(self._buffer) == 0 and self._pending_samples <= 0)
            if done or not self._started:
                break
            if time.perf_counter() > deadline:
                print(
                    "[stream] warning: timed out waiting for playback to "
                    "drain (audio device likely unrecoverable); closing anyway"
                )
                break
            time.sleep(0.05)

        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2.0)

        # Allow 0.5s for the physical sound card / ALSA DMA buffer to complete playback
        time.sleep(0.5)

        if self._stream is not None:
            try:
                # stop() blocks until buffered audio has actually finished
                # playing (drains the driver/ALSA buffer first). abort()
                # stops immediately and discards whatever is still sitting
                # in that buffer, which is what was truncating the tail of
                # playback right before output.wav got written.
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class _CodeStreamer:
    """Async codec decoder + playback pipeline.

    Important performance change versus the original implementation:
    decode_audio() is NOT called from the AR generation loop. Generated code
    frames are batched and handed to a decoder worker, allowing AR generation
    to continue while the previous audio chunk is decoded.

    The decoder receives the complete overlap window, but only the newly
    generated frames are emitted. A small tail is held back so crossfade does
    not duplicate already-played samples.
    """

    def __init__(
        self,
        model,
        num_codebooks: int,
        *,
        chunk_frames: int = 16,
        overlap_frames: int = 8,
        crossfade_ms: float = 20.0,
        prefetch_seconds: float = 0.8,
        max_buffer_seconds: float = 4.0,
        max_decode_queue: int = 2,
        play: bool = True,
        debug: bool = True,
    ):
        import queue
        import threading

        self.model = model
        self.num_codebooks = num_codebooks
        self.chunk_frames = max(1, int(chunk_frames))
        self.overlap_frames = max(0, int(overlap_frames))
        self.play = play
        self.debug = debug
        self.samples_per_frame = int(model.config.codec_frame_size)
        sample_rate = int(model.config.codec_sample_rate)
        self._crossfade_samples = max(
            0, int(sample_rate * crossfade_ms / 1000.0)
        )

        # Keep generated frames on GPU. The original code copied every single
        # codebook frame to CPU, which introduces a synchronization point on
        # every AR step. We only transfer one batched window per decode.
        self.frames: list[torch.Tensor] = []
        self.emitted_frames = 0
        self._tail: np.ndarray | None = None

        self._decode_queue: queue.Queue[tuple[torch.Tensor, int, int, int] | None] = queue.Queue(
            maxsize=max(1, int(max_decode_queue))
        )
        self._decode_thread = threading.Thread(
            target=self._decode_worker,
            name="audio8-codec",
            daemon=True,
        )
        self._decode_started = False
        self._decode_closed = False

        self._playback = (
            _PlaybackBuffer(
                sample_rate,
                prefetch_seconds=prefetch_seconds,
                max_buffer_seconds=max_buffer_seconds,
            )
            if play
            else None
        )

    def _start_decoder(self) -> None:
        if not self._decode_started:
            self._decode_started = True
            self._decode_thread.start()

    def put(self, value: torch.Tensor) -> None:
        if self.debug and not self.frames:
            print(
                f"[stream] first put() value shape={tuple(value.shape)} "
                f"dtype={value.dtype}"
            )

        step = value.reshape(-1)[: self.num_codebooks]
        if step.numel() < self.num_codebooks:
            return

        # Do NOT .to("cpu") here. That was a synchronization point for every
        # generated frame in the previous implementation.
        self.frames.append(step.detach())

        if len(self.frames) % self.chunk_frames != 0:
            return

        self._submit_decode()

    def _submit_decode(self) -> None:
        total_frames = len(self.frames)
        if total_frames <= self.emitted_frames:
            return

        start_frame = max(0, self.emitted_frames - self.overlap_frames)
        # One GPU tensor for the whole decode window; transfer/dispatch happens
        # once per chunk instead of once per generated frame.
        window = torch.stack(
            self.frames[start_frame:total_frames], dim=1
        ).unsqueeze(0)

        self._start_decoder()
        # Bounded queue: if the decoder falls behind, this is the only place
        # that applies backpressure. Generation never waits for each decode.
        self._decode_queue.put((window, start_frame, total_frames, total_frames - self.emitted_frames))

        self.emitted_frames = total_frames

    def _decode_worker(self) -> None:
        while True:
            item = self._decode_queue.get()
            try:
                if item is None:
                    return
                window, start_frame, total_frames, new_frames = item

                with torch.inference_mode():
                    waveforms, waveform_lengths = self.model.decode_audio(window)

                audio = (
                    waveforms[0, : int(waveform_lengths[0])]
                    .float()
                    .cpu()
                    .numpy()
                )
                self._emit_decoded(start_frame, total_frames, new_frames, audio)
            finally:
                self._decode_queue.task_done()

    def _emit_decoded(
        self, start_frame: int, total_frames: int, new_frames: int, audio: np.ndarray
    ) -> None:
        # Only the newly generated frames are emitted; overlap frames are
        # context for the decoder and are never played twice.
        spf = self.samples_per_frame
        # emitted boundary at submission time was the previous total_frames
        # minus the newly generated count. FIFO processing means this can be
        # reconstructed from the window size and overlap policy.
        emitted_before = total_frames - new_frames
        new_start = (emitted_before - start_frame) * spf
        slice_end = min(new_start + new_frames * spf, audio.size)

        if new_start >= slice_end:
            return

        chunk = np.ascontiguousarray(audio[new_start:slice_end])

        # Hold back the tail so the next chunk can crossfade into it.
        n = min(self._crossfade_samples, chunk.size)
        if n > 0:
            current_tail = chunk[-n:].copy()
            body = chunk[:-n]
        else:
            current_tail = None
            body = chunk

        if self._tail is not None and body.size > 0:
            ncf = min(self._tail.size, body.size)
            cross = _crossfade_pair(self._tail, body, ncf)
            body = np.concatenate((cross, body[ncf:]))

        if self._playback is not None and body.size:
            self._playback.feed(body)

        if current_tail is not None:
            self._tail = current_tail

    def end(self) -> None:
        if len(self.frames) > self.emitted_frames:
            self._submit_decode()

        if self._decode_started and not self._decode_closed:
            self._decode_queue.join()
            self._decode_queue.put(None)
            self._decode_thread.join(timeout=120)
            self._decode_closed = True

        # Flush the held tail only after every decoded chunk has been played.
        if self._playback is not None:
            if self._tail is not None and self._tail.size:
                self._playback.feed(self._tail)
                self._tail = None
            self._playback.close()

    def full_codes(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(
                (self.num_codebooks, 0), dtype=np.int64
            )
        return (
            torch.stack(self.frames, dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

def _generate_streaming(
    model: "AutoModel",
    streamer: _CodeStreamer,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    generator=None,
    **inputs,
) -> None:
    """Run Audio8 token generation and feed each frame to *streamer*."""
    from torch import Tensor
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.generation import StoppingCriteriaList
    _reset_fast_profile(model.device)
    prompt, prompt_mask = model._prepare_prompt(**inputs)
    batch_size, _, prompt_width = prompt.shape
    if prompt_width >= model.config.max_seq_len:
        raise ValueError(
            f"Prompt length {prompt_width} must be smaller than {model.config.max_seq_len}"
        )
    max_new_tokens = min(max_new_tokens, model.config.max_seq_len - prompt_width)
    model._setup_generation_caches(
        batch_size, prompt_width + max_new_tokens, next(model.parameters()).dtype
    )
    semantic_processors = model._as_processor_list(None)
    codebook_processors = model._as_processor_list(None)
    criteria = StoppingCriteriaList()
    cache_position = torch.arange(prompt_width, device=model.device, dtype=torch.long)
    position_ids = prompt_mask.cumsum(-1).sub(1).clamp_min(0)

    # Measure model generation separately from streamer.end(), which may
    # wait for codec decoding and audio playback to drain.
    timing_on_cuda = model.device.type == "cuda"
    if timing_on_cuda:
        generation_start_event = torch.cuda.Event(enable_timing=True)
        generation_end_event = torch.cuda.Event(enable_timing=True)
        generation_start_event.record()
        slow_step_seconds = 0.0
        codebook_seconds = 0.0
        slow_step_events = []
        codebook_events = []
    else:
        generation_start_time = time.perf_counter()
        slow_step_seconds = 0.0
        codebook_seconds = 0.0

    # Allow Flash / MemEfficient SDPA if supported by hardware
    sdp_backends = []
    if torch.cuda.is_available():
        if torch.backends.cuda.flash_sdp_enabled():
            for name in ("FLASH_ATTENTION", "FLASH"):
                if hasattr(SDPBackend, name):
                    sdp_backends.append(getattr(SDPBackend, name))
                    break
        if torch.backends.cuda.mem_efficient_sdp_enabled():
            for name in ("EFFICIENT_ATTENTION", "MEM_EFFICIENT"):
                if hasattr(SDPBackend, name):
                    sdp_backends.append(getattr(SDPBackend, name))
                    break
        if hasattr(SDPBackend, "CUDNN_ATTENTION") and getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: False)():
            sdp_backends.append(SDPBackend.CUDNN_ATTENTION)
        if torch.backends.cuda.math_sdp_enabled() and hasattr(SDPBackend, "MATH"):
            sdp_backends.append(SDPBackend.MATH)
    if not sdp_backends and hasattr(SDPBackend, "MATH"):
        sdp_backends = [SDPBackend.MATH]

    if timing_on_cuda:
        component_start = torch.cuda.Event(enable_timing=True)
        component_end = torch.cuda.Event(enable_timing=True)
        component_start.record()
    else:
        slow_start_time = time.perf_counter()
    with sdpa_kernel(sdp_backends):
        logits, slow_hidden = model._slow_step(prompt, cache_position, position_ids, prompt_mask)
    if timing_on_cuda:
        component_end.record()
        slow_step_events.append((component_start, component_end))
    else:
        slow_step_seconds += time.perf_counter() - slow_start_time
    # Keep the history in fixed-size buffers.  Appending with torch.cat() on
    # every AR step reallocates and copies the complete history, turning a
    # long generation into an avoidable O(n^2) copy workload.
    history_capacity = prompt_width + max_new_tokens
    semantic_history_buffer = torch.empty(
        (batch_size, history_capacity), dtype=prompt.dtype, device=model.device
    )
    semantic_history_buffer[:, :prompt_width].copy_(prompt[:, 0])
    history_length = prompt_width

    prompt_mask_buffer = torch.empty(
        (batch_size, history_capacity), dtype=prompt_mask.dtype, device=model.device
    )
    prompt_mask_buffer[:, :prompt_width].copy_(prompt_mask)
    prompt_lengths = prompt_mask.sum(-1)
    previous = None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=model.device)
    pending_codebooks = None
    pending_emitted = None
    try:
        for step in range(max_new_tokens):
            active_before = ~finished
            semantic_history = semantic_history_buffer[:, :history_length]
            semantic = model._sample_semantic(
                semantic_history, logits, semantic_processors, top_k, top_p,
                temperature, previous, do_sample, generator,
            )
            if timing_on_cuda:
                component_start = torch.cuda.Event(enable_timing=True)
                component_end = torch.cuda.Event(enable_timing=True)
                component_start.record()
            else:
                codebook_start_time = time.perf_counter()
            codebooks = model._generate_codebooks(
                slow_hidden, semantic, codebook_processors, top_k, top_p,
                temperature, do_sample, generator,
            )
            if timing_on_cuda:
                component_end.record()
                codebook_events.append((component_start, component_end))
            else:
                codebook_seconds += time.perf_counter() - codebook_start_time
            emitted = active_before & (semantic != model.config.eos_token_id)
            # Hold one frame back so the normal path does not synchronize on
            # emitted[0].item() every step.  Reaching the next iteration means
            # the previous frame was valid; the final pending frame is checked
            # once in the cleanup path below.
            if batch_size == 1 and pending_codebooks is not None:
                streamer.put(pending_codebooks)
            if batch_size == 1:
                pending_codebooks = codebooks[0]
                pending_emitted = emitted[0]
            semantic_history_buffer[:, history_length] = semantic
            history_length += 1

            if previous is None:
                previous = torch.zeros(
                    (batch_size, model.config.ras_window_size),
                    dtype=torch.long,
                    device=model.device,
                )
            else:
                previous = previous.roll(-1, dims=1)
                previous[:, -1] = semantic
            finished |= semantic.eq(model.config.eos_token_id)
            if criteria:
                stopped = criteria(
                    semantic_history_buffer[:, :history_length], logits
                )
                if not isinstance(stopped, Tensor):
                    stopped = torch.full_like(finished, bool(stopped))
                finished |= stopped.to(device=model.device, dtype=torch.bool)
            if finished.all():
                break

            next_column = torch.cat((semantic[:, None], codebooks), dim=1).unsqueeze(-1)
            new_valid = active_before.long()[:, None]
            prompt_mask_buffer[:, prompt_width + step] = new_valid[:, 0]
            prompt_mask = prompt_mask_buffer[:, : prompt_width + step + 1]
            physical_position = torch.tensor([prompt_width + step], device=model.device)
            token_position = (prompt_lengths + step)[:, None]
            if timing_on_cuda:
                component_start = torch.cuda.Event(enable_timing=True)
                component_end = torch.cuda.Event(enable_timing=True)
                component_start.record()
            else:
                slow_start_time = time.perf_counter()
            with sdpa_kernel(sdp_backends):
                logits, slow_hidden = model._slow_step(
                    next_column, physical_position, token_position, prompt_mask
                )
            if timing_on_cuda:
                component_end.record()
                slow_step_events.append((component_start, component_end))
            else:
                slow_step_seconds += time.perf_counter() - slow_start_time
    finally:
        if (
            batch_size == 1
            and pending_codebooks is not None
            and pending_emitted is not None
            and bool(pending_emitted.item())
        ):
            streamer.put(pending_codebooks)

        if timing_on_cuda:
            generation_end_event.record()
            generation_end_event.synchronize()
            generation_seconds = generation_start_event.elapsed_time(
                generation_end_event
            ) / 1000.0
            slow_step_seconds = sum(
                start.elapsed_time(end) for start, end in slow_step_events
            ) / 1000.0
            codebook_seconds = sum(
                start.elapsed_time(end) for start, end in codebook_events
            ) / 1000.0
        else:
            generation_seconds = time.perf_counter() - generation_start_time

        generated_frames = len(streamer.frames)
        frame_period = (
            float(model.config.codec_frame_size)
            / float(model.config.codec_sample_rate)
        )
        generated_audio_seconds = generated_frames * frame_period
        rtf = (
            generation_seconds / generated_audio_seconds
            if generated_audio_seconds > 0
            else float("inf")
        )
        print(
            f"[stream] generation={generation_seconds:.3f}s  "
            f"frames={generated_frames}  "
            f"audio={generated_audio_seconds:.3f}s  "
            f"frame_period={frame_period * 1000.0:.2f}ms  RTF={rtf:.3f}"
        )
        print(
            f"[stream] component_time slow_step={slow_step_seconds:.3f}s  "
            f"codebooks={codebook_seconds:.3f}s"
        )
        _report_fast_profile()
        if timing_on_cuda:
            print(
                "[stream] CUDA SDPA enabled: "
                f"flash={torch.backends.cuda.flash_sdp_enabled()}  "
                f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}  "
                f"math={torch.backends.cuda.math_sdp_enabled()}"
            )
        streamer.end()


def stream_synthesize(
    model: "AutoModel",
    processor: "AutoProcessor",
    text: str,
    voice_name: str,
    voices_dir: Path,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    seed: int | None = 1234,
    chunk_frames: int = 16,
    overlap_frames: int = 8,
    crossfade_ms: float = 20.0,
    prefetch_seconds: float = 0.8,
    max_buffer_seconds: float = 4.0,
    max_decode_queue: int = 2,
    play: bool = True,
    save_to: str | Path | None = None,
) -> np.ndarray:
    """Generate speech from *text* and decode+play audio incrementally as
    codes are produced, instead of waiting for the whole sequence.

    Streaming playback knobs:
      chunk_frames     — decode/play every N codec frames (~46 ms each)
      overlap_frames   — re-decode context to stabilise chunk boundaries
      crossfade_ms     — overlap blend between consecutive chunks
      prefetch_seconds — buffer this much audio before starting playback
    """
    voice_dir = voices_dir / voice_name
    codes_path = voice_dir / "codes.npy"
    meta_path = voice_dir / "meta.json"
    if not codes_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Voice '{voice_name}' not found in {voices_dir}. Call register_voice() first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    reference_text = meta["reference_text"]
    num_codebooks = int(meta["shape"][0])

    inputs = processor(
        text=[text],
        reference_text=[reference_text],
        reference_codes=[str(codes_path)],
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if seed is not None:
        torch.manual_seed(seed)
        if model.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    streamer = _CodeStreamer(
        model,
        num_codebooks,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        crossfade_ms=crossfade_ms,
        prefetch_seconds=prefetch_seconds,
        max_buffer_seconds=max_buffer_seconds,
        max_decode_queue=max_decode_queue,
        play=play,
    )

    print(f"[stream] voice='{voice_name}'  text='{text}'")
    t0 = time.time()
    with torch.inference_mode():
        _generate_streaming(
            model,
            streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            **inputs,
        )
    print(f"[stream] generated in {time.time() - t0:.1f}s, frames={len(streamer.frames)}")

    if not streamer.frames:
        raise RuntimeError("generation produced no audio frames")

    codes_np = streamer.full_codes()
    if save_to:
        with torch.inference_mode():
            waveforms, waveform_lengths = model.decode_audio(
                torch.from_numpy(codes_np).unsqueeze(0).to(model.device)
            )
        audio = waveforms[0, : int(waveform_lengths[0])].float().cpu().numpy()
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(save_to), audio, model.config.codec_sample_rate)
        print(f"[stream] also saved full audio to {save_to}")

    return codes_np


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
_install_optimized_codebook_generation(model)
load_compile_cache()  # populate caches BEFORE compiling / first call
compile_audio8_model(model)

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

# Step 1b — warm up torch.compile (if any modules are compiled) on a
# short throwaway sentence, so compilation cost is not counted in the
# real request's RTF or heard as playback stutter.
warmup_model(
    model=model,
    processor=processor,
    voices_dir=VOICES_DIR,
    voice_name=VOICE_NAME,
)
save_compile_cache()  # snapshot for the next process launch

# Step 2b — same thing, but streamed: decodes + plays audio as codes are
# produced instead of waiting for the full sequence. Requires:
#   pip install sounddevice
# Uncomment to use instead of / in addition to the blocking call above.
#
stream_synthesize(
    model=model,
    processor=processor,
    text=TEXT,
    voice_name=VOICE_NAME,
    voices_dir=VOICES_DIR,
    max_new_tokens=1024,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    seed=random.randint(0, 1000),
    chunk_frames=32,
    overlap_frames=16,
    crossfade_ms=20.0,
    prefetch_seconds=1.0,
    max_buffer_seconds=6.0,
    max_decode_queue=2,
    play=True,
    save_to=OUTPUT_PATH,
)
