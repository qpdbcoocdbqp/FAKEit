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

TEXT = "最高の音質体験をしていただくために、本物をサポートしてください。良いアニメーション、良い音楽、忘れられない思い出。忘れられないことを願っています。"
OUTPUT_PATH = "output.wav"

# Reference voice for zero-shot voice cloning.
REFERENCE_AUDIO = "docs/tts/resource/reference.wav"
# Must be the exact words spoken in REFERENCE_AUDIO.
REFERENCE_TEXT = "突然轉錯帳可能是某個系統整個當掉結果回頭一查才發現寫這段的是AI而唯一該把關的人從頭到尾沒看過他那這個鍋到底該誰扛"


# Where registered voices are stored.
VOICES_DIR = Path("./docs/tts/voices")
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


class _PlaybackBuffer:
    """Low-overhead PCM queue with a background PortAudio writer.

    The producer never calls sounddevice.write(), so model inference is not
    blocked by audio I/O. The queue is intentionally bounded to expose
    underruns/backpressure instead of growing without limit.
    """

    def __init__(
        self,
        sample_rate: int,
        prefetch_seconds: float = 0.8,
        max_buffer_seconds: float = 4.0,
        blocksize: int = 2048,
    ):
        import queue
        import threading

        self._sample_rate = sample_rate
        self._prefetch_samples = max(1, int(sample_rate * prefetch_seconds))
        self._max_buffer_samples = max(
            self._prefetch_samples,
            int(sample_rate * max_buffer_seconds),
        )
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._pending = 0
        self._started = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._stream = None
        self._blocksize = blocksize
        self._lock = threading.Lock()

    def buffered_seconds(self) -> float:
        with self._lock:
            return self._pending / self._sample_rate

    def feed(self, chunk: np.ndarray) -> None:
        if self._closed or chunk.size == 0:
            return
        chunk = np.ascontiguousarray(
            np.clip(chunk, -1.0, 1.0).astype(np.float32, copy=False)
        )

        # Don't allow an accidental producer burst to create an unbounded
        # latency queue. Waiting here is preferable to accumulating seconds
        # of stale audio.
        while not self._closed:
            with self._lock:
                enough_room = (
                    self._pending + chunk.size <= self._max_buffer_samples
                )
            if enough_room:
                break
            time.sleep(0.002)

        self._queue.put(chunk)
        with self._lock:
            self._pending += chunk.size

        if not self._started and self._pending >= self._prefetch_samples:
            self._start()

    def _start(self) -> None:
        import sounddevice as sd
        import threading

        if self._started:
            return
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._blocksize,
        )
        self._stream.start()
        self._started = True
        self._thread = threading.Thread(
            target=self._worker,
            name="audio8-playback",
            daemon=True,
        )
        self._thread.start()

    def _worker(self) -> None:
        while True:
            chunk = self._queue.get()
            try:
                if chunk is None:
                    return
                self._stream.write(chunk.reshape(-1, 1))
                with self._lock:
                    self._pending = max(0, self._pending - chunk.size)
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._started and self._pending > 0:
            self._start()
        if self._started:
            self._queue.join()
            self._queue.put(None)
            if self._thread is not None:
                self._thread.join(timeout=60)
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()


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
    logits, slow_hidden = model._slow_step(prompt, cache_position, position_ids, prompt_mask)
    semantic_history = prompt[:, 0]
    prompt_lengths = prompt_mask.sum(-1)
    previous = None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=model.device)
    try:
        for step in range(max_new_tokens):
            active_before = ~finished
            semantic = model._sample_semantic(
                semantic_history, logits, semantic_processors, top_k, top_p,
                temperature, previous, do_sample, generator,
            )
            codebooks = model._generate_codebooks(
                slow_hidden, semantic, codebook_processors, top_k, top_p,
                temperature, do_sample, generator,
            )
            emitted = active_before & (semantic != model.config.eos_token_id)
            if batch_size == 1 and bool(emitted[0].item()):
                streamer.put(codebooks[0])
            semantic_history = torch.cat((semantic_history, semantic[:, None]), dim=1)

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
                stopped = criteria(semantic_history, logits)
                if not isinstance(stopped, Tensor):
                    stopped = torch.full_like(finished, bool(stopped))
                finished |= stopped.to(device=model.device, dtype=torch.bool)
            if finished.all():
                break

            next_column = torch.cat((semantic[:, None], codebooks), dim=1).unsqueeze(-1)
            new_valid = active_before.long()[:, None]
            prompt_mask = torch.cat((prompt_mask, new_valid), dim=1)
            physical_position = torch.tensor([prompt_width + step], device=model.device)
            token_position = (prompt_lengths + step)[:, None]
            with sdpa_kernel(SDPBackend.MATH):
                logits, slow_hidden = model._slow_step(
                    next_column, physical_position, token_position, prompt_mask
                )
    finally:
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
    chunk_frames=16,
    overlap_frames=8,
    crossfade_ms=20.0,
    prefetch_seconds=0.8,
    max_buffer_seconds=4.0,
    max_decode_queue=2,
    play=True,
    save_to=OUTPUT_PATH,
)
