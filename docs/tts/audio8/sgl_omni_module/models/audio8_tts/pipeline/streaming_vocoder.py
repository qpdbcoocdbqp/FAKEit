# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sglang_omni.models.audio8_tts.pipeline.state_io import load_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class _Audio8StreamState:
    """Per-request streaming state for Audio8 vocoder."""
    frames: list[torch.Tensor] = field(default_factory=list)
    emitted_samples: int = 0


class Audio8StreamingVocoderScheduler(
    StreamingVocoderBase[_Audio8StreamState, None]
):
    """Incremental vocoder for Audio8-TTS using StreamingVocoderBase."""

    def __init__(
        self,
        codec: Any,
        *,
        device: str,
        eos_token_id: int,
        num_codebooks: int,
        sample_rate: int,
        chunk_frames: int = 12,
        context_frames: int = 128,
        guard_frames: int = 1,
        hop_length: int = 2048,
    ) -> None:
        self._codec = codec
        self._device = torch.device(device)
        self._eos_token_id = int(eos_token_id)
        self._num_codebooks = int(num_codebooks)
        self._chunk_frames = max(int(chunk_frames), 1)
        self._context_frames = max(int(context_frames), 0)
        self._guard_samples = max(int(guard_frames), 0) * int(hop_length)
        self._hop_length = int(hop_length)

        super().__init__(
            self._vocode_payload,
            sample_rate=sample_rate,
            stream_source_hint="Audio8-TTS",
        )

    # ------------------------------------------------------------------
    # StreamingVocoderBase abstract methods
    # ------------------------------------------------------------------

    def create_stream_state(self, request_id: str) -> _Audio8StreamState:
        del request_id
        return _Audio8StreamState()

    def validate_chunk(
        self,
        request_id: str,
        state: _Audio8StreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        # codes: [num_codebooks+1, T] from Audio8StepOutput
        codes = codes.to(dtype=torch.long)
        if codes.ndim == 1:
            codes = codes.unsqueeze(-1)
        if codes.shape[0] < self._num_codebooks + 1:
            raise ValueError(
                f"Audio8 stream chunk has {codes.shape[0]} rows, "
                f"expected >= {self._num_codebooks + 1}"
            )
        return codes

    def ingest(
        self,
        request_id: str,
        state: _Audio8StreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        # codes: [num_codebooks+1, T]; semantic is row 0, VQ are rows 1..
        for t in range(codes.shape[1]):
            semantic = int(codes[0, t].item())
            if semantic == self._eos_token_id:
                continue
            frame = codes[1 : self._num_codebooks + 1, t].clone()
            state.frames.append(frame)

    def should_decode(self, state: _Audio8StreamState, *, is_final: bool) -> bool:
        if is_final:
            return bool(state.frames)
        return len(state.frames) % self._chunk_frames == 0 and bool(state.frames)

    def decode_delta(
        self,
        request_id: str,
        state: _Audio8StreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        if not state.frames:
            return None
        frames = state.frames
        end = len(frames)
        start = max(0, end - self._context_frames - self._chunk_frames)
        audio = self._decode_frames_sync(frames, start, end)
        absolute_start = start * self._hop_length
        stable_end = max(0, len(audio) - self._guard_samples) if not is_final else len(audio)
        begin = max(0, state.emitted_samples - absolute_start)
        if stable_end <= begin:
            return None
        chunk = np.ascontiguousarray(audio[begin:stable_end])
        state.emitted_samples = absolute_start + stable_end
        return torch.from_numpy(chunk)

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _Audio8StreamState,
    ) -> dict[str, Any]:
        del request_id, state
        original_state = load_state(payload)
        data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        prompt_tokens = (
            len(original_state.input_ids) if original_state.input_ids is not None else 0
        )
        data["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": original_state.completion_tokens or 0,
            "total_tokens": prompt_tokens + (original_state.completion_tokens or 0),
        }
        return data

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: _Audio8StreamState,
    ) -> torch.Tensor | None:
        del request_id, payload
        if not state.frames:
            return None
        audio = self._decode_frames_sync(state.frames, 0, len(state.frames))
        return torch.from_numpy(np.ascontiguousarray(audio))

    # ------------------------------------------------------------------
    # Non-streaming (batch) path
    # ------------------------------------------------------------------

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        original_state = load_state(payload)
        if original_state.output_codes is None:
            data = audio_waveform_payload(
                torch.empty(0),
                sample_rate=self._sample_rate,
                modality="audio",
                source_hint="Audio8-TTS vocoder",
            )
        else:
            codes = original_state.output_codes
            # output_codes: [num_codebooks, T] — stack into [1, num_codebooks, T]
            codes_t = codes.to(device=self._device, dtype=torch.long).unsqueeze(0)
            with torch.inference_mode():
                audio_tensor = self._codec.decode(codes_t)[0, 0]
            audio_np = audio_tensor.detach().float().cpu().numpy()
            data = audio_waveform_payload(
                torch.from_numpy(audio_np),
                sample_rate=self._sample_rate,
                modality="audio",
                source_hint="Audio8-TTS vocoder",
            )
        prompt_tokens = (
            len(original_state.input_ids) if original_state.input_ids is not None else 0
        )
        data["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": original_state.completion_tokens or 0,
            "total_tokens": prompt_tokens + (original_state.completion_tokens or 0),
        }
        payload.data = data
        return payload

    # ------------------------------------------------------------------
    # Codec helper
    # ------------------------------------------------------------------

    def _decode_frames_sync(
        self,
        frames: list[torch.Tensor],
        start: int,
        end: int,
    ) -> np.ndarray:
        torch.cuda.set_device(self._device)
        codes = torch.stack(frames[start:end], dim=1).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            audio = self._codec.decode(codes)[0, 0]
        return audio.detach().float().cpu().numpy().copy()
