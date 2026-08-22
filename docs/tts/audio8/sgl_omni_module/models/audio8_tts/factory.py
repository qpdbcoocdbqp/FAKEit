# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import Any

import torch

from sglang_omni.models.audio8_tts.attention_backend import (
    ATTENTION_BACKEND_ENV,
    PORTABLE_ATTENTION_BACKEND,
    fa3_kernels_available,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class Audio8TTSEngineBuilder(TtsEngineBuilder):
    model_name = "Audio8-TTS"
    context_length = 4096
    model_arch_override = "ArkttsModel"

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "max_running_requests": int(
                os.getenv("AUDIO8_TTS_MAX_RUNNING_REQUESTS", "32")
            ),
            "dtype": dtype,
            "disable_cuda_graph": os.getenv("AUDIO8_TTS_DISABLE_CUDA_GRAPH", "0")
            == "1",
            "disable_overlap_schedule": True,
            "enable_torch_compile": os.getenv(
                "AUDIO8_TTS_ENABLE_TORCH_COMPILE", "0"
            )
            == "1",
            "chunked_prefill_size": int(
                os.getenv("AUDIO8_TTS_CHUNKED_PREFILL_SIZE", "2048")
            ),
            "mem_fraction_static": float(
                os.getenv("AUDIO8_TTS_MEM_FRACTION_STATIC", "0.2")
            ),
            "disable_radix_cache": os.getenv("AUDIO8_TTS_DISABLE_RADIX_CACHE", "1")
            == "1",
            "trust_remote_code": True,
            "sampling_backend": "pytorch",
        }

    def customize_server_args(self, server_args: Any) -> None:
        # Apply attention backend override from env / capability detection
        attention_backend = os.getenv(ATTENTION_BACKEND_ENV)
        if attention_backend is None and not fa3_kernels_available():
            attention_backend = PORTABLE_ATTENTION_BACKEND
        if attention_backend:
            server_args.attention_backend = attention_backend

        # Apply torch compile batch size if set
        torch_compile_max_bs = os.getenv("AUDIO8_TTS_TORCH_COMPILE_MAX_BS")
        if torch_compile_max_bs is not None:
            server_args.torch_compile_max_bs = int(torch_compile_max_bs)

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id
        model_worker.model_runner.model.setup_audio8_decode(
            server_args.max_running_requests
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.audio8_tts.runtime.audio8_sglang_ar import (
            Audio8ModelRunner,
        )

        return Audio8ModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.audio8_tts.pipeline.engine_io import (
            apply_tts_result,
            build_tts_request,
        )
        from sglang_omni.models.audio8_tts.pipeline.state_io import load_state, store_state
        from sglang_omni.models.audio8_tts.runtime.audio8_sglang_ar import (
            Audio8SGLangRequestData,
            Audio8StepOutput,
        )
        from sglang_omni.proto import StagePayload
        from sglang_omni.scheduling.messages import OutgoingMessage
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.checkpoint_dir,
            fix_mistral_regex=False,
        )
        vocab_size = int(model.config.vocab_size)

        def request_builder(payload: StagePayload) -> Audio8SGLangRequestData:
            req_data = build_tts_request(
                load_state(payload),
                tokenizer,
                payload.request_id,
                vocab_size=vocab_size,
            )
            req_data.stage_payload = payload
            return req_data

        def result_adapter(data: Audio8SGLangRequestData) -> StagePayload:
            original_payload = data.stage_payload
            state = load_state(original_payload)
            apply_tts_result(state, data)
            return store_state(original_payload, state)

        def stream_output_builder(
            request_id: str,
            data: Audio8SGLangRequestData,
            req_output: Any,
        ) -> list[OutgoingMessage]:
            step_output = req_output.data
            if not isinstance(step_output, Audio8StepOutput):
                return []
            params = data.stage_payload.request.params
            if not isinstance(params, dict) or not params.get("stream"):
                return []
            codes = step_output.codes.detach().to(dtype=torch.long)
            # codes: [num_codebooks+1, 1] — semantic row 0 + VQ rows 1..N
            return [
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=codes,
                    target="vocoder",
                    metadata={"modality": "audio_codes", "stream": True},
                )
            ]

        self._stream_output_builder = stream_output_builder
        return request_builder, result_adapter

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        builder = getattr(self, "_stream_output_builder", None)
        if builder is None:
            return {}
        return {"stream_output_builder": builder}


def create_audio8_engine(model_path: str, *, gpu_id: int, **kwargs: Any) -> Any:
    """Compatibility entry point — build an Audio8-TTS engine stage."""
    return Audio8TTSEngineBuilder().build(model_path, gpu_id=gpu_id)


def make_server_args(model_path: str) -> Any:
    """Return a ServerArgs instance for Audio8-TTS (used by verify_install)."""
    from sglang.srt.server_args import ServerArgs
    from sglang_omni.models.audio8_tts.attention_backend import (
        ATTENTION_BACKEND_ENV,
        PORTABLE_ATTENTION_BACKEND,
        fa3_kernels_available,
    )

    server_args = ServerArgs(
        model_path=model_path,
        tp_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        mem_fraction_static=float(os.getenv("AUDIO8_TTS_MEM_FRACTION_STATIC", "0.2")),
        chunked_prefill_size=int(os.getenv("AUDIO8_TTS_CHUNKED_PREFILL_SIZE", "2048")),
        max_running_requests=int(os.getenv("AUDIO8_TTS_MAX_RUNNING_REQUESTS", "32")),
        disable_radix_cache=os.getenv("AUDIO8_TTS_DISABLE_RADIX_CACHE", "1") == "1",
    )
    setattr(
        server_args,
        "disable_" + "cuda_graph",
        os.getenv("AUDIO8_TTS_DISABLE_CUDA_GRAPH", "0") == "1",
    )
    server_args.enable_torch_compile = (
        os.getenv("AUDIO8_TTS_ENABLE_TORCH_COMPILE", "0") == "1"
    )
    torch_compile_max_bs = os.getenv("AUDIO8_TTS_TORCH_COMPILE_MAX_BS")
    if torch_compile_max_bs is not None:
        server_args.torch_compile_max_bs = int(torch_compile_max_bs)
    attention_backend = os.getenv(ATTENTION_BACKEND_ENV)
    if attention_backend is None and not fa3_kernels_available():
        attention_backend = PORTABLE_ATTENTION_BACKEND
    if attention_backend:
        server_args.attention_backend = attention_backend
    return server_args
