# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.audio8_tts.io import Audio8TTSState
from sglang_omni.models.audio8_tts.runtime.audio8_sglang_ar import (
    Audio8SGLangRequestData,
)


def build_tts_request(
    state: Audio8TTSState,
    tokenizer: Any,
    request_id: str,
    *,
    vocab_size: int | None = None,
) -> Audio8SGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    input_ids = list(state.input_ids)
    # Audio8 semantic IDs live above the text tokenizer's vocabulary.  SGLang
    # treats an output ID beyond Req.vocab_size as a NaN sentinel, so use the
    # model's full vocabulary when the engine provides it.
    request_vocab_size = int(vocab_size or tokenizer.vocab_size)
    sampling_params = SamplingParams(
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        # Audio8 emits EOS from its semantic head.  The legacy iteration
        # controller handled that token after a real decode step; allowing the
        # generic scheduler to stop on the prefill logits terminates TTS before
        # any audio frame is generated.
        stop_token_ids=[],
    )
    # tokenizer_manager.normalize() is bypassed in our custom pipeline;
    # without it stop_strs / stop_regex_strs stay None and the upstream
    # scheduler's update_finish_state trips on ``len(None)``.
    sampling_params.normalize(tokenizer=None)
    # Audio8 completion is controlled by its semantic EOS token.  Do not let
    # generic text stop matching terminate a speech request after one frame.
    sampling_params.stop_strs = []
    sampling_params.stop_regex_strs = []
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
        eos_token_ids=set(),
        vocab_size=request_vocab_size,
    )
    req.tokenizer = None
    # Req snapshots these fields in some SGLang versions.  Clear both the
    # request copy and SamplingParams copy so text stopping cannot terminate
    # Audio8 after its first semantic frame.
    req.stop_strs = []
    req.stop_regex_strs = []
    req.stop_str_max_len = 0
    req.eos_token_ids = set()
    req.sampling_params.stop_strs = []
    req.sampling_params.stop_regex_strs = []
    return Audio8SGLangRequestData(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        req=req,
        vq_mask_tokens=(
            torch.as_tensor(state.vq_mask_tokens, dtype=torch.bool)
            if state.vq_mask_tokens is not None
            else None
        ),
        vq_parts=(
            [torch.as_tensor(part, dtype=torch.long) for part in state.vq_parts]
            if state.vq_parts is not None
            else None
        ),
        num_codebooks=state.num_codebooks,
        codebook_size=state.codebook_size,
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        do_sample=state.do_sample,
    )


def apply_tts_result(state: Audio8TTSState, result: Audio8SGLangRequestData) -> None:
    if result.output_codes:
        all_codes = torch.cat(result.output_codes, dim=1)
        state.output_codes = all_codes[1:]
        state.completion_tokens = int(all_codes.shape[1])
    else:
        state.output_codes = None
    state.prompt_tokens = len(result.input_ids) if result.input_ids is not None else 0
