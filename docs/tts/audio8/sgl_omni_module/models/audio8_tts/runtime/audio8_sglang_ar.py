# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput, SchedulerRequest

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang_omni.model_runner.model_worker import ModelWorker


@dataclass
class Audio8StepOutput:
    codes: torch.Tensor


@dataclass
class Audio8SGLangRequestData(SGLangARRequestData):
    vq_mask_tokens: torch.Tensor | None = None
    vq_parts: list[torch.Tensor] | None = None
    num_codebooks: int = 10
    codebook_size: int = 4096
    output_codes: list[torch.Tensor] = field(default_factory=list)
    max_new_tokens: int | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    _previous_semantic_tokens: list[int] = field(default_factory=list)
    _last_codebook_values: torch.Tensor | None = None


class Audio8ModelRunner(ModelRunner):
    """Model runner for Audio8-TTS.

    Drives the per-step VQ sampling buffers through the ModelRunner hook API
    and injects reference VQ embeddings during prefill.
    """

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)

    # ------------------------------------------------------------------
    # Prefill hook: inject reference VQ embeddings into input_embeds
    # ------------------------------------------------------------------

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        device = forward_batch.input_ids.device
        model = self.tp_worker.model_runner.model
        # These buffers are persistent across requests.  The legacy runner
        # reset them for every prefill; retain that isolation under the hook
        # based runner as well.
        batch = len(requests)
        model._vq_mask[:batch].zero_()
        model._previous_valid[:batch].zero_()
        text_embeds = model.get_embed_tokens()(forward_batch.input_ids)
        offset = 0
        for sched_req in requests:
            data: Audio8SGLangRequestData = sched_req.data
            request_length = int(data.req.extend_range.length)
            prefix_length = len(data.req.prefix_indices)
            if data.vq_mask_tokens is not None and data.vq_parts:
                mask = data.vq_mask_tokens.to(device=device, dtype=torch.bool).flatten()
                expected_length = prefix_length + request_length
                if mask.numel() != expected_length:
                    raise ValueError(
                        f"Audio8 reference mask length {mask.numel()} != "
                        f"request length {expected_length}"
                    )
                mask_slice = mask[prefix_length : prefix_length + request_length]
                parts = [
                    part.to(device=device, dtype=torch.long).T for part in data.vq_parts
                ]
                all_codes = torch.cat(parts, dim=0)
                if all_codes.shape[0] != int(mask.sum().item()):
                    raise ValueError("Audio8 reference mask/code length mismatch")
                before = int(mask[:prefix_length].sum().item())
                count = int(mask_slice.sum().item())
                if count:
                    codes = all_codes[before : before + count]
                    if codes.shape[1] != model.config.num_codebooks:
                        raise ValueError(
                            "Audio8 reference has the wrong number of codebooks"
                        )
                    if (
                        int(codes.min().item()) < 0
                        or int(codes.max().item()) >= model.config.codebook_size
                    ):
                        raise ValueError(
                            "Audio8 reference code is outside the embedding range"
                        )
                    embedded_codes = codes + model._codebook_offsets[None]
                    codebook_sum = model.codebook_embeddings(embedded_codes).sum(dim=1)
                    indices = mask_slice.nonzero(as_tuple=True)[0] + offset
                    text_embeds[indices] += codebook_sum.to(text_embeds.dtype)
            offset += request_length
        forward_batch.input_embeds = text_embeds

    # ------------------------------------------------------------------
    # Decode hook: update per-request sampling buffers before forward
    # ------------------------------------------------------------------

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead, schedule_batch
        model = self.tp_worker.model_runner.model
        batch = len(requests)
        model._previous_valid[:batch].zero_()
        input_ids = forward_batch.input_ids
        semantic_mask = (input_ids >= model.config.semantic_begin_id) & (
            input_ids <= model.config.semantic_end_id
        )
        model._vq_mask[:batch].copy_(semantic_mask)

        for index, sched_req in enumerate(requests):
            data: Audio8SGLangRequestData = sched_req.data
            model._temperature[index] = max(float(data.temperature), 1e-5)
            model._top_p[index] = min(max(float(data.top_p), 1e-5), 1.0)
            model._top_k[index] = min(
                max(int(data.top_k), 1),
                int(model.config.codebook_size),
            )
            model._do_sample[index] = bool(data.do_sample)
            history = data._previous_semantic_tokens[-model.config.ras_window_size :]
            if history:
                length = len(history)
                model._previous_semantic[index, -length:] = torch.tensor(
                    history,
                    device=model._previous_semantic.device,
                    dtype=torch.long,
                )
                model._previous_valid[index, -length:] = True
            if (
                data._last_codebook_values is not None
                and bool(model._vq_mask[index].item())
            ):
                model._vq_codes[index].copy_(data._last_codebook_values)

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        """Return the semantic token sampled by Audio8's model head.

        ``Audio8SGLangModel.forward`` samples the semantic token together with
        its VQ codebooks and stores it in ``_output_semantic_ids``.  Sampling
        ``logits_output`` again through the generic SGLang sampler breaks that
        coupling: the next AR input can describe a different frame than the
        VQ codes sent to the vocoder.
        """
        del logits_output, schedule_batch
        model = self.tp_worker.model_runner.model
        # The model buffers are allocated for max_running_requests, while the
        # execution bridge expects exactly one token per scheduled request.
        return model._output_semantic_ids[: len(requests)].clone()

    def finalize_skip_rids(self, scheduler_output: Any) -> set[str]:
        """Do not count the synthetic prefill step as generated audio."""
        batch = getattr(scheduler_output, "batch_data", None)
        mode = getattr(batch, "forward_mode", None)
        if bool(getattr(batch, "is_prefill_only", False)) or bool(
            mode is not None and mode.is_extend()
        ):
            return {req.request_id for req in scheduler_output.requests}
        return set()

    # ------------------------------------------------------------------
    # post_process_outputs: collect codes, drive per-request state
    # ------------------------------------------------------------------

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        model = self.tp_worker.model_runner.model

        batch = len(scheduler_output.requests)
        scheduler_output.batch_data.output_ids = model._output_semantic_ids[
            :batch
        ].clone()

        # The prefill-only pass runs the model once to populate the KV cache;
        # its Audio8 output buffer is not a generated waveform frame.
        batch_data = getattr(scheduler_output, "batch_data", None)
        mode = getattr(batch_data, "forward_mode", None)
        if bool(mode is not None and mode.is_extend()):
            for sched_req in scheduler_output.requests:
                req = sched_req.data.req
                logger.info(
                    "Audio8 prefill rid=%s max_new_tokens=%s output_ids=%s "
                    "finished_reason=%r finished=%s stop_strs=%r "
                    "stop_regex_strs=%r eos_token_ids=%r tokenizer=%r",
                    sched_req.request_id,
                    getattr(req.sampling_params, "max_new_tokens", None),
                    getattr(req, "output_ids", None),
                    getattr(req, "finished_reason", None),
                    req.finished(),
                    getattr(req, "stop_strs", None),
                    getattr(req, "stop_regex_strs", None),
                    getattr(req, "eos_token_ids", None),
                    getattr(req, "tokenizer", None),
                )
            return

        for index, sched_req in enumerate(scheduler_output.requests):
            data: Audio8SGLangRequestData = sched_req.data
            req = data.req
            req_output = outputs[sched_req.request_id]

            # Chunked-prefill rows: no codebook output yet
            if getattr(req, "is_chunked", 0) > 0:
                req_output.data = None
                req.is_chunked -= 1
                continue

            codes = model._output_codes[index].unsqueeze(-1).clone()
            semantic = int(codes[0, -1].item())

            logger.info(
                "Audio8 decode rid=%s semantic=%s output_frames=%s max_new_tokens=%s",
                sched_req.request_id,
                semantic,
                len(data.output_codes),
                data.max_new_tokens,
            )

            req_output.data = Audio8StepOutput(codes)

            # EOS is a control token and has no corresponding waveform frame.
            # Do not feed its placeholder codebook values to the codec.
            if semantic == model.config.eos_token_id:
                req.finished_reason = FINISH_MATCHED_TOKEN(semantic)
                continue

            # Accumulate codes and update rolling state.
            if data.output_codes:
                # Only push semantic to history after the first decode step
                data._previous_semantic_tokens.append(semantic)
            data.output_codes.append(codes)
            data._last_codebook_values = codes[1:, 0].clone()

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        req = req_data.req
        finished_reason = getattr(req, "finished_reason", None)
        finish_json = (
            finished_reason.to_json()
            if finished_reason is not None and hasattr(finished_reason, "to_json")
            else finished_reason
        )
        logger.info(
            "Audio8 terminal rid=%s output_frames=%s output_ids=%s "
            "finish_reason=%r finish_json=%r max_new_tokens=%s stop_strs=%r "
            "stop_regex_strs=%r eos_token_ids=%r",
            request_id,
            len(req_data.output_codes),
            getattr(req, "output_ids", None),
            finished_reason,
            finish_json,
            req_data.max_new_tokens,
            getattr(req, "stop_strs", None),
            getattr(req, "stop_regex_strs", None),
            getattr(req, "eos_token_ids", None),
        )
