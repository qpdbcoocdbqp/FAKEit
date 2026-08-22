# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.audio8_tts.pipeline.stages"


class Audio8TTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "ArkttsModel"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.create_preprocessing_executor",
            factory_args={"device": "cuda:0"},
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.create_sglang_tts_engine_executor",
            factory_args={"device": "cuda:0", "max_new_tokens": 1024},
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.create_vocoder_executor",
            factory_args={"device": "cuda:0"},
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


EntryClass = Audio8TTSPipelineConfig
