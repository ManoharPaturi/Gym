# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for NeMo-Sim multi-turn episodes.

The contracts separate five lifetimes:

* ``NeMoSimProtocolConfig`` is immutable Processor configuration for a run.
* ``NeMoSimRunRequest`` is task input supplied by one benchmark dataset row.
* ``NeMoSimSeedSessionResponse`` is Resources Server-resolved episode state.
* ``NeMoSimVerifyRequest`` is the completed episode submitted for scoring.
* ``NeMoSimProcessorResponse`` is the standardized rollout result returned to the collector.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


NeMoSimModelAlias = Literal[
    "user_model",
    "assistant_model",
    "api_response_model",
    "judge_model",
    "summary_model",
]
NEMO_SIM_MODEL_ALIASES = frozenset(
    {
        "user_model",
        "assistant_model",
        "api_response_model",
        "judge_model",
        "summary_model",
    }
)
EPISODE_INTERACTION_PROTOCOL = "nemo_sim.ConversationLoop"


class NeMoSimProtocolConfig(BaseModel):
    """Supported run-wide NeMo-Sim behavior, independent of Data Designer plumbing."""

    model_config = ConfigDict(extra="forbid")

    max_query_attempts: int = Field(3, ge=1)
    max_assistant_attempts: int = Field(1, ge=1)
    enforce_user_language: bool = True
    user_language_min_script_compliance: float = Field(0.6, ge=0.0, le=1.0)
    user_language_min_letters: int = Field(8, ge=0)
    incremental_disclosure_ratio: float = Field(0.6, ge=0.0, le=1.0)
    persona_grounding_ratio: float = Field(1.0, ge=0.0, le=1.0)
    context_compression: bool = True
    compression_window: int = Field(1, ge=1)
    store_reasoning: bool = True
    random_seed: int | None = None
    verbosity: int = Field(1, ge=0, le=2)


class NeMoSimSamplingRequest(BaseModel):
    """Per-task selectors stored in a benchmark dataset row."""

    model_config = ConfigDict(extra="forbid")

    locale: str = Field("en_US", pattern=r"^[A-Za-z0-9_]+$")
    seed: int
    probe_type: str | None = None


class NeMoSimTheme(BaseModel):
    """Resolved probe theme and user objective."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1)
    goal: str = Field(min_length=1)


class NeMoSimScenarioTheme(BaseModel):
    """Theme representation consumed by supported NeMo-Sim probes."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    description: str = Field(min_length=1)


class NeMoSimScenario(BaseModel):
    """Executable row passed to NeMo-Sim's conversation generator."""

    model_config = ConfigDict(extra="allow")

    persona: dict[str, Any]
    probe_type: str
    theme: NeMoSimScenarioTheme | str
    goal: str
    locale: str


class ResolvedNeMoSimContext(BaseModel):
    """Immutable provenance for scenario selection, without scenario duplication."""

    model_config = ConfigDict(extra="forbid")

    locale: str
    seed: int
    personas_dataset_version: str
    personas_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    personas_panel_seed: int


class NeMoSimRunRequest(BaseRunRequest):
    """External Processor input: one task row plus optional per-alias call overrides."""

    # The rollout collector adds framework routing/correlation fields to rows. Preserve those
    # without treating them as NeMo-Sim task semantics.
    model_config = ConfigDict(extra="allow")

    nemo_sim_sampling: NeMoSimSamplingRequest
    model_responses_create_params: dict[NeMoSimModelAlias, NeMoGymResponseCreateParamsNonStreaming] = Field(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def reject_non_task_fields(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        forbidden = {"nemo_sim_context", "nemo_sim_result", "scenario", "simulation_config"} & values.keys()
        if forbidden:
            raise ValueError(f"dataset rows contain NeMo-Sim fields owned by another lifecycle: {sorted(forbidden)}")
        return values


class NeMoSimSeedSessionRequest(BaseSeedSessionRequest):
    """Minimal task selectors sent from the Processor to the Resources Server."""

    model_config = ConfigDict(extra="forbid")

    nemo_sim_sampling: NeMoSimSamplingRequest


class NeMoSimSeedSessionResponse(BaseSeedSessionResponse):
    """Resolved state returned exactly once before participant invocations."""

    model_config = ConfigDict(extra="forbid")

    scenario: NeMoSimScenario
    nemo_sim_context: ResolvedNeMoSimContext


class NeMoSimInvocation(BaseModel):
    """One attributed participant or support-model invocation."""

    model_config = ConfigDict(extra="forbid")

    alias: NeMoSimModelAlias
    executor: Literal["agent", "model"]
    call_index: int = Field(ge=0)
    request: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    ng_trajectory: dict[str, Any] | None = None


class NeMoSimSimulationResult(BaseModel):
    """Structured Gym episode outputs plus probe-specific extensions.

    NeMo-Sim emits several Data Designer columns as JSON strings. The adapter
    decodes those strings here so Gym rollout consumers receive one stable
    structured contract.
    """

    model_config = ConfigDict(extra="allow")

    conversation_messages: list[dict[str, Any]]
    conversation_status: bool
    simulation_outcome: dict[str, Any]
    conversation_metadata: dict[str, Any] | None = None
    simulation_traces: list[dict[str, Any]] | None = None
    trajectory_id: str | None = None
    persona_uuid: str | None = None
    probe_family: str | None = None
    probe_variant: str | None = None
    num_turns: int | None = Field(None, ge=0)
    num_tool_calls: int | None = Field(None, ge=0)
    user_query: str | None = None

    @field_validator(
        "conversation_messages",
        "simulation_outcome",
        "conversation_metadata",
        "simulation_traces",
        mode="before",
    )
    @classmethod
    def decode_data_designer_json_columns(cls, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value


class NeMoSimVerifyRequest(BaseVerifyRequest):
    """Explicit completed episode submitted to the Resources Server."""

    model_config = ConfigDict(extra="forbid")

    nemo_sim_sampling: NeMoSimSamplingRequest
    scenario: NeMoSimScenario
    nemo_sim_context: ResolvedNeMoSimContext
    nemo_sim_result: NeMoSimSimulationResult
    invocations: list[NeMoSimInvocation]
    episode_interaction_protocol: Literal["nemo_sim.ConversationLoop"] = EPISODE_INTERACTION_PROTOCOL


class NeMoSimProcessorResponse(BaseVerifyResponse):
    """Final Processor result consumed and extended by the rollout collector."""

    model_config = ConfigDict(extra="allow")

    nemo_sim_sampling: NeMoSimSamplingRequest
    scenario: NeMoSimScenario
    nemo_sim_context: ResolvedNeMoSimContext
    nemo_sim_result: NeMoSimSimulationResult
    invocations: list[NeMoSimInvocation]
    episode_interaction_protocol: Literal["nemo_sim.ConversationLoop"] = EPISODE_INTERACTION_PROTOCOL
    scenario_completed: bool
