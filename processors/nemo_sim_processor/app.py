# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NeMo-Sim's episode interaction protocol through Gym Agents."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import Body, Request
from pydantic import Field

from nemo_gym.config_types import AgentServerRef, ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.processors.multi_agent import (
    BaseMultiTurnProcessor,
    BaseMultiTurnProcessorConfig,
    ParticipantTurn,
)
from processors.nemo_sim_processor.contracts import (
    EPISODE_INTERACTION_PROTOCOL,
    NEMO_SIM_MODEL_ALIASES,
    NeMoSimProcessorResponse,
    NeMoSimProtocolConfig,
    NeMoSimRunRequest,
    NeMoSimScenario,
    NeMoSimSeedSessionRequest,
    NeMoSimSeedSessionResponse,
    NeMoSimSimulationResult,
    NeMoSimVerifyRequest,
    ResolvedNeMoSimContext,
)


@dataclass(frozen=True)
class _ResolvedNeMoSimEpisode:
    """Internal episode state created from task input and one seed response."""

    task: NeMoSimRunRequest
    scenario: NeMoSimScenario
    context: ResolvedNeMoSimContext


class NeMoSimProcessorConfig(BaseMultiTurnProcessorConfig):
    """Configure participant Agents separately from support Model Servers."""

    user_agent: AgentServerRef
    assistant_agent: AgentServerRef
    judge_model: ModelServerRef
    summary_model: ModelServerRef
    api_response_model: ModelServerRef
    max_turns: int = Field(5, ge=1)
    agent_call_timeout_s: float = Field(300.0, gt=0)
    protocol_config: NeMoSimProtocolConfig = Field(default_factory=NeMoSimProtocolConfig)
    skip_verification: Literal[False] = False

    def target_for_alias(self, alias: str) -> AgentServerRef | ModelServerRef:
        return {
            "user_model": self.user_agent,
            "assistant_model": self.assistant_agent,
            "judge_model": self.judge_model,
            "summary_model": self.summary_model,
            "api_response_model": self.api_response_model,
        }[alias]


class _GymModelFacade:
    """Synchronous facade expected by NeMo-Sim's Data Designer integration."""

    def __init__(self, alias: str, bridge: "_ConversationBridge") -> None:
        self.alias = alias
        self.model_name = bridge.processor.config.target_for_alias(alias).name
        self._bridge = bridge

    def completion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        if kwargs.get("tools"):
            raise NotImplementedError(
                "NeMoSimProcessor currently supports non-tool probes only: tool execution ownership between "
                "ConversationLoop and Gym Agents remains an open design question."
            )
        unsupported = set(kwargs) - {"max_tokens", "tools"}
        if unsupported:
            raise NotImplementedError(f"Unsupported NeMo-Sim completion options: {sorted(unsupported)}")
        return self._bridge.complete_from_worker(self.alias, messages, max_tokens=kwargs.get("max_tokens"))


class _GeneratorHarness:
    """Duck-typed host for NeMo-Sim's existing Data Designer row adapter."""

    def __init__(self, config: Any, models: Mapping[str, _GymModelFacade]) -> None:
        self.config = config
        self._models = models

    def get_model(self, alias: str) -> _GymModelFacade:
        return self._models[alias]


class _ConversationBridge:
    """Bridge blocking NeMo-Sim calls to async Gym Agent turns and support-model calls."""

    def __init__(
        self,
        *,
        processor: "NeMoSimProcessor",
        body: NeMoSimRunRequest,
        event_loop: asyncio.AbstractEventLoop,
        cookies: Mapping[str, Any],
    ) -> None:
        self.processor = processor
        self.body = body
        self.event_loop = event_loop
        self.cookies_by_alias = {alias: dict(cookies) for alias in NEMO_SIM_MODEL_ALIASES}
        self.turns: list[ParticipantTurn] = []
        self.responses_by_alias: dict[str, list[NeMoGymResponse]] = {alias: [] for alias in NEMO_SIM_MODEL_ALIASES}

    def complete_from_worker(
        self,
        alias: str,
        messages: Sequence[Any],
        *,
        max_tokens: int | None,
    ) -> SimpleNamespace:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self.event_loop:
            raise RuntimeError("NeMo-Sim's synchronous ConversationLoop must run outside the Processor event loop")

        future = asyncio.run_coroutine_threadsafe(
            self._invoke_agent(alias, messages, max_tokens=max_tokens),
            self.event_loop,
        )
        try:
            return future.result(timeout=self.processor.config.agent_call_timeout_s)
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError(
                f"Timed out after {self.processor.config.agent_call_timeout_s}s waiting for {alias}"
            ) from error

    async def _invoke_agent(
        self,
        alias: str,
        messages: Sequence[Any],
        *,
        max_tokens: int | None,
    ) -> SimpleNamespace:
        params = self.body.model_responses_create_params.get(alias, self.body.responses_create_params)
        input_messages = [_to_responses_input(message) for message in messages]
        request_values = params.model_dump(mode="json", exclude_none=True)
        request_values.update({"input": input_messages, "instructions": None, "tools": []})
        if max_tokens is not None:
            request_values["max_output_tokens"] = max_tokens
        request_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(request_values)

        target = self.processor.config.target_for_alias(alias)
        result = await self.processor._call_responses_actor(
            target=target,
            params=request_params,
            body=self.body,
            cookies=self.cookies_by_alias[alias],
        )
        gym_response = result.response
        self.cookies_by_alias[alias].update(result.response_cookies)
        self.responses_by_alias[alias].append(gym_response)
        if alias in {"user_model", "assistant_model"}:
            self.turns.append(
                ParticipantTurn(
                    turn_index=len(self.turns),
                    participant="user" if alias == "user_model" else "assistant",
                    request=request_params,
                    response=gym_response,
                    agent_trajectory=result.agent_trajectory,
                )
            )

        tool_calls = [
            {
                "id": item.call_id,
                "type": "function",
                "function": {"name": item.name, "arguments": item.arguments},
            }
            for item in gym_response.output
            if isinstance(item, NeMoGymResponseFunctionToolCall)
        ]
        usage = gym_response.usage
        return SimpleNamespace(
            message=SimpleNamespace(
                content=_response_text(gym_response),
                reasoning_content=None,
                tool_calls=tool_calls or None,
            ),
            usage=(
                SimpleNamespace(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
                if usage is not None
                else None
            ),
        )


def _to_responses_input(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        value = message.model_dump(mode="json", exclude_none=True)
    elif isinstance(message, Mapping):
        value = dict(message)
    else:
        value = {
            "role": getattr(message, "role"),
            "content": getattr(message, "content", ""),
        }

    role = value.get("role")
    if hasattr(role, "value"):
        role = role.value
    if role not in {"system", "developer", "user", "assistant"}:
        raise NotImplementedError(f"NeMo-Sim message role {role!r} is not supported by NeMoSimProcessor")
    return {"type": "message", "role": role, "content": value.get("content", "")}


def _response_text(response: NeMoGymResponse) -> str:
    chunks: list[str] = []
    for item in response.output:
        if not isinstance(item, NeMoGymResponseOutputMessage):
            continue
        for content in item.content:
            text = getattr(content, "text", None)
            refusal = getattr(content, "refusal", None)
            if text:
                chunks.append(text)
            elif refusal:
                chunks.append(refusal)
    return "\n".join(chunks)


class NeMoSimProcessor(BaseMultiTurnProcessor):
    """Let NeMo-Sim orchestrate participant Agents and support Model Servers."""

    config: NeMoSimProcessorConfig

    def _run_nemo_sim(self, bridge: _ConversationBridge, scenario: NeMoSimScenario) -> dict[str, Any]:
        from conversation_plugin.config import ConversationSimulatorConfig
        from conversation_plugin.core.llm import set_debug_log_path
        from conversation_plugin.generator import ConversationSimulatorGenerator

        set_debug_log_path(None)
        config_values = self.config.protocol_config.model_dump(mode="python", exclude_none=True)
        config_values.update(
            {
                "name": "conversation_messages",
                "locale": scenario.locale,
                "max_turns": self.config.max_turns,
            }
        )
        simulation_config = ConversationSimulatorConfig.model_validate(config_values)
        models = {alias: _GymModelFacade(alias, bridge) for alias in NEMO_SIM_MODEL_ALIASES}
        generator = _GeneratorHarness(simulation_config, models)
        scenario_data = scenario.model_dump(mode="python", exclude={"locale"})
        return ConversationSimulatorGenerator.generate(generator, scenario_data)

    async def run(
        self,
        request: Request,
        body: NeMoSimRunRequest = Body(),
    ) -> NeMoSimProcessorResponse:
        seed_request = NeMoSimSeedSessionRequest(nemo_sim_sampling=body.nemo_sim_sampling)
        seed_json, environment_cookies = await self._seed_episode(
            payload=seed_request,
            cookies=dict(request.cookies),
        )
        seed_result = NeMoSimSeedSessionResponse.model_validate(seed_json)
        episode = _ResolvedNeMoSimEpisode(
            task=body,
            scenario=seed_result.scenario,
            context=seed_result.nemo_sim_context,
        )
        bridge = _ConversationBridge(
            processor=self,
            body=episode.task,
            event_loop=asyncio.get_running_loop(),
            cookies=environment_cookies,
        )
        nemo_sim_result = NeMoSimSimulationResult.model_validate(
            await asyncio.to_thread(self._run_nemo_sim, bridge, episode.scenario)
        )
        assistant_responses = bridge.responses_by_alias["assistant_model"]
        focal_response = assistant_responses[-1] if assistant_responses else _empty_assistant_response(self.config)

        verify_request = NeMoSimVerifyRequest(
            responses_create_params=episode.task.responses_create_params,
            response=focal_response,
            nemo_sim_sampling=episode.task.nemo_sim_sampling,
            scenario=episode.scenario,
            nemo_sim_context=episode.context,
            nemo_sim_result=nemo_sim_result,
            turns=bridge.turns,
            episode_interaction_protocol=EPISODE_INTERACTION_PROTOCOL,
        )
        result = await self._verify_episode(
            payload=verify_request,
            cookies=environment_cookies,
        )
        return NeMoSimProcessorResponse.model_validate(result)


def _empty_assistant_response(config: NeMoSimProcessorConfig) -> NeMoGymResponse:
    """Preserve a structured NeMo-Sim failure that occurs before an assistant turn."""

    return NeMoGymResponse.model_validate(
        {
            "id": "",
            "created_at": 0,
            "model": config.assistant_agent.name,
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


if __name__ == "__main__":
    NeMoSimProcessor.run_webserver()
