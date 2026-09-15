# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.processors.multi_agent import BaseMultiTurnProcessor
from nemo_gym.server_utils import ServerClient
from processors.nemo_sim_processor.app import (
    NeMoSimProcessor,
    NeMoSimProcessorConfig,
    _ConversationBridge,
    _GymModelFacade,
)
from processors.nemo_sim_processor.contracts import NeMoSimProtocolConfig, NeMoSimRunRequest


def _model_response(response_id: str, text: str) -> dict:
    return {
        "id": response_id,
        "created_at": 1,
        "model": "model",
        "object": "response",
        "output": [
            {
                "id": f"{response_id}-message",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _processor(*, protocol_config: dict | None = None) -> NeMoSimProcessor:
    config = NeMoSimProcessorConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="nemo-sim-processor",
        user_agent=AgentServerRef(type="responses_api_agents", name="user-agent"),
        assistant_agent=AgentServerRef(type="responses_api_agents", name="assistant-agent"),
        judge_model=ModelServerRef(type="responses_api_models", name="judge-model"),
        summary_model=ModelServerRef(type="responses_api_models", name="summary-model"),
        api_response_model=ModelServerRef(type="responses_api_models", name="api-response-model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="nemo-sim-resources"),
        max_turns=2,
        protocol_config=protocol_config or {},
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    return NeMoSimProcessor(config=config, server_client=client)


def _request() -> NeMoSimRunRequest:
    return NeMoSimRunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "nemo_sim_sampling": {"locale": "en_US", "seed": 1042},
            "processor_ref": {"type": "processors", "name": "nemo_sim_processor"},
            "_ng_task_index": 3,
            "_ng_rollout_index": 1,
        }
    )


def _scenario() -> dict:
    return {
        "persona": {"first_name": "Morgan", "age": 42},
        "probe_type": "general_open_ended",
        "theme": {"type": "recommendation", "description": "Plan dinner."},
        "goal": "Plan dinner.",
        "locale": "en_US",
    }


def _context() -> dict:
    return {
        "locale": "en_US",
        "seed": 1042,
        "personas_dataset_version": "0.0.2",
        "personas_source_sha256": "a" * 64,
        "personas_panel_seed": 42,
    }


def test_inherits_protocol_agnostic_multi_turn_base() -> None:
    assert isinstance(_processor(), BaseMultiTurnProcessor)


@pytest.mark.asyncio
async def test_conversation_loop_calls_participant_agents_and_support_models(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor()
    posts: list[dict] = []
    model_call_count = 0

    async def post(**kwargs):
        nonlocal model_call_count
        posts.append(kwargs)
        if kwargs["url_path"] == "/seed_session":
            payload = {"scenario": _scenario(), "nemo_sim_context": _context()}
            cookies = {"session": "environment"}
        elif kwargs["url_path"] == "/verify":
            payload = kwargs["json"] | {"reward": 1.0, "scenario_completed": True}
            cookies = {"session": "environment"}
        else:
            model_call_count += 1
            payload = _model_response(f"response-{model_call_count}", f"output-{model_call_count}")
            cookies = {}
        response = MagicMock(status=200, ok=True, cookies=cookies)
        response.content.read = AsyncMock(return_value=b"")
        response.read = AsyncMock(return_value=json.dumps(payload))
        return response

    processor.server_client.post = post

    def run_interaction_protocol(bridge, body):
        del body
        models = {
            alias: _GymModelFacade(alias, bridge)
            for alias in ("user_model", "assistant_model", "judge_model", "summary_model")
        }
        models["user_model"].completion(
            [{"role": "system", "content": "Act as the user."}],
            max_tokens=77,
        )
        models["judge_model"].completion([{"role": "user", "content": "Judge the user turn."}])
        models["assistant_model"].completion([{"role": "user", "content": "Help me."}])
        models["summary_model"].completion([{"role": "user", "content": "Should the episode stop?"}])
        return {
            "conversation_status": True,
            "conversation_messages": "[]",
            "simulation_outcome": "{}",
        }

    monkeypatch.setattr(processor, "_run_nemo_sim", run_interaction_protocol)
    result = await processor.run(SimpleNamespace(cookies={"session": "shared"}), _request())

    assert [post["server_name"] for post in posts] == [
        "nemo-sim-resources",
        "user-agent",
        "judge-model",
        "assistant-agent",
        "summary-model",
        "nemo-sim-resources",
    ]
    assert posts[1]["json"].max_output_tokens == 77
    assert posts[0]["json"] == {"nemo_sim_sampling": {"locale": "en_US", "seed": 1042, "probe_type": None}}
    assert posts[-1]["json"]["scenario"]["persona"]["first_name"] == "Morgan"
    assert posts[-1]["json"]["nemo_sim_context"] == _context()
    assert set(posts[-1]["json"]) == {
        "responses_create_params",
        "response",
        "nemo_sim_sampling",
        "scenario",
        "nemo_sim_context",
        "nemo_sim_result",
        "invocations",
        "episode_interaction_protocol",
    }
    assert all(post["cookies"]["session"] == "environment" for post in posts[1:-1])
    assert [(call.alias, call.executor) for call in result.invocations] == [
        ("user_model", "agent"),
        ("judge_model", "model"),
        ("assistant_model", "agent"),
        ("summary_model", "model"),
    ]
    assert result.response.output[0].content[0].text == "output-3"
    assert result.reward == 1.0
    assert result.nemo_sim_context.seed == 1042
    assert result.scenario_completed is True
    assert result.episode_interaction_protocol == "nemo_sim.ConversationLoop"
    assert isinstance(result, BaseVerifyResponse)
    assert "processor_ref" not in result.model_dump(mode="json")
    assert "_ng_task_index" not in result.model_dump(mode="json")


def test_rejects_unknown_response_parameter_alias() -> None:
    with pytest.raises(ValueError, match="Input should be"):
        NeMoSimRunRequest.model_validate(
            {
                **_request().model_dump(mode="json"),
                "model_responses_create_params": {"not_a_nemo_sim_alias": {"input": []}},
            }
        )


def test_protocol_config_is_typed_processor_configuration_not_task_input() -> None:
    processor = _processor(protocol_config={"max_query_attempts": 4, "context_compression": False})

    assert processor.config.protocol_config == NeMoSimProtocolConfig(
        max_query_attempts=4,
        context_compression=False,
    )
    with pytest.raises(ValueError, match="owned by another lifecycle"):
        NeMoSimRunRequest.model_validate(
            {
                **_request().model_dump(mode="json"),
                "simulation_config": {"max_steps": 7},
            }
        )


@pytest.mark.parametrize("field", ["scenario", "nemo_sim_context", "nemo_sim_result"])
def test_dataset_row_rejects_output_only_episode_fields(field: str) -> None:
    with pytest.raises(ValueError, match="owned by another lifecycle"):
        NeMoSimRunRequest.model_validate(
            {
                **_request().model_dump(mode="json"),
                field: {},
            }
        )


@pytest.mark.parametrize("field", ["name", "locale", "max_turns", "assets_dir", "finance_tier"])
def test_protocol_config_hides_data_designer_and_unsupported_probe_fields(field: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _processor(protocol_config={field: "override"})


@pytest.mark.asyncio
async def test_preserves_failure_before_first_assistant_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor()

    async def post(**kwargs):
        payload = (
            {"scenario": _scenario(), "nemo_sim_context": _context()}
            if kwargs["url_path"] == "/seed_session"
            else kwargs["json"] | {"reward": 0.0, "scenario_completed": False}
        )
        response = MagicMock(status=200, ok=True, cookies={"session": "environment"})
        response.content.read = AsyncMock(return_value=b"")
        response.read = AsyncMock(return_value=json.dumps(payload))
        return response

    processor.server_client.post = post

    def fail_user_gate(bridge, body):
        del bridge, body
        return {
            "conversation_status": False,
            "conversation_messages": "[]",
            "simulation_outcome": '{"status":"failed"}',
        }

    monkeypatch.setattr(processor, "_run_nemo_sim", fail_user_gate)
    result = await processor.run(SimpleNamespace(cookies={}), _request())

    assert result.nemo_sim_result.conversation_status is False
    assert result.nemo_sim_result.conversation_messages == []
    assert result.nemo_sim_result.simulation_outcome == {"status": "failed"}
    assert result.response.output == []
    assert result.invocations == []


@pytest.mark.asyncio
async def test_rejects_sync_bridge_call_on_processor_event_loop() -> None:
    processor = _processor()
    bridge = _ConversationBridge(
        processor=processor,
        body=_request(),
        event_loop=asyncio.get_running_loop(),
        cookies={},
    )

    with pytest.raises(RuntimeError, match="must run outside"):
        bridge.complete_from_worker("user_model", [], max_tokens=None)
