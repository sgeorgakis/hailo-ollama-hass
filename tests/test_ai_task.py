"""Tests for Hailo Ollama AI task entity."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hailo_ollama.const import (
    CONF_HOST,
    CONF_MODEL,
    CONF_PORT,
    CONF_SHOW_THINKING,
    CONF_STREAMING,
    CONF_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DOMAIN,
)
from custom_components.hailo_ollama.ai_task import (
    HailoAITaskEntity,
    async_setup_entry,
)
from custom_components.hailo_ollama.conversation import HailoError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry for AI task tests."""
    entry = MagicMock()
    entry.entry_id = "test_entry_id"
    entry.data = {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    }
    entry.options = {}
    return entry


def _make_entity(config_entry_data: dict) -> HailoAITaskEntity:
    """Build an entity with hass attached."""
    entry = MagicMock()
    entry.entry_id = "test_entry_id"
    entry.data = config_entry_data
    entry.options = {}
    entity = HailoAITaskEntity(entry)
    entity.hass = MagicMock()
    return entity


# ---------------------------------------------------------------------------
# async_setup_entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_setup_entry_adds_entity():
    """async_setup_entry should add exactly one HailoAITaskEntity."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    }
    entry.options = {}
    hass = MagicMock()
    async_add_entities = MagicMock()

    await async_setup_entry(hass, entry, async_add_entities)

    async_add_entities.assert_called_once()
    entities = async_add_entities.call_args[0][0]
    assert len(entities) == 1
    assert isinstance(entities[0], HailoAITaskEntity)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_entity_init(mock_config_entry):
    """Test entity initialization reads config correctly."""
    entity = HailoAITaskEntity(mock_config_entry)

    assert entity._host == "localhost"
    assert entity._port == 8000
    assert entity._model == "llama3.2:3b"
    assert entity._system_prompt == DEFAULT_SYSTEM_PROMPT
    assert entity._streaming is False
    assert entity._show_thinking is False
    assert entity._vision_model is False
    assert entity._attr_unique_id == "test_entry_id_ai_task"
    assert entity._base_url == "http://localhost:8000"


def test_entity_unique_id_includes_suffix(mock_config_entry):
    """The unique_id for AI task entity has '_ai_task' suffix to avoid collision with conversation entity."""
    entity = HailoAITaskEntity(mock_config_entry)
    assert entity._attr_unique_id == f"{mock_config_entry.entry_id}_ai_task"


def test_entity_options_override_data():
    """Options take precedence over data for reconfigurable fields."""
    entry = MagicMock()
    entry.entry_id = "test"
    entry.data = {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "original-model",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
        CONF_SHOW_THINKING: False,
    }
    entry.options = {
        CONF_MODEL: "new-model",
        CONF_SYSTEM_PROMPT: "Custom prompt",
        CONF_STREAMING: False,
        CONF_SHOW_THINKING: True,
    }
    entity = HailoAITaskEntity(entry)

    assert entity._model == "new-model"
    assert entity._system_prompt == "Custom prompt"
    assert entity._streaming is False
    assert entity._show_thinking is True
    # Host/port always come from data
    assert entity._host == "localhost"
    assert entity._port == 8000


# ---------------------------------------------------------------------------
# device_info
# ---------------------------------------------------------------------------


def test_device_info(mock_config_entry):
    """device_info returns the expected structure shared with conversation entity."""
    entity = HailoAITaskEntity(mock_config_entry)

    assert entity.device_info == {
        "identifiers": {(DOMAIN, "test_entry_id")},
        "name": "Hailo Ollama (llama3.2:3b)",
        "manufacturer": "Hailo",
        "model": "llama3.2:3b",
    }


# ---------------------------------------------------------------------------
# _async_generate_data — non-streaming success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_data_non_streaming_success():
    """_async_generate_data with streaming=False calls _call_non_streaming and returns result."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    })
    entity._call_non_streaming = AsyncMock(return_value="Generated content.")

    task = MagicMock()
    task.instructions = "Write a poem about smart homes."

    result = await entity._async_generate_data(task)

    entity._call_non_streaming.assert_called_once()
    messages_sent = entity._call_non_streaming.call_args[0][0]
    assert messages_sent[0]["role"] == "system"
    assert messages_sent[1] == {"role": "user", "content": "Write a poem about smart homes."}
    # RunDataTaskResult is the mock from conftest, just verify it was constructed
    assert result is not None


# ---------------------------------------------------------------------------
# _async_generate_data — streaming success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_data_streaming_success():
    """_async_generate_data with streaming=True calls _call_streaming and returns result."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })
    entity._call_streaming = AsyncMock(return_value="Streamed content.")

    task = MagicMock()
    task.instructions = "Summarise today's weather."

    result = await entity._async_generate_data(task)

    entity._call_streaming.assert_called_once()
    messages_sent = entity._call_streaming.call_args[0][0]
    assert messages_sent[1] == {"role": "user", "content": "Summarise today's weather."}
    assert result is not None


# ---------------------------------------------------------------------------
# _async_generate_data — HailoError returns error message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_data_hailo_error_returns_error_message():
    """When HailoError is raised, _async_generate_data returns a result without re-raising."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    })
    entity._call_non_streaming = AsyncMock(
        side_effect=HailoError("service unavailable")
    )

    task = MagicMock()
    task.instructions = "Do something."

    mock_result_cls = MagicMock()
    with patch("custom_components.hailo_ollama.ai_task.RunDataTaskResult", mock_result_cls):
        # Should not raise — error is caught and embedded in the result
        result = await entity._async_generate_data(task)

    assert result is not None
    mock_result_cls.assert_called_once()
    call_kwargs = mock_result_cls.call_args
    data_arg = call_kwargs.kwargs.get("data") or (call_kwargs.args[0] if call_kwargs.args else None)
    assert "service unavailable" in data_arg


# ---------------------------------------------------------------------------
# _async_generate_data — thinking tags are processed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_data_strips_thinking_tags():
    """_process_thinking is applied to the response before returning the result."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    })
    entity._call_non_streaming = AsyncMock(
        return_value="<think>internal reasoning</think>The real answer."
    )

    task = MagicMock()
    task.instructions = "Answer this."

    mock_result_cls = MagicMock()
    with patch("custom_components.hailo_ollama.ai_task.RunDataTaskResult", mock_result_cls):
        await entity._async_generate_data(task)

    mock_result_cls.assert_called_once()
    call_kwargs = mock_result_cls.call_args
    data_arg = call_kwargs.kwargs.get("data") or (call_kwargs.args[0] if call_kwargs.args else None)
    assert data_arg == "The real answer."
    assert "<think>" not in data_arg
