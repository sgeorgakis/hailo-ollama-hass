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


def _make_chat_log(conversation_id: str = "test-conv-id") -> MagicMock:
    """Build a mock ChatLog."""
    chat_log = MagicMock()
    chat_log.conversation_id = conversation_id
    return chat_log


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
    chat_log = _make_chat_log()

    result = await entity._async_generate_data(task, chat_log)

    entity._call_non_streaming.assert_called_once()
    messages_sent = entity._call_non_streaming.call_args[0][0]
    assert messages_sent[0]["role"] == "system"
    assert messages_sent[1] == {"role": "user", "content": "Write a poem about smart homes."}
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
    chat_log = _make_chat_log()

    result = await entity._async_generate_data(task, chat_log)

    entity._call_streaming.assert_called_once()
    messages_sent = entity._call_streaming.call_args[0][0]
    assert messages_sent[1] == {"role": "user", "content": "Summarise today's weather."}
    assert result is not None


# ---------------------------------------------------------------------------
# _async_generate_data — conversation_id is propagated from chat_log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_data_uses_chat_log_conversation_id():
    """GenDataTaskResult receives the conversation_id from the chat_log."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    })
    entity._call_non_streaming = AsyncMock(return_value="Answer.")
    chat_log = _make_chat_log("my-specific-conv-id")

    mock_result_cls = MagicMock()
    with patch("custom_components.hailo_ollama.ai_task.GenDataTaskResult", mock_result_cls):
        await entity._async_generate_data(MagicMock(instructions="Q"), chat_log)

    call_kwargs = mock_result_cls.call_args.kwargs
    assert call_kwargs["conversation_id"] == "my-specific-conv-id"


# ---------------------------------------------------------------------------
# _async_generate_data — HailoError re-raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_data_hailo_error_reraises():
    """When HailoError is raised, _async_generate_data re-raises it."""
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

    with pytest.raises(HailoError, match="service unavailable"):
        await entity._async_generate_data(task, _make_chat_log())


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
    with patch("custom_components.hailo_ollama.ai_task.GenDataTaskResult", mock_result_cls):
        await entity._async_generate_data(task, _make_chat_log())

    mock_result_cls.assert_called_once()
    call_kwargs = mock_result_cls.call_args.kwargs
    assert call_kwargs["data"] == "The real answer."
    assert "<think>" not in call_kwargs["data"]
