"""Tests for Hailo Ollama conversation entity."""

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hailo_ollama.const import (
    CONF_HOST,
    CONF_MODEL,
    CONF_PORT,
    CONF_STREAMING,
    CONF_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DOMAIN,
)
from custom_components.hailo_ollama.conversation import (
    HailoError,
    HailoOllamaConversationEntity,
    THINK_TAG_RE,
)


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry."""
    entry = MagicMock()
    entry.entry_id = "test_entry_id"
    entry.data = {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    }
    return entry


def test_think_tag_regex():
    """Test the regex for stripping <think> tags."""
    text_with_think = "<think>Let me think about this...</think>Here is my response."
    result = THINK_TAG_RE.sub("", text_with_think).strip()
    assert result == "Here is my response."

    text_multiline = "<think>\nThinking...\nMore thinking...\n</think>\nActual response"
    result = THINK_TAG_RE.sub("", text_multiline).strip()
    assert result == "Actual response"

    text_no_think = "Just a normal response"
    result = THINK_TAG_RE.sub("", text_no_think).strip()
    assert result == "Just a normal response"


def test_conversation_entity_init(mock_config_entry):
    """Test conversation entity initialization."""
    entity = HailoOllamaConversationEntity(mock_config_entry)

    assert entity._host == "localhost"
    assert entity._port == 8000
    assert entity._model == "llama3.2:3b"
    assert entity._system_prompt == DEFAULT_SYSTEM_PROMPT
    assert entity._streaming is False
    assert entity._base_url == "http://localhost:8000"


def test_conversation_entity_properties(mock_config_entry):
    """Test conversation entity properties."""
    entity = HailoOllamaConversationEntity(mock_config_entry)

    assert entity.supported_languages == ["en"]
    assert entity.device_info == {
        "identifiers": {(DOMAIN, "test_entry_id")},
        "name": "Hailo Ollama (llama3.2:3b)",
        "manufacturer": "Hailo",
        "model": "llama3.2:3b",
    }


def test_build_payload(mock_config_entry):
    """Test payload building."""
    entity = HailoOllamaConversationEntity(mock_config_entry)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    payload = entity._build_payload(messages, stream=False)
    assert payload == {
        "model": "llama3.2:3b",
        "messages": messages,
        "stream": False,
    }

    payload_stream = entity._build_payload(messages, stream=True)
    assert payload_stream["stream"] is True


def test_hailo_error():
    """Test HailoError exception."""
    error = HailoError("Connection failed")
    assert str(error) == "Connection failed"


@pytest.mark.asyncio
async def test_call_non_streaming_success(mock_config_entry, mock_chat_response):
    """Test successful non-streaming API call."""
    entity = HailoOllamaConversationEntity(mock_config_entry)
    entity.hass = MagicMock()

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=mock_chat_response)

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        messages = [{"role": "user", "content": "Hello"}]
        result = await entity._call_non_streaming(messages)

    assert result == "Hello! How can I help you today?"


@pytest.mark.asyncio
async def test_call_non_streaming_http_error(mock_config_entry):
    """Test non-streaming API call with HTTP error."""
    entity = HailoOllamaConversationEntity(mock_config_entry)
    entity.hass = MagicMock()

    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.text = AsyncMock(return_value="Internal Server Error")

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(HailoError) as exc_info:
            await entity._call_non_streaming(messages)

    assert "HTTP 500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_call_non_streaming_empty_response(mock_config_entry):
    """Test non-streaming API call with empty response."""
    entity = HailoOllamaConversationEntity(mock_config_entry)
    entity.hass = MagicMock()

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"message": {}})

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(HailoError) as exc_info:
            await entity._call_non_streaming(messages)

    assert "No content in response" in str(exc_info.value)


@pytest.mark.asyncio
async def test_call_non_streaming_connection_error(mock_config_entry):
    """Test non-streaming API call with connection error."""
    import aiohttp

    entity = HailoOllamaConversationEntity(mock_config_entry)
    entity.hass = MagicMock()

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused")))
        )
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(HailoError) as exc_info:
            await entity._call_non_streaming(messages)

    assert "Cannot connect" in str(exc_info.value)
