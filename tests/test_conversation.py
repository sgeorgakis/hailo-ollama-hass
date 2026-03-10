"""Tests for Hailo Ollama conversation entity."""

import json
import re
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


def test_show_thinking_false_strips_tags(mock_config_entry):
    """When show_thinking is False, <think> tags are stripped from the response."""
    mock_config_entry.data = {**mock_config_entry.data, CONF_SHOW_THINKING: False}
    entity = HailoOllamaConversationEntity(mock_config_entry)

    response = "<think>Internal reasoning here.</think>The actual answer."
    from custom_components.hailo_ollama.conversation import THINK_TAG_RE
    result = THINK_TAG_RE.sub("", response).strip() if not entity._show_thinking else response.strip()

    assert result == "The actual answer."
    assert "<think>" not in result


def test_show_thinking_true_keeps_tags(mock_config_entry):
    """When show_thinking is True, <think> tags are preserved in the response."""
    mock_config_entry.data = {**mock_config_entry.data, CONF_SHOW_THINKING: True}
    entity = HailoOllamaConversationEntity(mock_config_entry)

    response = "<think>Internal reasoning here.</think>The actual answer."
    from custom_components.hailo_ollama.conversation import THINK_TAG_RE
    result = THINK_TAG_RE.sub("", response).strip() if not entity._show_thinking else response.strip()

    assert "<think>" in result
    assert "Internal reasoning here." in result


def test_show_thinking_defaults_to_false(mock_config_entry):
    """show_thinking defaults to False when not set in config entry."""
    entry = MagicMock()
    entry.entry_id = "test"
    entry.data = {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
        # CONF_SHOW_THINKING intentionally omitted
    }
    entity = HailoOllamaConversationEntity(entry)
    assert entity._show_thinking is False


# ---------------------------------------------------------------------------
# async_setup_entry (conversation platform)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_setup_entry_adds_entity():
    """async_setup_entry should add exactly one HailoOllamaConversationEntity."""
    from custom_components.hailo_ollama.conversation import async_setup_entry

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    }
    hass = MagicMock()
    async_add_entities = MagicMock()

    await async_setup_entry(hass, entry, async_add_entities)

    async_add_entities.assert_called_once()
    entities = async_add_entities.call_args[0][0]
    assert len(entities) == 1
    assert isinstance(entities[0], HailoOllamaConversationEntity)


# ---------------------------------------------------------------------------
# Helper: build an entity with hass attached
# ---------------------------------------------------------------------------

def _make_entity(config_entry_data: dict) -> HailoOllamaConversationEntity:
    entry = MagicMock()
    entry.entry_id = "test_entry_id"
    entry.data = config_entry_data
    entity = HailoOllamaConversationEntity(entry)
    entity.hass = MagicMock()
    return entity


# ---------------------------------------------------------------------------
# async_process
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_process_non_streaming():
    """async_process with streaming=False calls _call_non_streaming and returns result."""
    import sys
    mock_conversation = sys.modules["homeassistant.components.conversation"]

    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    })
    entity._call_non_streaming = AsyncMock(return_value="Hello back!")

    user_input = MagicMock()
    user_input.text = "Hello"
    user_input.agent_id = "test_agent"

    chat_log = MagicMock()
    chat_log.content = []
    chat_log.async_add_assistant_content_from_result = MagicMock()

    result = await entity.async_process(user_input, chat_log)

    entity._call_non_streaming.assert_called_once()
    messages_sent = entity._call_non_streaming.call_args[0][0]
    assert messages_sent[-1]["content"] == "Hello"
    assert messages_sent[0]["role"] == "system"

    chat_log.async_add_assistant_content_from_result.assert_called_once()
    assert result is not None


@pytest.mark.asyncio
async def test_async_process_streaming():
    """async_process with streaming=True calls _call_streaming and returns result."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })
    entity._call_streaming = AsyncMock(return_value="Streaming response")

    user_input = MagicMock()
    user_input.text = "Hi"
    user_input.agent_id = "agent_x"

    chat_log = MagicMock()
    chat_log.content = []
    chat_log.async_add_assistant_content_from_result = MagicMock()

    result = await entity.async_process(user_input, chat_log)

    entity._call_streaming.assert_called_once()
    assert result is not None


@pytest.mark.asyncio
async def test_async_process_hailo_error():
    """When _call_non_streaming raises HailoError, result content contains the error."""
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

    user_input = MagicMock()
    user_input.text = "Hello"
    user_input.agent_id = "agent"

    chat_log = MagicMock()
    chat_log.content = []
    chat_log.async_add_assistant_content_from_result = MagicMock()

    result = await entity.async_process(user_input, chat_log)

    # The assistant content should mention the error
    assistant_content_call = chat_log.async_add_assistant_content_from_result.call_args[0][0]
    assert "service unavailable" in assistant_content_call.content


@pytest.mark.asyncio
async def test_async_process_with_chat_history():
    """Chat history messages (user/assistant roles) are included in the API payload."""
    import sys
    mock_conversation = sys.modules["homeassistant.components.conversation"]

    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    })
    entity._call_non_streaming = AsyncMock(return_value="response")

    # Build history entries that pass isinstance(entry, conversation.ChatMessage)
    # In conftest mock_conversation.ChatMessage is MagicMock (a class), so
    # we use spec to create instances that pass isinstance checks.
    ChatMessage = mock_conversation.ChatMessage

    history_user = MagicMock(spec=ChatMessage)
    history_user.role = "user"
    history_user.content = "Previous question"

    history_assistant = MagicMock(spec=ChatMessage)
    history_assistant.role = "assistant"
    history_assistant.content = "Previous answer"

    user_input = MagicMock()
    user_input.text = "Follow-up"
    user_input.agent_id = "agent"

    chat_log = MagicMock()
    chat_log.content = [history_user, history_assistant]
    chat_log.async_add_assistant_content_from_result = MagicMock()

    await entity.async_process(user_input, chat_log)

    messages_sent = entity._call_non_streaming.call_args[0][0]
    roles = [m["role"] for m in messages_sent]
    assert "system" in roles
    assert messages_sent[-1]["content"] == "Follow-up"


# ---------------------------------------------------------------------------
# _call_non_streaming: ClientPayloadError fallback and TimeoutError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_non_streaming_payload_error_falls_back_to_streaming():
    """ClientPayloadError during non-streaming causes fallback to _call_streaming."""
    import aiohttp

    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    })
    entity._call_streaming = AsyncMock(return_value="fallback streamed")

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(side_effect=aiohttp.ClientPayloadError("truncated"))

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

    assert result == "fallback streamed"
    entity._call_streaming.assert_called_once_with(messages)


@pytest.mark.asyncio
async def test_call_non_streaming_timeout():
    """TimeoutError during non-streaming raises HailoError with 'Timed out'."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: False,
    })

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(side_effect=TimeoutError("timed out")))
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(HailoError) as exc_info:
            await entity._call_non_streaming(messages)

    assert "Timed out" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _call_streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_streaming_success():
    """Streaming collects ndjson chunks and concatenates content."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })

    chunks = [
        b'{"model":"llama3.2:3b","message":{"role":"assistant","content":"Hello"},"done":false}\n',
        b'{"model":"llama3.2:3b","message":{"role":"assistant","content":" world"},"done":false}\n',
        b'{"model":"llama3.2:3b","message":{"role":"assistant","content":"!"},"done":false}\n',
    ]

    mock_response = AsyncMock()
    mock_response.status = 200

    async def fake_iter():
        for chunk in chunks:
            yield chunk

    mock_response.content.iter_any = fake_iter

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        messages = [{"role": "user", "content": "Hi"}]
        result = await entity._call_streaming(messages)

    assert result == "Hello world!"


@pytest.mark.asyncio
async def test_call_streaming_http_error():
    """Non-200 response during streaming raises HailoError."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })

    mock_response = AsyncMock()
    mock_response.status = 503
    mock_response.text = AsyncMock(return_value="Service Unavailable")

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        with pytest.raises(HailoError) as exc_info:
            await entity._call_streaming([{"role": "user", "content": "Hi"}])

    assert "HTTP 503" in str(exc_info.value)


@pytest.mark.asyncio
async def test_call_streaming_connection_error():
    """ClientConnectorError during streaming raises HailoError with 'Cannot connect'."""
    import aiohttp

    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(
                side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("refused"))
            )
        )
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        with pytest.raises(HailoError) as exc_info:
            await entity._call_streaming([{"role": "user", "content": "Hi"}])

    assert "Cannot connect" in str(exc_info.value)


@pytest.mark.asyncio
async def test_call_streaming_timeout():
    """TimeoutError during streaming raises HailoError with 'Timed out'."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(side_effect=TimeoutError("deadline exceeded"))
        )
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        with pytest.raises(HailoError) as exc_info:
            await entity._call_streaming([{"role": "user", "content": "Hi"}])

    assert "Timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_call_streaming_no_chunks():
    """Streaming that yields no parseable chunks raises HailoError about 0 chunks."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })

    mock_response = AsyncMock()
    mock_response.status = 200

    async def fake_iter():
        # yield nothing — empty stream
        return
        yield  # make it an async generator

    mock_response.content.iter_any = fake_iter

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        with pytest.raises(HailoError) as exc_info:
            await entity._call_streaming([{"role": "user", "content": "Hi"}])

    assert "0 chunks" in str(exc_info.value)


@pytest.mark.asyncio
async def test_call_streaming_last_chunk_has_full_content():
    """When the last done=true chunk has full content, return it directly."""
    entity = _make_entity({
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })

    # Last chunk has done=true and full accumulated content
    chunks = [
        b'{"message":{"role":"assistant","content":"He"},"done":false}\n',
        b'{"message":{"role":"assistant","content":"llo"},"done":false}\n',
        b'{"message":{"role":"assistant","content":"Hello, full response!"},"done":true}\n',
    ]

    mock_response = AsyncMock()
    mock_response.status = 200

    async def fake_iter():
        for chunk in chunks:
            yield chunk

    mock_response.content.iter_any = fake_iter

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.conversation.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await entity._call_streaming([{"role": "user", "content": "Hi"}])

    # The last chunk's full content is returned directly, not concatenated
    assert result == "Hello, full response!"
