"""Tests for Hailo Ollama config flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hailo_ollama.const import (
    CONF_HOST,
    CONF_MODEL,
    CONF_PORT,
    CONF_STREAMING,
    CONF_SYSTEM_PROMPT,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_STREAMING,
    DEFAULT_SYSTEM_PROMPT,
    DOMAIN,
)
from custom_components.hailo_ollama.config_flow import HailoOllamaConfigFlow


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    return hass


@pytest.fixture
def config_flow(mock_hass):
    """Create a config flow instance."""
    flow = HailoOllamaConfigFlow()
    flow.hass = mock_hass
    return flow


@pytest.mark.asyncio
async def test_test_connection_success(config_flow):
    """Test successful connection test."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"version": "1.0.0"})

    mock_session = MagicMock()
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._test_connection("localhost", 8000)

    assert result == "1.0.0"


@pytest.mark.asyncio
async def test_test_connection_failure(config_flow):
    """Test connection test failure."""
    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=Exception("Connection refused"))

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._test_connection("localhost", 8000)

    assert result is None


@pytest.mark.asyncio
async def test_fetch_models_from_api_tags(config_flow):
    """Test fetching models from /api/tags."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={"models": [{"name": "llama3.2:3b"}, {"name": "deepseek-r1:1.5b"}]}
    )

    mock_session = MagicMock()
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._fetch_models("localhost", 8000)

    assert result == ["llama3.2:3b", "deepseek-r1:1.5b"]


@pytest.mark.asyncio
async def test_fetch_models_string_format(config_flow):
    """Test fetching models when returned as strings."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={"models": ["llama3.2:3b", "deepseek-r1:1.5b"]}
    )

    mock_session = MagicMock()
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._fetch_models("localhost", 8000)

    assert result == ["llama3.2:3b", "deepseek-r1:1.5b"]


@pytest.mark.asyncio
async def test_fetch_models_empty(config_flow):
    """Test fetching models when none available."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"models": []})

    mock_session = MagicMock()
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._fetch_models("localhost", 8000)

    assert result == []


def test_config_flow_init():
    """Test config flow initialization."""
    flow = HailoOllamaConfigFlow()

    assert flow._host == DEFAULT_HOST
    assert flow._port == DEFAULT_PORT
    assert flow._models == []


# ---------------------------------------------------------------------------
# _fetch_models: fallback and both-fail paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_models_api_tags_fails_fallback_success(config_flow):
    """When /api/tags raises, fall back to /hailo/v1/list and return its models."""
    fallback_response = AsyncMock()
    fallback_response.status = 200
    fallback_response.json = AsyncMock(return_value={"models": ["hailo-model:1b"]})

    call_count = 0

    def get_side_effect(url, timeout):
        nonlocal call_count
        call_count += 1
        if "api/tags" in url:
            raise Exception("tags endpoint down")
        # fallback endpoint
        return AsyncMock(__aenter__=AsyncMock(return_value=fallback_response))

    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=get_side_effect)

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._fetch_models("localhost", 8000)

    assert result == ["hailo-model:1b"]


@pytest.mark.asyncio
async def test_fetch_models_both_fail(config_flow):
    """When both /api/tags and /hailo/v1/list fail, return an empty list."""
    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=Exception("network error"))

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._fetch_models("localhost", 8000)

    assert result == []


# ---------------------------------------------------------------------------
# async_step_user
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_step_user_no_input(config_flow):
    """Calling async_step_user(None) shows the host/port form."""
    form_result = {"type": "form", "step_id": "user"}
    config_flow.async_show_form = MagicMock(return_value=form_result)

    result = await config_flow.async_step_user(None)

    assert result["step_id"] == "user"
    config_flow.async_show_form.assert_called_once()


@pytest.mark.asyncio
async def test_async_step_user_cannot_connect(config_flow):
    """When _test_connection returns None, show form with cannot_connect error."""
    form_result = {"type": "form", "step_id": "user", "errors": {"base": "cannot_connect"}}
    config_flow.async_show_form = MagicMock(return_value=form_result)
    config_flow._test_connection = AsyncMock(return_value=None)

    user_input = {CONF_HOST: "badhost", CONF_PORT: 9999}
    result = await config_flow.async_step_user(user_input)

    assert result["errors"]["base"] == "cannot_connect"


@pytest.mark.asyncio
async def test_async_step_user_no_models(config_flow):
    """When connection succeeds but _fetch_models returns [], show no_models error."""
    form_result = {"type": "form", "step_id": "user", "errors": {"base": "no_models"}}
    config_flow.async_show_form = MagicMock(return_value=form_result)
    config_flow._test_connection = AsyncMock(return_value="1.0.0")
    config_flow._fetch_models = AsyncMock(return_value=[])

    user_input = {CONF_HOST: "localhost", CONF_PORT: 8000}
    result = await config_flow.async_step_user(user_input)

    assert result["errors"]["base"] == "no_models"


@pytest.mark.asyncio
async def test_async_step_user_success_proceeds_to_pick_model(config_flow):
    """When connection and models succeed, async_step_pick_model is called and form shown."""
    pick_model_form = {"type": "form", "step_id": "pick_model"}
    config_flow.async_show_form = MagicMock(return_value=pick_model_form)
    config_flow._test_connection = AsyncMock(return_value="1.0.0")
    config_flow._fetch_models = AsyncMock(return_value=["llama3.2:3b"])

    user_input = {CONF_HOST: "localhost", CONF_PORT: 8000}
    result = await config_flow.async_step_user(user_input)

    # async_step_pick_model(None) was called internally; it renders the pick_model form
    assert result["step_id"] == "pick_model"


# ---------------------------------------------------------------------------
# async_step_pick_model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_step_pick_model_no_input(config_flow):
    """Calling async_step_pick_model(None) shows the pick_model form."""
    form_result = {"type": "form", "step_id": "pick_model"}
    config_flow.async_show_form = MagicMock(return_value=form_result)
    config_flow._models = ["llama3.2:3b", "deepseek-r1:1.5b"]

    result = await config_flow.async_step_pick_model(None)

    assert result["step_id"] == "pick_model"
    config_flow.async_show_form.assert_called_once()


@pytest.mark.asyncio
async def test_async_step_pick_model_with_input_creates_entry(config_flow):
    """Providing model input creates a config entry with the correct data."""
    from custom_components.hailo_ollama.const import (
        CONF_SHOW_THINKING,
        DEFAULT_SHOW_THINKING,
    )

    created_entry = {
        "type": "create_entry",
        "title": "Hailo (llama3.2:3b)",
        "data": {
            CONF_HOST: "localhost",
            CONF_PORT: 8000,
            CONF_MODEL: "llama3.2:3b",
            CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
            CONF_STREAMING: True,
            CONF_SHOW_THINKING: DEFAULT_SHOW_THINKING,
        },
    }
    config_flow.async_create_entry = MagicMock(return_value=created_entry)
    config_flow._host = "localhost"
    config_flow._port = 8000
    config_flow._models = ["llama3.2:3b"]

    user_input = {
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
        CONF_SHOW_THINKING: DEFAULT_SHOW_THINKING,
    }
    result = await config_flow.async_step_pick_model(user_input)

    config_flow.async_create_entry.assert_called_once()
    call_kwargs = config_flow.async_create_entry.call_args
    assert call_kwargs.kwargs["title"] == "Hailo (llama3.2:3b)"
    assert call_kwargs.kwargs["data"][CONF_MODEL] == "llama3.2:3b"
    assert call_kwargs.kwargs["data"][CONF_HOST] == "localhost"
    assert call_kwargs.kwargs["data"][CONF_PORT] == 8000
