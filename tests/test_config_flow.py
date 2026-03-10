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
