"""Tests for Hailo Ollama service calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hailo_ollama.const import CONF_HOST, CONF_PORT, DOMAIN
from custom_components.hailo_ollama.services import (
    SERVICE_LIST_MODELS,
    SERVICE_PULL_MODEL,
    _get_base_url,
    _handle_list_models,
    _handle_pull_model,
    async_register_services,
    async_unregister_services,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hass(entries=None):
    hass = MagicMock()
    hass.services.has_service.return_value = False
    hass.config_entries.async_entries.return_value = entries or []
    return hass


def _make_entry(entry_id="entry1", host="localhost", port=8000):
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.domain = DOMAIN
    entry.data = {CONF_HOST: host, CONF_PORT: port}
    return entry


def _make_call(data: dict):
    call = MagicMock()
    call.data = data
    return call


# ---------------------------------------------------------------------------
# _get_base_url
# ---------------------------------------------------------------------------


def test_get_base_url_uses_first_entry_when_no_id_given():
    entry = _make_entry(host="myhost", port=9000)
    hass = _make_hass(entries=[entry])
    assert _get_base_url(hass, None) == "http://myhost:9000"


def test_get_base_url_uses_specific_entry_id():
    entry = _make_entry(entry_id="abc", host="server2", port=8001)
    hass = _make_hass(entries=[entry])
    hass.config_entries.async_get_entry.return_value = entry
    assert _get_base_url(hass, "abc") == "http://server2:8001"


def test_get_base_url_raises_when_no_entries():
    from homeassistant.exceptions import ServiceValidationError
    hass = _make_hass(entries=[])
    with pytest.raises(ServiceValidationError):
        _get_base_url(hass, None)


def test_get_base_url_raises_when_entry_id_not_found():
    from homeassistant.exceptions import ServiceValidationError
    hass = _make_hass(entries=[_make_entry()])
    hass.config_entries.async_get_entry.return_value = None
    with pytest.raises(ServiceValidationError):
        _get_base_url(hass, "nonexistent")


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models_returns_model_list():
    entry = _make_entry()
    hass = _make_hass(entries=[entry])
    call = _make_call({})

    resp = MagicMock()
    resp.status = 200
    resp.json = AsyncMock(return_value={"models": ["llama3.2:3b", "qwen2:1.5b"]})
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get.return_value = ctx

    with patch("custom_components.hailo_ollama.services.async_get_clientsession", return_value=session):
        result = await _handle_list_models(hass, call)

    assert result == {"models": ["llama3.2:3b", "qwen2:1.5b"]}


@pytest.mark.asyncio
async def test_list_models_raises_on_http_error():
    from homeassistant.exceptions import ServiceValidationError
    entry = _make_entry()
    hass = _make_hass(entries=[entry])
    call = _make_call({})

    resp = MagicMock()
    resp.status = 500
    resp.text = AsyncMock(return_value="internal error")
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get.return_value = ctx

    with patch("custom_components.hailo_ollama.services.async_get_clientsession", return_value=session):
        with pytest.raises(ServiceValidationError):
            await _handle_list_models(hass, call)


@pytest.mark.asyncio
async def test_list_models_raises_on_connection_error():
    import aiohttp
    from homeassistant.exceptions import ServiceValidationError
    entry = _make_entry()
    hass = _make_hass(entries=[entry])
    call = _make_call({})

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientConnectorError(MagicMock(), MagicMock()))
    ctx.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get.return_value = ctx

    with patch("custom_components.hailo_ollama.services.async_get_clientsession", return_value=session):
        with pytest.raises(ServiceValidationError):
            await _handle_list_models(hass, call)


@pytest.mark.asyncio
async def test_list_models_uses_config_entry_id():
    entry = _make_entry(entry_id="specific", host="remote", port=9999)
    hass = _make_hass(entries=[entry])
    hass.config_entries.async_get_entry.return_value = entry
    call = _make_call({"config_entry_id": "specific"})

    resp = MagicMock()
    resp.status = 200
    resp.json = AsyncMock(return_value={"models": []})
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get.return_value = ctx

    with patch("custom_components.hailo_ollama.services.async_get_clientsession", return_value=session):
        await _handle_list_models(hass, call)

    session.get.assert_called_once()
    url = session.get.call_args[0][0]
    assert "remote:9999" in url


# ---------------------------------------------------------------------------
# pull_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pull_model_returns_success_status():
    entry = _make_entry()
    hass = _make_hass(entries=[entry])
    call = _make_call({"model": "llama3.2:3b"})

    chunks = [
        b'{"status":"pulling manifest"}\n',
        b'{"status":"downloading","total":1000,"completed":500}\n',
        b'{"status":"success"}\n',
    ]

    resp = MagicMock()
    resp.status = 200

    async def fake_iter():
        for chunk in chunks:
            yield chunk

    resp.content.iter_any = fake_iter
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = ctx

    with patch("custom_components.hailo_ollama.services.async_get_clientsession", return_value=session):
        result = await _handle_pull_model(hass, call)

    assert result == {"model": "llama3.2:3b", "status": "success"}


@pytest.mark.asyncio
async def test_pull_model_raises_on_error_chunk():
    from homeassistant.exceptions import ServiceValidationError
    entry = _make_entry()
    hass = _make_hass(entries=[entry])
    call = _make_call({"model": "badmodel:tag"})

    chunks = [b'{"error":"model not found"}\n']

    resp = MagicMock()
    resp.status = 200

    async def fake_iter():
        for chunk in chunks:
            yield chunk

    resp.content.iter_any = fake_iter
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = ctx

    with patch("custom_components.hailo_ollama.services.async_get_clientsession", return_value=session):
        with pytest.raises(ServiceValidationError, match="model not found"):
            await _handle_pull_model(hass, call)


@pytest.mark.asyncio
async def test_pull_model_raises_on_http_error():
    from homeassistant.exceptions import ServiceValidationError
    entry = _make_entry()
    hass = _make_hass(entries=[entry])
    call = _make_call({"model": "llama3.2:3b"})

    resp = MagicMock()
    resp.status = 404
    resp.text = AsyncMock(return_value="not found")
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = ctx

    with patch("custom_components.hailo_ollama.services.async_get_clientsession", return_value=session):
        with pytest.raises(ServiceValidationError):
            await _handle_pull_model(hass, call)


@pytest.mark.asyncio
async def test_pull_model_raises_on_connection_error():
    import aiohttp
    from homeassistant.exceptions import ServiceValidationError
    entry = _make_entry()
    hass = _make_hass(entries=[entry])
    call = _make_call({"model": "llama3.2:3b"})

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientConnectorError(MagicMock(), MagicMock()))
    ctx.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = ctx

    with patch("custom_components.hailo_ollama.services.async_get_clientsession", return_value=session):
        with pytest.raises(ServiceValidationError):
            await _handle_pull_model(hass, call)


# ---------------------------------------------------------------------------
# Service registration
# ---------------------------------------------------------------------------


def test_async_register_services_registers_both():
    hass = MagicMock()
    hass.services.has_service.return_value = False

    async_register_services(hass)

    assert hass.services.async_register.call_count == 2
    registered = {call[0][1] for call in hass.services.async_register.call_args_list}
    assert SERVICE_LIST_MODELS in registered
    assert SERVICE_PULL_MODEL in registered


def test_async_register_services_is_idempotent():
    hass = MagicMock()
    hass.services.has_service.return_value = True  # already registered

    async_register_services(hass)

    hass.services.async_register.assert_not_called()


def test_async_unregister_services_removes_when_no_entries_left():
    hass = MagicMock()
    hass.config_entries.async_entries.return_value = []

    async_unregister_services(hass)

    assert hass.services.async_remove.call_count == 2


def test_async_unregister_services_keeps_when_entries_remain():
    hass = MagicMock()
    hass.config_entries.async_entries.return_value = [_make_entry()]

    async_unregister_services(hass)

    hass.services.async_remove.assert_not_called()
