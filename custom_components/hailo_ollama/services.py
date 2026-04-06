"""Service calls for Hailo Ollama."""

from __future__ import annotations

import json
import logging

import aiohttp
import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import CONF_HOST, CONF_PORT, DEFAULT_TIMEOUT, DOMAIN

_LOGGER = logging.getLogger(__name__)

SERVICE_LIST_MODELS = "list_models"
SERVICE_PULL_MODEL = "pull_model"

_LIST_MODELS_SCHEMA = vol.Schema(
    {
        vol.Optional("config_entry_id"): str,
    }
)

_PULL_MODEL_SCHEMA = vol.Schema(
    {
        vol.Required("model"): str,
        vol.Optional("config_entry_id"): str,
    }
)


def _get_base_url(hass: HomeAssistant, entry_id: str | None) -> str:
    """Resolve the base URL for the given entry (or the first configured entry)."""
    entries = hass.config_entries.async_entries(DOMAIN)
    if not entries:
        raise ServiceValidationError("No Hailo Ollama integration configured")

    if entry_id:
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(f"Config entry '{entry_id}' not found")
    else:
        entry = entries[0]

    host = entry.data[CONF_HOST]
    port = entry.data[CONF_PORT]
    return f"http://{host}:{port}"


async def _handle_list_models(hass: HomeAssistant, call: ServiceCall) -> ServiceResponse:
    """Return the list of models available on the Hailo server."""
    base_url = _get_base_url(hass, call.data.get("config_entry_id"))
    session = async_get_clientsession(hass)
    timeout = aiohttp.ClientTimeout(total=10)

    try:
        async with session.get(f"{base_url}/hailo/v1/list", timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ServiceValidationError(
                    f"Failed to list models: HTTP {resp.status}: {body[:200]}"
                )
            data = await resp.json()
    except aiohttp.ClientConnectorError as err:
        raise ServiceValidationError(f"Cannot connect to Hailo-Ollama: {err}") from err
    except aiohttp.ClientError as err:
        raise ServiceValidationError(f"Request error: {err}") from err

    models = data.get("models", [])
    _LOGGER.debug("Listed %d models: %s", len(models), models)
    return {"models": models}


async def _handle_pull_model(hass: HomeAssistant, call: ServiceCall) -> ServiceResponse:
    """Pull a model from the Hailo server and return the final status."""
    model = call.data["model"]
    base_url = _get_base_url(hass, call.data.get("config_entry_id"))
    session = async_get_clientsession(hass)
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT, sock_read=DEFAULT_TIMEOUT)
    payload = {"model": model, "stream": True}

    _LOGGER.info("Pulling model '%s' from %s", model, base_url)

    buffer = b""
    last_status = ""

    try:
        async with session.post(
            f"{base_url}/api/pull", json=payload, timeout=timeout
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ServiceValidationError(
                    f"Pull failed: HTTP {resp.status}: {body[:200]}"
                )

            async for data in resp.content.iter_any():
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        status = chunk.get("status", "")
                        if status:
                            last_status = status
                            _LOGGER.debug("Pull '%s': %s", model, status)
                        if chunk.get("error"):
                            raise ServiceValidationError(
                                f"Pull error: {chunk['error']}"
                            )
                    except json.JSONDecodeError:
                        _LOGGER.warning("Bad pull chunk: %s", line[:100])

            if buffer.strip():
                try:
                    chunk = json.loads(buffer.strip())
                    if chunk.get("status"):
                        last_status = chunk["status"]
                    if chunk.get("error"):
                        raise ServiceValidationError(f"Pull error: {chunk['error']}")
                except json.JSONDecodeError:
                    pass

    except aiohttp.ClientConnectorError as err:
        raise ServiceValidationError(f"Cannot connect to Hailo-Ollama: {err}") from err
    except aiohttp.ClientError as err:
        raise ServiceValidationError(f"Request error: {err}") from err

    _LOGGER.info("Pull '%s' completed: %s", model, last_status)
    return {"model": model, "status": last_status}


def async_register_services(hass: HomeAssistant) -> None:
    """Register Hailo Ollama service calls (idempotent)."""
    if hass.services.has_service(DOMAIN, SERVICE_LIST_MODELS):
        return

    hass.services.async_register(
        DOMAIN,
        SERVICE_LIST_MODELS,
        lambda call: _handle_list_models(hass, call),
        schema=_LIST_MODELS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_PULL_MODEL,
        lambda call: _handle_pull_model(hass, call),
        schema=_PULL_MODEL_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    _LOGGER.debug("Registered Hailo Ollama services")


def async_unregister_services(hass: HomeAssistant) -> None:
    """Unregister services if no entries remain."""
    if hass.config_entries.async_entries(DOMAIN):
        return
    hass.services.async_remove(DOMAIN, SERVICE_LIST_MODELS)
    hass.services.async_remove(DOMAIN, SERVICE_PULL_MODEL)
