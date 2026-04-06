"""Config flow for Hailo Ollama."""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import SelectSelector, SelectSelectorConfig

from .const import (
    CONF_HOST,
    CONF_MODEL,
    CONF_PORT,
    CONF_SHOW_THINKING,
    CONF_STREAMING,
    CONF_SYSTEM_PROMPT,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_SHOW_THINKING,
    DEFAULT_STREAMING,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

# Sentinel value shown in the model dropdown to navigate to the pull step.
PULL_NEW_MODEL_SENTINEL = "Download a new model…"

# Field key used in the pull_model step.
CONF_MODEL_TO_PULL = "model_to_pull"


async def _pull_model(
    session: aiohttp.ClientSession, host: str, port: int, model_name: str
) -> tuple[bool, str]:
    """Pull a model from the Hailo-Ollama server. Returns (success, status_or_error)."""
    url = f"http://{host}:{port}/api/pull"
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT, sock_read=DEFAULT_TIMEOUT)
    buffer = b""
    last_status = ""
    try:
        async with session.post(
            url, json={"model": model_name, "stream": True}, timeout=timeout
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return False, f"HTTP {resp.status}: {body[:100]}"
            async for data in resp.content.iter_any():
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            return False, chunk["error"]
                        if chunk.get("status"):
                            last_status = chunk["status"]
                    except json.JSONDecodeError:
                        pass
            if buffer.strip():
                try:
                    chunk = json.loads(buffer.strip())
                    if chunk.get("error"):
                        return False, chunk["error"]
                    if chunk.get("status"):
                        last_status = chunk["status"]
                except json.JSONDecodeError:
                    pass
    except aiohttp.ClientConnectorError as err:
        return False, str(err)
    except TimeoutError:
        return False, "Request timed out"
    except aiohttp.ClientError as err:
        return False, str(err)
    return True, last_status


class HailoOllamaConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Hailo Ollama."""

    VERSION = 1

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the options flow handler."""
        return HailoOllamaOptionsFlow(config_entry)

    def __init__(self) -> None:
        """Initialize."""
        self._host: str = DEFAULT_HOST
        self._port: int = DEFAULT_PORT
        self._models: list[str] = []
        self._available_models: list[str] = []

    async def _test_connection(self, host: str, port: int) -> str | None:
        """Test connection via /api/version. Returns version or None."""
        session = async_get_clientsession(self.hass)
        url = f"http://{host}:{port}/api/version"
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("version", "unknown")
        except Exception as err:
            _LOGGER.error("Connection test failed: %s", err)
        return None

    async def _fetch_models(self, host: str, port: int) -> list[str]:
        """Fetch downloaded models from /api/tags."""
        session = async_get_clientsession(self.hass)
        url = f"http://{host}:{port}/api/tags"
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("models", [])
                    if models and isinstance(models[0], dict):
                        return [m["name"] for m in models]
                    if models and isinstance(models[0], str):
                        return models
        except Exception as err:
            _LOGGER.warning("Failed to fetch from /api/tags: %s", err)

        # Fallback to /hailo/v1/list
        _LOGGER.info("Trying /hailo/v1/list as fallback")
        url = f"http://{host}:{port}/hailo/v1/list"
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("models", [])
        except Exception as err:
            _LOGGER.error("Failed to fetch models: %s", err)

        return []

    async def _fetch_available_models(self, host: str, port: int) -> list[str]:
        """Fetch models available to download from /hailo/v1/list."""
        session = async_get_clientsession(self.hass)
        url = f"http://{host}:{port}/hailo/v1/list"
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("models", [])
        except Exception as err:
            _LOGGER.warning("Failed to fetch available models: %s", err)
        return []

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle the initial step — host and port."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._host = user_input[CONF_HOST]
            self._port = user_input[CONF_PORT]

            version = await self._test_connection(self._host, self._port)
            if version is None:
                errors["base"] = "cannot_connect"
            else:
                _LOGGER.info("Connected to Hailo Ollama %s", version)
                self._models = await self._fetch_models(self._host, self._port)
                if not self._models:
                    errors["base"] = "no_models"
                else:
                    return await self.async_step_pick_model()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_HOST, default=DEFAULT_HOST): str,
                vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
            }),
            errors=errors,
        )

    async def async_step_pick_model(
        self, user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Let the user pick a model or navigate to the download step."""
        if user_input is not None:
            if user_input[CONF_MODEL] == PULL_NEW_MODEL_SENTINEL:
                return await self.async_step_pull_model()
            return self.async_create_entry(
                title=f"Hailo ({user_input[CONF_MODEL]})",
                data={
                    CONF_HOST: self._host,
                    CONF_PORT: self._port,
                    CONF_MODEL: user_input[CONF_MODEL],
                    CONF_SYSTEM_PROMPT: user_input.get(
                        CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
                    ),
                    CONF_STREAMING: user_input.get(
                        CONF_STREAMING, DEFAULT_STREAMING
                    ),
                    CONF_SHOW_THINKING: user_input.get(
                        CONF_SHOW_THINKING, DEFAULT_SHOW_THINKING
                    ),
                },
            )

        default_model = (
            DEFAULT_MODEL if DEFAULT_MODEL in self._models else self._models[0]
        )
        options = self._models + [PULL_NEW_MODEL_SENTINEL]

        return self.async_show_form(
            step_id="pick_model",
            data_schema=vol.Schema({
                vol.Required(CONF_MODEL, default=default_model): vol.In(options),
                vol.Optional(
                    CONF_SYSTEM_PROMPT, default=DEFAULT_SYSTEM_PROMPT
                ): str,
                vol.Optional(
                    CONF_STREAMING, default=DEFAULT_STREAMING
                ): bool,
                vol.Optional(
                    CONF_SHOW_THINKING, default=DEFAULT_SHOW_THINKING
                ): bool,
            }),
        )

    async def async_step_pull_model(
        self, user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Show available models and pull the selected one."""
        errors: dict[str, str] = {}

        if not self._available_models:
            self._available_models = await self._fetch_available_models(
                self._host, self._port
            )

        if user_input is not None:
            model_name = user_input[CONF_MODEL_TO_PULL].strip()
            session = async_get_clientsession(self.hass)
            success, msg = await _pull_model(session, self._host, self._port, model_name)
            if success:
                _LOGGER.info("Pulled model '%s': %s", model_name, msg)
                self._models = await self._fetch_models(self._host, self._port)
                self._available_models = []
                return await self.async_step_pick_model()
            else:
                _LOGGER.error("Failed to pull model '%s': %s", model_name, msg)
                errors[CONF_MODEL_TO_PULL] = "pull_failed"

        downloadable = [m for m in self._available_models if m not in self._models]

        return self.async_show_form(
            step_id="pull_model",
            data_schema=vol.Schema({
                vol.Required(CONF_MODEL_TO_PULL): SelectSelector(
                    SelectSelectorConfig(options=downloadable)
                ),
            }),
            errors=errors,
        )


class HailoOllamaOptionsFlow(OptionsFlow):
    """Handle options for an existing Hailo Ollama entry."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize."""
        self._config_entry = config_entry
        self._models: list[str] = []
        self._available_models: list[str] = []

    async def _fetch_models(self, host: str, port: int) -> list[str]:
        """Fetch downloaded models from /api/tags."""
        session = async_get_clientsession(self.hass)
        url = f"http://{host}:{port}/api/tags"
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("models", [])
                    if models and isinstance(models[0], dict):
                        return [m["name"] for m in models]
                    elif models and isinstance(models[0], str):
                        return models
        except Exception as err:
            _LOGGER.warning("Options flow: failed to fetch models: %s", err)
        return []

    async def _fetch_available_models(self, host: str, port: int) -> list[str]:
        """Fetch models available to download from /hailo/v1/list."""
        session = async_get_clientsession(self.hass)
        url = f"http://{host}:{port}/hailo/v1/list"
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("models", [])
        except Exception as err:
            _LOGGER.warning("Options flow: failed to fetch available models: %s", err)
        return []

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Fetch downloaded models and show the options form."""
        host = self._config_entry.data[CONF_HOST]
        port = self._config_entry.data[CONF_PORT]

        if not self._models:
            self._models = await self._fetch_models(host, port)

        if user_input is not None:
            if user_input[CONF_MODEL] == PULL_NEW_MODEL_SENTINEL:
                return await self.async_step_pull_model()
            return self.async_create_entry(title="", data=user_input)

        current = self._config_entry.options or self._config_entry.data
        current_model = current.get(CONF_MODEL, "")
        available_models = self._models or ([current_model] if current_model else [])
        default_model = current_model if current_model in available_models else (available_models[0] if available_models else current_model)
        options = available_models + [PULL_NEW_MODEL_SENTINEL]

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(CONF_MODEL, default=default_model): vol.In(options),
                vol.Optional(
                    CONF_SYSTEM_PROMPT,
                    default=current.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
                ): str,
                vol.Optional(
                    CONF_STREAMING,
                    default=current.get(CONF_STREAMING, DEFAULT_STREAMING),
                ): bool,
                vol.Optional(
                    CONF_SHOW_THINKING,
                    default=current.get(CONF_SHOW_THINKING, DEFAULT_SHOW_THINKING),
                ): bool,
            }),
        )

    async def async_step_pull_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show available models and pull the selected one."""
        host = self._config_entry.data[CONF_HOST]
        port = self._config_entry.data[CONF_PORT]
        errors: dict[str, str] = {}

        if not self._available_models:
            self._available_models = await self._fetch_available_models(host, port)

        if user_input is not None:
            model_name = user_input[CONF_MODEL_TO_PULL].strip()
            session = async_get_clientsession(self.hass)
            success, msg = await _pull_model(session, host, port, model_name)
            if success:
                _LOGGER.info("Pulled model '%s': %s", model_name, msg)
                self._models = []  # force re-fetch on return to init
                self._available_models = []
                return await self.async_step_init()
            else:
                _LOGGER.error("Failed to pull model '%s': %s", model_name, msg)
                errors[CONF_MODEL_TO_PULL] = "pull_failed"

        downloadable = [m for m in self._available_models if m not in self._models]

        return self.async_show_form(
            step_id="pull_model",
            data_schema=vol.Schema({
                vol.Required(CONF_MODEL_TO_PULL): SelectSelector(
                    SelectSelectorConfig(options=downloadable)
                ),
            }),
            errors=errors,
        )
