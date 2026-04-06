"""Config flow for Hailo Ollama."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.helpers.aiohttp_client import async_get_clientsession

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
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


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
        """Let the user pick a model, system prompt, and streaming mode."""
        if user_input is not None:
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

        return self.async_show_form(
            step_id="pick_model",
            data_schema=vol.Schema({
                vol.Required(CONF_MODEL, default=default_model): vol.In(
                    self._models
                ),
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


class HailoOllamaOptionsFlow(OptionsFlow):
    """Handle options for an existing Hailo Ollama entry."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize."""
        self._config_entry = config_entry
        self._models: list[str] = []

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Fetch models and show the options form."""
        host = self._config_entry.data[CONF_HOST]
        port = self._config_entry.data[CONF_PORT]

        if not self._models:
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
                            self._models = [m["name"] for m in models]
                        elif models and isinstance(models[0], str):
                            self._models = models
            except Exception as err:
                _LOGGER.warning("Options flow: failed to fetch models: %s", err)

        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        current = self._config_entry.options or self._config_entry.data
        current_model = current.get(CONF_MODEL, "")
        available_models = self._models or [current_model] if current_model else self._models
        default_model = current_model if current_model in available_models else (available_models[0] if available_models else current_model)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(CONF_MODEL, default=default_model): vol.In(
                    available_models or [current_model]
                ),
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