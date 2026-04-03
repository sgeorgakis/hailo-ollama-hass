"""AI task entity for Hailo Ollama."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.ai_task import (
    AITaskEntity,
    AITaskEntityFeature,
    GenDataTask,
    GenDataTaskResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_HOST,
    CONF_MODEL,
    CONF_PORT,
    CONF_SHOW_THINKING,
    CONF_STREAMING,
    CONF_SYSTEM_PROMPT,
    CONF_VISION_MODEL,
    DEFAULT_SHOW_THINKING,
    DEFAULT_STREAMING,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VISION_MODEL,
    DOMAIN,
)
from .conversation import HailoError, HailoOllamaClientMixin, _process_thinking

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the AI task entity."""
    async_add_entities([HailoAITaskEntity(entry)])


class HailoAITaskEntity(AITaskEntity, HailoOllamaClientMixin):
    """Hailo Ollama AI task entity for data generation tasks."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = AITaskEntityFeature.GENERATE_DATA

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize."""
        self._entry = entry
        self._host: str = entry.data[CONF_HOST]
        self._port: int = entry.data[CONF_PORT]
        opts = entry.options or {}
        self._model: str = opts.get(CONF_MODEL) or entry.data[CONF_MODEL]
        self._system_prompt: str = opts.get(CONF_SYSTEM_PROMPT) or entry.data.get(
            CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
        )
        self._streaming: bool = opts.get(
            CONF_STREAMING, entry.data.get(CONF_STREAMING, DEFAULT_STREAMING)
        )
        self._show_thinking: bool = opts.get(
            CONF_SHOW_THINKING, entry.data.get(CONF_SHOW_THINKING, DEFAULT_SHOW_THINKING)
        )
        self._vision_model: bool = opts.get(
            CONF_VISION_MODEL, entry.data.get(CONF_VISION_MODEL, DEFAULT_VISION_MODEL)
        )
        self._attr_unique_id = f"{entry.entry_id}_ai_task"
        self._base_url = f"http://{self._host}:{self._port}"

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device info."""
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": f"Hailo Ollama ({self._model})",
            "manufacturer": "Hailo",
            "model": self._model,
        }

    async def _async_generate_data(
        self,
        task: GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> GenDataTaskResult:
        """Execute a data generation task against the Hailo-Ollama API."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task.instructions},
        ]

        try:
            if self._streaming:
                response_text = await self._call_streaming(messages)
            else:
                response_text = await self._call_non_streaming(messages)
        except HailoError as err:
            _LOGGER.error("Hailo AI task error: %s", err)
            raise

        clean_text = _process_thinking(response_text, self._show_thinking)
        return GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=clean_text,
        )
