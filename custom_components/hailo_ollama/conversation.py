"""Conversation agent for Hailo Ollama."""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Any

import aiohttp

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_HOST,
    CONF_MODEL,
    CONF_PORT,
    CONF_SHOW_THINKING,
    CONF_STREAMING,
    CONF_SYSTEM_PROMPT,
    DEFAULT_SHOW_THINKING,
    DEFAULT_STREAMING,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
    DOMAIN,
    SIGNAL_AVAILABILITY_CHANGED,
    SIGNAL_METRICS_UPDATED,
)

_LOGGER = logging.getLogger(__name__)


def _process_thinking(response_text: str, show_thinking: bool) -> str:
    """Strip or format <think>...</think> reasoning blocks.

    Handles both well-formed <think>...</think> and responses where the
    opening <think> tag is absent (some models omit it in streaming).
    When show_thinking is True the thinking content is wrapped in <i>
    tags so the UI can present it in italic style.
    """
    if "</think>" not in response_text:
        return response_text.strip()

    # Split on the first </think>; everything before is thinking content.
    think_part, _, answer_part = response_text.partition("</think>")

    # Strip the optional leading <think> tag.
    thinking = think_part.removeprefix("<think>").strip()
    answer = answer_part.strip()

    if show_thinking and thinking:
        return f"<i>{thinking}</i>\n\n{answer}"
    return answer


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation agent."""
    async_add_entities([HailoOllamaConversationEntity(entry)])


class HailoOllamaClientMixin:
    """Mixin that provides HTTP client methods for communicating with Hailo-Ollama.

    Subclasses must expose:
      - self._base_url: str
      - self._model: str
      - self._host: str
      - self._port: int
      - self.hass: HomeAssistant
    """

    def _build_payload(
        self, messages: list[dict[str, Any]], stream: bool
    ) -> dict:
        """Build the minimal /api/chat payload."""
        return {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }

    async def _call_non_streaming(
        self, messages: list[dict[str, Any]]
    ) -> str:
        """Call /api/chat with stream:false — single JSON response."""
        url = f"{self._base_url}/api/chat"
        payload = self._build_payload(messages, stream=False)

        _LOGGER.debug(
            "POST %s (non-streaming, %d messages, model=%s)",
            url,
            len(messages),
            self._model,
        )

        session = async_get_clientsession(self.hass)
        timeout = aiohttp.ClientTimeout(
            total=DEFAULT_TIMEOUT, sock_read=DEFAULT_TIMEOUT
        )

        try:
            async with session.post(
                url, json=payload, timeout=timeout
            ) as resp:
                _LOGGER.debug("Response: status=%s", resp.status)

                if resp.status != 200:
                    body = await resp.text()
                    raise HailoError(f"HTTP {resp.status}: {body[:300]}")

                data = await resp.json()

        except aiohttp.ClientPayloadError as err:
            _LOGGER.warning(
                "stream:false payload error (%s), retrying with streaming",
                err,
            )
            return await self._call_streaming(messages)

        except aiohttp.ClientConnectorError as err:
            raise HailoError(
                f"Cannot connect to {self._host}:{self._port}"
            ) from err

        except TimeoutError as err:
            raise HailoError(f"Timed out after {DEFAULT_TIMEOUT}s") from err

        content = data.get("message", {}).get("content", "")
        if not content:
            raise HailoError(
                f"No content in response. "
                f"Raw: {json.dumps(data)[:300]}"
            )

        return content

    async def _call_streaming(
        self, messages: list[dict[str, Any]]
    ) -> str:
        """Call /api/chat with stream:true — collect ndjson chunks."""
        url = f"{self._base_url}/api/chat"
        payload = self._build_payload(messages, stream=True)

        _LOGGER.debug(
            "POST %s (streaming, %d messages, model=%s)",
            url,
            len(messages),
            self._model,
        )

        session = async_get_clientsession(self.hass)
        timeout = aiohttp.ClientTimeout(
            total=DEFAULT_TIMEOUT, sock_read=DEFAULT_TIMEOUT
        )

        chunks: list[dict] = []
        buffer = b""

        try:
            async with session.post(
                url, json=payload, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise HailoError(f"HTTP {resp.status}: {body[:300]}")

                try:
                    async for data in resp.content.iter_any():
                        buffer += data
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                chunks.append(json.loads(line))
                            except json.JSONDecodeError:
                                _LOGGER.warning(
                                    "Bad chunk: %s", line[:100]
                                )

                    # Flush remaining buffer
                    if buffer.strip():
                        try:
                            chunks.append(json.loads(buffer.strip()))
                        except json.JSONDecodeError:
                            pass

                except aiohttp.ClientPayloadError:
                    _LOGGER.debug(
                        "ClientPayloadError (expected), got %d chunks",
                        len(chunks),
                    )

        except aiohttp.ClientConnectorError as err:
            raise HailoError(
                f"Cannot connect to {self._host}:{self._port}"
            ) from err

        except TimeoutError as err:
            raise HailoError(f"Timed out after {DEFAULT_TIMEOUT}s") from err

        _LOGGER.debug("Stream: %d chunks collected", len(chunks))

        if not chunks:
            raise HailoError("Streaming returned 0 chunks")

        last = chunks[-1]

        # Last done=true chunk has full content on some server versions
        if last.get("done") and last.get("message", {}).get("content"):
            return last["message"]["content"]

        # Concatenate all chunk contents
        full = "".join(
            c.get("message", {}).get("content", "") for c in chunks
        )

        if not full:
            raise HailoError(
                f"Got {len(chunks)} chunks but no content. "
                f"Last: {json.dumps(last)[:200]}"
            )

        return full


class HailoOllamaConversationEntity(
    conversation.ConversationEntity, HailoOllamaClientMixin
):
    """Hailo Ollama conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize."""
        self._entry = entry
        # Host and port are only set during initial config, never in options
        self._host: str = entry.data[CONF_HOST]
        self._port: int = entry.data[CONF_PORT]
        # Remaining settings may be overridden via the options flow
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
        self._attr_unique_id = entry.entry_id
        self._base_url = f"http://{self._host}:{self._port}"
        self._conversations: dict[str, list[dict[str, Any]]] = {}

    async def async_added_to_hass(self) -> None:
        """Subscribe to availability changes."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_AVAILABILITY_CHANGED.format(self._entry.entry_id),
                self._handle_availability,
            )
        )

    def _handle_availability(self, available: bool) -> None:
        self.async_write_ha_state()

    @property
    def available(self) -> bool:
        """Return True when the Hailo-Ollama server is reachable."""
        return (
            self.hass.data.get(DOMAIN, {})
            .get(self._entry.entry_id, {})
            .get("available", True)
        )

    @property
    def supported_languages(self) -> str:
        """Return supported languages."""
        return conversation.MATCH_ALL

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device info."""
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": f"Hailo Ollama ({self._model})",
            "manufacturer": "Hailo",
            "model": self._model,
        }

    def _build_user_message(
        self, text: str, attachments: list | None
    ) -> dict[str, Any]:
        """Build the user message dict, encoding image attachments when applicable."""
        if not attachments:
            return {"role": "user", "content": text}

        images: list[str] = []
        for attachment in attachments:
            raw = getattr(attachment, "content", None)
            if raw is None and isinstance(attachment, (bytes, bytearray)):
                raw = attachment
            if isinstance(raw, (bytes, bytearray)):
                images.append(base64.b64encode(raw).decode("ascii"))
            else:
                _LOGGER.warning(
                    "Skipping attachment with unreadable content: %s",
                    type(attachment),
                )

        if images:
            return {"role": "user", "content": text, "images": images}
        return {"role": "user", "content": text}

    async def async_process(
        self,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult:
        """Process a conversation turn."""
        t0 = time.monotonic()
        user_text = user_input.text
        _LOGGER.debug("User: %s", user_text)

        conversation_id = user_input.conversation_id or str(uuid.uuid4())
        history = self._conversations.get(conversation_id, [])

        attachments = getattr(user_input, "attachments", None)
        user_message = self._build_user_message(user_text, attachments)

        # Build messages: system prompt + history + new user message
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        messages.extend(history)
        messages.append(user_message)

        # Call Hailo with configured mode
        success = False
        try:
            if self._streaming:
                response_text = await self._call_streaming(messages)
            else:
                response_text = await self._call_non_streaming(messages)
            success = True
        except HailoError as err:
            _LOGGER.error("Hailo error: %s", err)
            response_text = f"Sorry, I encountered an error: {err}"

        elapsed = time.monotonic() - t0

        clean_text = _process_thinking(response_text, self._show_thinking)
        if success:
            async_dispatcher_send(
                self.hass,
                SIGNAL_METRICS_UPDATED.format(self._entry.entry_id),
                {"response_time": round(elapsed, 2), "response_chars": len(clean_text)},
            )
        if clean_text != response_text.strip():
            _LOGGER.debug(
                "Processed <think> tags: %d → %d chars",
                len(response_text),
                len(clean_text),
            )

        _LOGGER.info(
            "Hailo responded in %.1fs (%d chars): %.100s",
            elapsed,
            len(clean_text),
            clean_text,
        )

        # Store plain text in history regardless of how the message was sent
        updated_history = list(history)
        updated_history.append({"role": "user", "content": user_text})
        updated_history.append({"role": "assistant", "content": clean_text})
        self._conversations[conversation_id] = updated_history

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(clean_text)

        return conversation.ConversationResult(
            response=response,
            conversation_id=conversation_id,
        )


class HailoError(Exception):
    """Error from Hailo Ollama."""
