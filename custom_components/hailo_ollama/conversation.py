"""Conversation agent for Hailo Ollama."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.conversation import ChatLog
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
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
)

_LOGGER = logging.getLogger(__name__)

# Some models wrap reasoning in <think>...</think>
THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation agent."""
    async_add_entities([HailoOllamaConversationEntity(entry)])


class HailoOllamaConversationEntity(conversation.ConversationEntity):
    """Hailo Ollama conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize."""
        self._entry = entry
        self._host: str = entry.data[CONF_HOST]
        self._port: int = entry.data[CONF_PORT]
        self._model: str = entry.data[CONF_MODEL]
        self._system_prompt: str = entry.data.get(
            CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
        )
        self._streaming: bool = entry.data.get(CONF_STREAMING, DEFAULT_STREAMING)
        self._show_thinking: bool = entry.data.get(
            CONF_SHOW_THINKING, DEFAULT_SHOW_THINKING
        )
        self._attr_unique_id = entry.entry_id
        self._base_url = f"http://{self._host}:{self._port}"

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return ["en"]

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device info."""
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": f"Hailo Ollama ({self._model})",
            "manufacturer": "Hailo",
            "model": self._model,
        }

    async def async_process(
        self,
        user_input: conversation.ConversationInput,
        chat_log: ChatLog,
    ) -> conversation.ConversationResult:
        """Process a conversation turn."""
        t0 = time.monotonic()
        user_text = user_input.text
        _LOGGER.debug("User: %s", user_text)

        # Build messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
        ]

        for entry in chat_log.content:
            if isinstance(entry, conversation.ChatMessage):
                if entry.role in ("user", "assistant"):
                    messages.append({
                        "role": entry.role,
                        "content": entry.content,
                    })

        messages.append({"role": "user", "content": user_text})

        # Call Hailo with configured mode
        try:
            if self._streaming:
                response_text = await self._call_streaming(messages)
            else:
                response_text = await self._call_non_streaming(messages)
        except HailoError as err:
            _LOGGER.error("Hailo error: %s", err)
            response_text = f"Sorry, I encountered an error: {err}"

        elapsed = time.monotonic() - t0

        # Conditionally strip <think> tags
        if self._show_thinking:
            clean_text = response_text.strip()
        else:
            clean_text = THINK_TAG_RE.sub("", response_text).strip()
            if clean_text != response_text:
                _LOGGER.debug(
                    "Stripped <think> tags: %d → %d chars",
                    len(response_text),
                    len(clean_text),
                )

        _LOGGER.info(
            "Hailo responded in %.1fs (%d chars): %.100s",
            elapsed,
            len(clean_text),
            clean_text,
        )

        chat_log.async_add_assistant_content_from_result(
            conversation.AssistantContent(
                agent_id=user_input.agent_id,
                content=clean_text,
            )
        )

        return conversation.ConversationResult(
            response=conversation.IntentResponseType(clean_text),
            chat_log=chat_log,
        )

    def _build_payload(
        self, messages: list[dict[str, str]], stream: bool
    ) -> dict:
        """Build the minimal /api/chat payload."""
        return {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }

    async def _call_non_streaming(
        self, messages: list[dict[str, str]]
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
        self, messages: list[dict[str, str]]
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

        # Last done=true chunk has full content
        last = chunks[-1]
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


class HailoError(Exception):
    """Error from Hailo Ollama."""