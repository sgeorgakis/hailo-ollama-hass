"""Hailo Ollama integration for Home Assistant."""

from __future__ import annotations

import logging
from datetime import timedelta

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    CONF_HOST,
    CONF_PORT,
    DOMAIN,
    HEALTH_CHECK_INTERVAL,
    SIGNAL_AVAILABILITY_CHANGED,
)
from .services import async_register_services, async_unregister_services

PLATFORMS = [Platform.CONVERSATION, Platform.AI_TASK, Platform.SENSOR]

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Hailo Ollama from a config entry."""
    _LOGGER.info("Setting up Hailo Ollama: %s", entry.title)

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {"available": True}

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    async_register_services(hass)

    async def _health_check(_now=None) -> None:
        host = entry.data[CONF_HOST]
        port = entry.data[CONF_PORT]
        session = async_get_clientsession(hass)
        was_available = hass.data[DOMAIN][entry.entry_id]["available"]
        try:
            async with session.get(
                f"http://{host}:{port}/api/version",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                available = resp.status == 200
        except Exception:
            available = False

        hass.data[DOMAIN][entry.entry_id]["available"] = available
        if available != was_available:
            _LOGGER.info(
                "Hailo Ollama %s: server is now %s",
                entry.title,
                "available" if available else "unavailable",
            )
            async_dispatcher_send(
                hass,
                SIGNAL_AVAILABILITY_CHANGED.format(entry.entry_id),
                available,
            )

    entry.async_on_unload(
        async_track_time_interval(
            hass, _health_check, timedelta(seconds=HEALTH_CHECK_INTERVAL)
        )
    )

    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry when options are updated."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Hailo Ollama config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        hass.data[DOMAIN].pop(entry.entry_id, None)
        async_unregister_services(hass)
    return unloaded
