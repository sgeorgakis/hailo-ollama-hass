"""Tests for Hailo Ollama __init__.py entry point."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hailo_ollama import PLATFORMS, async_setup_entry, async_unload_entry


@pytest.mark.asyncio
async def test_async_setup_entry():
    """Test async_setup_entry forwards to PLATFORMS and returns True."""
    hass = MagicMock()
    hass.config_entries.async_forward_entry_setups = AsyncMock(return_value=None)
    entry = MagicMock()
    entry.title = "Test Hailo"

    result = await async_setup_entry(hass, entry)

    assert result is True
    hass.config_entries.async_forward_entry_setups.assert_called_once_with(entry, PLATFORMS)


@pytest.mark.asyncio
async def test_async_unload_entry():
    """Test async_unload_entry delegates to async_unload_platforms and returns its result."""
    hass = MagicMock()
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
    entry = MagicMock()

    result = await async_unload_entry(hass, entry)

    assert result is True
    hass.config_entries.async_unload_platforms.assert_called_once_with(entry, PLATFORMS)


@pytest.mark.asyncio
async def test_async_unload_entry_returns_false_when_platform_unload_fails():
    """Test async_unload_entry propagates False from async_unload_platforms."""
    hass = MagicMock()
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=False)
    entry = MagicMock()

    result = await async_unload_entry(hass, entry)

    assert result is False
