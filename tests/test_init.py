"""Tests for Hailo Ollama __init__.py entry point."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hailo_ollama import (
    PLATFORMS,
    _async_update_listener,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.hailo_ollama.const import (
    CONF_HOST,
    CONF_PORT,
    DOMAIN,
    SIGNAL_AVAILABILITY_CHANGED,
)


def _make_hass(entry_id="test_entry_id"):
    hass = MagicMock()
    hass.config_entries.async_forward_entry_setups = AsyncMock(return_value=None)
    hass.data = {}
    return hass


def _make_entry(entry_id="test_entry_id"):
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.title = "Test Hailo"
    entry.data = {CONF_HOST: "localhost", CONF_PORT: 8000}
    return entry


@pytest.mark.asyncio
async def test_async_setup_entry_returns_true():
    """async_setup_entry forwards platforms and returns True."""
    hass = _make_hass()
    entry = _make_entry()

    with patch("custom_components.hailo_ollama.async_track_time_interval"):
        result = await async_setup_entry(hass, entry)

    assert result is True
    hass.config_entries.async_forward_entry_setups.assert_called_once_with(entry, PLATFORMS)


@pytest.mark.asyncio
async def test_async_setup_entry_initialises_domain_data():
    """async_setup_entry stores availability state in hass.data."""
    hass = _make_hass()
    entry = _make_entry()

    with patch("custom_components.hailo_ollama.async_track_time_interval"):
        await async_setup_entry(hass, entry)

    assert hass.data[DOMAIN][entry.entry_id]["available"] is True


@pytest.mark.asyncio
async def test_async_setup_entry_registers_update_listener():
    hass = _make_hass()
    entry = _make_entry()

    with patch("custom_components.hailo_ollama.async_track_time_interval"):
        await async_setup_entry(hass, entry)

    entry.add_update_listener.assert_called_once_with(_async_update_listener)


@pytest.mark.asyncio
async def test_async_setup_entry_registers_health_check():
    """async_setup_entry schedules the periodic health check."""
    hass = _make_hass()
    entry = _make_entry()

    with patch(
        "custom_components.hailo_ollama.async_track_time_interval"
    ) as mock_interval:
        await async_setup_entry(hass, entry)

    mock_interval.assert_called_once()


@pytest.mark.asyncio
async def test_health_check_marks_unavailable_on_connection_error():
    """Health check sets available=False when server is unreachable."""
    hass = _make_hass()
    entry = _make_entry()

    captured_callback = None

    def capture_interval(h, callback, interval):
        nonlocal captured_callback
        captured_callback = callback
        return MagicMock()

    with patch("custom_components.hailo_ollama.async_track_time_interval", side_effect=capture_interval):
        with patch("custom_components.hailo_ollama.async_get_clientsession") as mock_session:
            session = MagicMock()
            mock_session.return_value = session
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
            ctx.__aexit__ = AsyncMock(return_value=False)
            session.get.return_value = ctx

            await async_setup_entry(hass, entry)
            await captured_callback()

    assert hass.data[DOMAIN][entry.entry_id]["available"] is False


@pytest.mark.asyncio
async def test_health_check_dispatches_signal_on_state_change():
    """Health check fires SIGNAL_AVAILABILITY_CHANGED when availability changes."""
    hass = _make_hass()
    entry = _make_entry()
    fired_signals = []

    def capture_interval(h, callback, interval):
        return MagicMock(), callback

    captured_callback = None

    def capture_interval(h, callback, interval):
        nonlocal captured_callback
        captured_callback = callback
        return MagicMock()

    with patch("custom_components.hailo_ollama.async_track_time_interval", side_effect=capture_interval):
        with patch("custom_components.hailo_ollama.async_get_clientsession") as mock_session:
            with patch("custom_components.hailo_ollama.async_dispatcher_send") as mock_send:
                session = MagicMock()
                mock_session.return_value = session
                ctx = MagicMock()
                ctx.__aenter__ = AsyncMock(side_effect=Exception("down"))
                ctx.__aexit__ = AsyncMock(return_value=False)
                session.get.return_value = ctx

                await async_setup_entry(hass, entry)
                await captured_callback()

                mock_send.assert_called_once_with(
                    hass,
                    SIGNAL_AVAILABILITY_CHANGED.format(entry.entry_id),
                    False,
                )


@pytest.mark.asyncio
async def test_health_check_no_signal_when_state_unchanged():
    """Health check does not fire a signal when availability stays the same."""
    hass = _make_hass()
    entry = _make_entry()
    captured_callback = None

    def capture_interval(h, callback, interval):
        nonlocal captured_callback
        captured_callback = callback
        return MagicMock()

    with patch("custom_components.hailo_ollama.async_track_time_interval", side_effect=capture_interval):
        with patch("custom_components.hailo_ollama.async_get_clientsession") as mock_session:
            with patch("custom_components.hailo_ollama.async_dispatcher_send") as mock_send:
                session = MagicMock()
                mock_session.return_value = session
                resp = MagicMock()
                resp.status = 200
                ctx = MagicMock()
                ctx.__aenter__ = AsyncMock(return_value=resp)
                ctx.__aexit__ = AsyncMock(return_value=False)
                session.get.return_value = ctx

                await async_setup_entry(hass, entry)
                await captured_callback()  # still available → no signal

                mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_async_update_listener_reloads_entry():
    """_async_update_listener triggers a config entry reload."""
    hass = MagicMock()
    hass.config_entries.async_reload = AsyncMock()
    entry = MagicMock()
    entry.entry_id = "test_entry_id"

    await _async_update_listener(hass, entry)

    hass.config_entries.async_reload.assert_called_once_with(entry.entry_id)


@pytest.mark.asyncio
async def test_async_unload_entry_succeeds():
    """async_unload_entry delegates to async_unload_platforms and cleans up data."""
    hass = _make_hass()
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
    entry = _make_entry()
    hass.data[DOMAIN] = {entry.entry_id: {"available": True}}

    result = await async_unload_entry(hass, entry)

    assert result is True
    hass.config_entries.async_unload_platforms.assert_called_once_with(entry, PLATFORMS)
    assert entry.entry_id not in hass.data[DOMAIN]


@pytest.mark.asyncio
async def test_async_unload_entry_returns_false_when_platform_unload_fails():
    """async_unload_entry propagates False and does not clean up data."""
    hass = _make_hass()
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=False)
    entry = _make_entry()
    hass.data[DOMAIN] = {entry.entry_id: {"available": True}}

    result = await async_unload_entry(hass, entry)

    assert result is False
    assert entry.entry_id in hass.data[DOMAIN]
