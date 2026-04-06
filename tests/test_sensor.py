"""Tests for Hailo Ollama sensor entities."""

from unittest.mock import MagicMock, patch

import pytest

from custom_components.hailo_ollama.const import (
    CONF_HOST,
    CONF_MODEL,
    CONF_PORT,
    CONF_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    SIGNAL_METRICS_UPDATED,
)
from custom_components.hailo_ollama.sensor import (
    HailoResponseCharsSensor,
    HailoResponseTimeSensor,
    async_setup_entry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry for sensor tests."""
    entry = MagicMock()
    entry.entry_id = "test_entry_id"
    entry.data = {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
    }
    entry.options = {}
    return entry


# ---------------------------------------------------------------------------
# async_setup_entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_setup_entry_adds_two_sensors(mock_config_entry):
    """async_setup_entry registers both metric sensors."""
    hass = MagicMock()
    add_entities = MagicMock()

    await async_setup_entry(hass, mock_config_entry, add_entities)

    add_entities.assert_called_once()
    entities = add_entities.call_args[0][0]
    assert len(entities) == 2
    types = {type(e) for e in entities}
    assert HailoResponseTimeSensor in types
    assert HailoResponseCharsSensor in types


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_response_time_sensor_unique_id(mock_config_entry):
    sensor = HailoResponseTimeSensor(mock_config_entry)
    assert sensor._attr_unique_id == "test_entry_id_response_time"


def test_response_chars_sensor_unique_id(mock_config_entry):
    sensor = HailoResponseCharsSensor(mock_config_entry)
    assert sensor._attr_unique_id == "test_entry_id_response_chars"


def test_sensors_start_with_no_value(mock_config_entry):
    """Sensors report None until the first metrics update."""
    assert HailoResponseTimeSensor(mock_config_entry)._attr_native_value is None
    assert HailoResponseCharsSensor(mock_config_entry)._attr_native_value is None


# ---------------------------------------------------------------------------
# HailoResponseTimeSensor metric computation
# ---------------------------------------------------------------------------


def test_response_time_set_from_response_time(mock_config_entry):
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor._update_from_metrics({"response_time": 5.0})
    assert sensor._attr_native_value == 5.0


def test_response_time_preserves_decimals(mock_config_entry):
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor._update_from_metrics({"response_time": 1.23})
    assert sensor._attr_native_value == 1.23


def test_response_time_zero_gives_none(mock_config_entry):
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor._update_from_metrics({"response_time": 0})
    assert sensor._attr_native_value is None


def test_response_time_missing_key_gives_none(mock_config_entry):
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor._update_from_metrics({})
    assert sensor._attr_native_value is None


# ---------------------------------------------------------------------------
# HailoResponseCharsSensor metric computation
# ---------------------------------------------------------------------------


def test_response_chars_set_from_response_chars(mock_config_entry):
    sensor = HailoResponseCharsSensor(mock_config_entry)
    sensor._update_from_metrics({"response_chars": 256})
    assert sensor._attr_native_value == 256


def test_response_chars_zero_gives_none(mock_config_entry):
    sensor = HailoResponseCharsSensor(mock_config_entry)
    sensor._update_from_metrics({"response_chars": 0})
    assert sensor._attr_native_value is None


def test_response_chars_missing_key_gives_none(mock_config_entry):
    sensor = HailoResponseCharsSensor(mock_config_entry)
    sensor._update_from_metrics({})
    assert sensor._attr_native_value is None


# ---------------------------------------------------------------------------
# Dispatcher subscription
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_added_to_hass_subscribes_to_signals(mock_config_entry):
    """async_added_to_hass connects to both dispatcher signals."""
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor.hass = MagicMock()
    sensor.async_on_remove = MagicMock()

    disconnect = MagicMock()
    with patch(
        "custom_components.hailo_ollama.sensor.async_dispatcher_connect",
        return_value=disconnect,
    ) as mock_connect:
        await sensor.async_added_to_hass()

        assert mock_connect.call_count == 2
        calls = [call.args[1] for call in mock_connect.call_args_list]
        assert SIGNAL_METRICS_UPDATED.format(mock_config_entry.entry_id) in calls
        assert sensor.async_on_remove.call_count == 2


def test_handle_metrics_updates_state_and_writes(mock_config_entry):
    """_handle_metrics updates value and calls async_write_ha_state."""
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor.async_write_ha_state = MagicMock()

    sensor._handle_metrics({"response_time": 3.0, "response_chars": 150})

    assert sensor._attr_native_value == 3.0
    sensor.async_write_ha_state.assert_called_once()
