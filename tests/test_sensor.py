"""Tests for Hailo Ollama sensor entities."""

from unittest.mock import AsyncMock, MagicMock, patch

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
    HailoResponseTimeSensor,
    HailoTokenCountSensor,
    HailoTokensPerSecondSensor,
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
async def test_async_setup_entry_adds_three_sensors(mock_config_entry):
    """async_setup_entry registers all three metric sensors."""
    hass = MagicMock()
    add_entities = MagicMock()

    await async_setup_entry(hass, mock_config_entry, add_entities)

    add_entities.assert_called_once()
    entities = add_entities.call_args[0][0]
    assert len(entities) == 3
    types = {type(e) for e in entities}
    assert HailoResponseTimeSensor in types
    assert HailoTokensPerSecondSensor in types
    assert HailoTokenCountSensor in types


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_response_time_sensor_unique_id(mock_config_entry):
    sensor = HailoResponseTimeSensor(mock_config_entry)
    assert sensor._attr_unique_id == "test_entry_id_response_time"


def test_tokens_per_second_sensor_unique_id(mock_config_entry):
    sensor = HailoTokensPerSecondSensor(mock_config_entry)
    assert sensor._attr_unique_id == "test_entry_id_tokens_per_second"


def test_token_count_sensor_unique_id(mock_config_entry):
    sensor = HailoTokenCountSensor(mock_config_entry)
    assert sensor._attr_unique_id == "test_entry_id_token_count"


def test_sensors_start_with_no_value(mock_config_entry):
    """Sensors report None until the first metrics update."""
    assert HailoResponseTimeSensor(mock_config_entry)._attr_native_value is None
    assert HailoTokensPerSecondSensor(mock_config_entry)._attr_native_value is None
    assert HailoTokenCountSensor(mock_config_entry)._attr_native_value is None


# ---------------------------------------------------------------------------
# HailoResponseTimeSensor metric computation
# ---------------------------------------------------------------------------


def test_response_time_computed_from_total_duration(mock_config_entry):
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor._update_from_metrics({"total_duration": 5_000_000_000})
    assert sensor._attr_native_value == 5.0


def test_response_time_rounds_to_two_decimals(mock_config_entry):
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor._update_from_metrics({"total_duration": 1_234_567_890})
    assert sensor._attr_native_value == 1.23


def test_response_time_zero_total_duration_gives_none(mock_config_entry):
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor._update_from_metrics({"total_duration": 0})
    assert sensor._attr_native_value is None


def test_response_time_missing_key_gives_none(mock_config_entry):
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor._update_from_metrics({})
    assert sensor._attr_native_value is None


# ---------------------------------------------------------------------------
# HailoTokensPerSecondSensor metric computation
# ---------------------------------------------------------------------------


def test_tokens_per_second_computed_correctly(mock_config_entry):
    sensor = HailoTokensPerSecondSensor(mock_config_entry)
    sensor._update_from_metrics({"eval_count": 100, "eval_duration": 2_000_000_000})
    assert sensor._attr_native_value == 50.0


def test_tokens_per_second_rounds_to_one_decimal(mock_config_entry):
    sensor = HailoTokensPerSecondSensor(mock_config_entry)
    sensor._update_from_metrics({"eval_count": 113, "eval_duration": 1_206_390_000})
    assert sensor._attr_native_value == round(113 / (1_206_390_000 / 1e9), 1)


def test_tokens_per_second_zero_eval_count_gives_none(mock_config_entry):
    sensor = HailoTokensPerSecondSensor(mock_config_entry)
    sensor._update_from_metrics({"eval_count": 0, "eval_duration": 1_000_000_000})
    assert sensor._attr_native_value is None


def test_tokens_per_second_zero_eval_duration_gives_none(mock_config_entry):
    """Guard against division by zero when eval_duration is 0."""
    sensor = HailoTokensPerSecondSensor(mock_config_entry)
    sensor._update_from_metrics({"eval_count": 50, "eval_duration": 0})
    assert sensor._attr_native_value is None


def test_tokens_per_second_missing_keys_give_none(mock_config_entry):
    sensor = HailoTokensPerSecondSensor(mock_config_entry)
    sensor._update_from_metrics({})
    assert sensor._attr_native_value is None


# ---------------------------------------------------------------------------
# HailoTokenCountSensor metric computation
# ---------------------------------------------------------------------------


def test_token_count_from_eval_count(mock_config_entry):
    sensor = HailoTokenCountSensor(mock_config_entry)
    sensor._update_from_metrics({"eval_count": 113})
    assert sensor._attr_native_value == 113


def test_token_count_zero_gives_none(mock_config_entry):
    sensor = HailoTokenCountSensor(mock_config_entry)
    sensor._update_from_metrics({"eval_count": 0})
    assert sensor._attr_native_value is None


def test_token_count_missing_key_gives_none(mock_config_entry):
    sensor = HailoTokenCountSensor(mock_config_entry)
    sensor._update_from_metrics({})
    assert sensor._attr_native_value is None


# ---------------------------------------------------------------------------
# Dispatcher subscription
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_added_to_hass_subscribes_to_signal(mock_config_entry):
    """async_added_to_hass connects to the metrics dispatcher signal."""
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor.hass = MagicMock()
    sensor.async_on_remove = MagicMock()

    disconnect = MagicMock()
    with patch(
        "custom_components.hailo_ollama.sensor.async_dispatcher_connect",
        return_value=disconnect,
    ) as mock_connect:
        await sensor.async_added_to_hass()

        mock_connect.assert_called_once_with(
            sensor.hass,
            SIGNAL_METRICS_UPDATED.format(mock_config_entry.entry_id),
            sensor._handle_metrics,
        )
        sensor.async_on_remove.assert_called_once_with(disconnect)


def test_handle_metrics_updates_state_and_writes(mock_config_entry):
    """_handle_metrics updates value and calls async_write_ha_state."""
    sensor = HailoResponseTimeSensor(mock_config_entry)
    sensor.async_write_ha_state = MagicMock()

    sensor._handle_metrics({"total_duration": 3_000_000_000, "eval_count": 50, "eval_duration": 1_000_000_000})

    assert sensor._attr_native_value == 3.0
    sensor.async_write_ha_state.assert_called_once()
