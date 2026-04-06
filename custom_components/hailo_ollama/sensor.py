"""Sensor entities for Hailo Ollama response metrics."""

from __future__ import annotations

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN, SIGNAL_AVAILABILITY_CHANGED, SIGNAL_METRICS_UPDATED


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Hailo Ollama sensor entities."""
    async_add_entities([
        HailoResponseTimeSensor(entry),
        HailoTokensPerSecondSensor(entry),
        HailoTokenCountSensor(entry),
    ])


class _HailoMetricSensor(SensorEntity):
    """Base class for Hailo response metric sensors."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize."""
        self._entry = entry

    @property
    def device_info(self) -> dict:
        """Return device info."""
        return {"identifiers": {(DOMAIN, self._entry.entry_id)}}

    async def async_added_to_hass(self) -> None:
        """Subscribe to metrics and availability updates."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_METRICS_UPDATED.format(self._entry.entry_id),
                self._handle_metrics,
            )
        )
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

    def _handle_metrics(self, metrics: dict) -> None:
        """Update state from metrics dict and push to HA."""
        self._update_from_metrics(metrics)
        self.async_write_ha_state()

    def _update_from_metrics(self, metrics: dict) -> None:
        """Extract the relevant metric value from the metrics dict."""
        raise NotImplementedError


class HailoResponseTimeSensor(_HailoMetricSensor):
    """Total response time for the last conversation turn."""

    _attr_translation_key = "response_time"
    _attr_device_class = SensorDeviceClass.DURATION
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTime.SECONDS
    _attr_suggested_display_precision = 2
    _attr_native_value: float | None = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize."""
        super().__init__(entry)
        self._attr_unique_id = f"{entry.entry_id}_response_time"

    def _update_from_metrics(self, metrics: dict) -> None:
        total_ns = metrics.get("total_duration", 0)
        self._attr_native_value = round(total_ns / 1e9, 2) if total_ns else None


class HailoTokensPerSecondSensor(_HailoMetricSensor):
    """Token generation speed for the last conversation turn."""

    _attr_translation_key = "tokens_per_second"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "tok/s"
    _attr_suggested_display_precision = 1
    _attr_native_value: float | None = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize."""
        super().__init__(entry)
        self._attr_unique_id = f"{entry.entry_id}_tokens_per_second"

    def _update_from_metrics(self, metrics: dict) -> None:
        eval_count = metrics.get("eval_count", 0)
        eval_duration_ns = metrics.get("eval_duration", 0)
        if eval_count and eval_duration_ns:
            self._attr_native_value = round(eval_count / (eval_duration_ns / 1e9), 1)
        else:
            self._attr_native_value = None


class HailoTokenCountSensor(_HailoMetricSensor):
    """Number of tokens generated in the last conversation turn."""

    _attr_translation_key = "token_count"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "tokens"
    _attr_native_value: int | None = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize."""
        super().__init__(entry)
        self._attr_unique_id = f"{entry.entry_id}_token_count"

    def _update_from_metrics(self, metrics: dict) -> None:
        eval_count = metrics.get("eval_count", 0)
        self._attr_native_value = eval_count if eval_count else None
