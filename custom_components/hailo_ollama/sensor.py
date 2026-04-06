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
        HailoResponseCharsSensor(entry),
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
    """Wall-clock response time for the last conversation turn."""

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
        rt = metrics.get("response_time")
        self._attr_native_value = rt if rt else None


class HailoResponseCharsSensor(_HailoMetricSensor):
    """Number of characters in the last response."""

    _attr_translation_key = "response_chars"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "chars"
    _attr_native_value: int | None = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize."""
        super().__init__(entry)
        self._attr_unique_id = f"{entry.entry_id}_response_chars"

    def _update_from_metrics(self, metrics: dict) -> None:
        chars = metrics.get("response_chars")
        self._attr_native_value = chars if chars else None
