"""Fixtures for Hailo Ollama tests."""

import sys
from unittest.mock import MagicMock

import pytest

# Older HA test packages may not include Platform.AI_TASK — add it so that
# __init__.py can reference it without raising AttributeError at import time.
from homeassistant.const import Platform
if not hasattr(Platform, "AI_TASK"):
    Platform.AI_TASK = "ai_task"  # type: ignore[attr-defined]


# Create a proper base class for ConversationEntity
class MockConversationEntity:
    """Mock base class for ConversationEntity."""
    _attr_has_entity_name = True
    _attr_name = None
    _attr_unique_id = None

    @property
    def hass(self):
        return getattr(self, "_hass", None)

    @hass.setter
    def hass(self, value):
        self._hass = value


# Mock homeassistant modules before they're imported.
# For each module, try importing the real HA package first so that tests run
# against the actual types when a compatible HA version is installed.  Fall
# back to lightweight mocks only when the real module is unavailable (e.g. the
# PyPI homeassistant package pre-dates the component).

try:
    from homeassistant.components.conversation import (  # noqa: F401
        ConversationEntity as _ConversationEntity,
        ConversationInput as _ConversationInput,
        ConversationResult as _ConversationResult,
    )
except ImportError:
    mock_conversation = MagicMock()
    mock_conversation.ConversationEntity = MockConversationEntity
    mock_conversation.ConversationInput = MagicMock
    mock_conversation.ConversationResult = MagicMock
    sys.modules["homeassistant.components.conversation"] = mock_conversation

try:
    from homeassistant.helpers.entity_platform import (  # noqa: F401
        AddConfigEntryEntitiesCallback as _AddConfigEntryEntitiesCallback,
    )
except ImportError:
    mock_entity_platform = MagicMock()
    mock_entity_platform.AddConfigEntryEntitiesCallback = MagicMock
    sys.modules["homeassistant.helpers.entity_platform"] = mock_entity_platform


class MockAITaskEntity:
    """Mock base class for AITaskEntity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_unique_id = None
    _attr_supported_tasks = MagicMock()

    @property
    def hass(self):
        return getattr(self, "_hass", None)

    @hass.setter
    def hass(self, value):
        self._hass = value


try:
    # Verify the real ai_task module exports the names we depend on.
    # If this import succeeds the tests will exercise the real HA types and
    # catch any future renames early.
    from homeassistant.components.ai_task import (  # noqa: F401
        AITaskEntity as _AITaskEntity,
        AITaskEntityFeature as _AITaskEntityFeature,
        GenDataTask as _GenDataTask,
        GenDataTaskResult as _GenDataTaskResult,
    )
except ImportError:
    mock_ai_task = MagicMock()
    mock_ai_task.AITaskEntity = MockAITaskEntity
    mock_ai_task.AITaskEntityFeature = MagicMock()
    mock_ai_task.AITaskEntityFeature.GENERATE_DATA = MagicMock()
    mock_ai_task.GenDataTask = MagicMock
    mock_ai_task.GenDataTaskResult = MagicMock
    sys.modules["homeassistant.components.ai_task"] = mock_ai_task


@pytest.fixture
def mock_chat_response():
    """Mock successful /api/chat response."""
    return {
        "model": "llama3.2:3b",
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
        },
        "done": True,
    }


@pytest.fixture
def mock_streaming_chunks():
    """Mock streaming /api/chat response chunks."""
    return [
        b'{"model":"llama3.2:3b","message":{"role":"assistant","content":"Hello"},"done":false}\n',
        b'{"model":"llama3.2:3b","message":{"role":"assistant","content":"!"},"done":false}\n',
        b'{"model":"llama3.2:3b","message":{"role":"assistant","content":" How can I help?"},"done":true}\n',
    ]
