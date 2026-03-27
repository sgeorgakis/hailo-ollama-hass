"""Fixtures for Hailo Ollama tests."""

import sys
from unittest.mock import MagicMock

import pytest


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


# Mock homeassistant modules before they're imported
mock_conversation = MagicMock()
mock_conversation.ConversationEntity = MockConversationEntity
mock_conversation.ConversationInput = MagicMock
mock_conversation.ConversationResult = MagicMock

mock_entity_platform = MagicMock()
mock_entity_platform.AddConfigEntryEntitiesCallback = MagicMock

sys.modules["homeassistant.components.conversation"] = mock_conversation
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


mock_ai_task = MagicMock()
mock_ai_task.AITaskEntity = MockAITaskEntity
mock_ai_task.AITaskEntityFeature = MagicMock()
mock_ai_task.AITaskEntityFeature.GENERATE_DATA = MagicMock()
mock_ai_task.GenDataTask = MagicMock
mock_ai_task.RunDataTaskResult = MagicMock

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
