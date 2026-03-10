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
mock_conversation.ChatMessage = MagicMock
mock_conversation.AssistantContent = MagicMock
mock_conversation.IntentResponseType = MagicMock

mock_entity_platform = MagicMock()
mock_entity_platform.AddConfigEntryEntitiesCallback = MagicMock

sys.modules["homeassistant.components.conversation"] = mock_conversation
sys.modules["homeassistant.helpers.entity_platform"] = mock_entity_platform


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
