# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Home Assistant custom integration (`hailo_ollama`) that connects to a Hailo-Ollama server—a local LLM service running on Hailo AI accelerator hardware. The integration exposes the LLM as a Home Assistant conversation agent.

## Architecture

The integration follows the standard Home Assistant custom component structure:

- **`__init__.py`** - Entry point; forwards setup to the conversation platform
- **`config_flow.py`** - Two-step UI config flow: (1) host/port connection, (2) model selection with system prompt and streaming options
- **`conversation.py`** - The conversation entity that implements `ConversationEntity`; handles both streaming and non-streaming API calls to `/api/chat`
- **`const.py`** - Configuration keys and defaults (host: localhost, port: 8000)
- **`manifest.json`** - Integration metadata; declares dependency on the `conversation` component

## API Endpoints Used

The integration communicates with the Hailo-Ollama server via:
- `GET /api/version` - Connection test during setup
- `GET /api/tags` - Fetch available models (with fallback to `/hailo/v1/list`)
- `POST /api/chat` - Send conversation messages (supports both `stream: true` and `stream: false`)

## Key Implementation Details

- **Streaming vs non-streaming**: Configurable per-integration instance. Streaming collects ndjson chunks; non-streaming expects a single JSON response. Non-streaming automatically falls back to streaming on `ClientPayloadError`.
- **Reasoning tag support**: Response text can optionally strip `<think>...</think>` tags (configurable per integration instance).
- **Chat history**: The conversation entity reads from Home Assistant's `ChatLog` to maintain multi-turn context.
- **Timeout**: 500 seconds default (`DEFAULT_TIMEOUT`) for LLM responses.

## Development

Install the integration by copying/symlinking this directory to `config/custom_components/hailo_ollama/` in your Home Assistant installation.

No build step required. The integration uses Home Assistant's bundled dependencies (`aiohttp`, `voluptuous`).

## Workflow
- Create a new branch for a new feature
- Always run tests for a change
- Always write tests for a change
- Run single tests instead of all of them for performance
- Make sure that the project compiles
- Make sure that all tests pass

## Workflow
- Create a new branch for a new feature
- Always run tests for a change
- Always write tests for a change
- Run single tests instead of all of them for performance
- Make sure that the project compiles
- Make sure that all tests pass

