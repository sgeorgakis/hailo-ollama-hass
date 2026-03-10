# Hailo Ollama for Home Assistant

[![Tests](https://github.com/sgeorgakis/hailo-ollama-hass/actions/workflows/tests.yml/badge.svg)](https://github.com/sgeorgakis/hailo-ollama-hass/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/sgeorgakis/hailo-ollama-hass/branch/main/graph/badge.svg)](https://codecov.io/gh/sgeorgakis/hailo-ollama-hass)
[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A Home Assistant custom integration that connects to a [Hailo-Ollama](https://github.com/hailo-ai/hailo-ollama) server, enabling local LLM-powered conversation agents using Hailo AI accelerator hardware.

## Features

- Local LLM inference using Hailo AI accelerator
- Conversation agent integration for Home Assistant Assist
- Streaming and non-streaming response modes
- Multi-turn conversation support
- Configurable system prompts
- Optional display of model reasoning (`<think>` tag content)

## Requirements

- Home Assistant 2025.1 or newer
- A running [Hailo-Ollama](https://github.com/hailo-ai/hailo-ollama) server with at least one model downloaded

## Installation

### HACS (Recommended)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=sgeorgakis&repository=hailo-ollama-hass&category=integration)

Or manually add via HACS:

1. Open HACS in Home Assistant
2. Click the three dots in the top right corner
3. Select "Custom repositories"
4. Add `https://github.com/sgeorgakis/hailo-ollama-hass` with category "Integration"
5. Click "Download" on the Hailo Ollama card
6. Restart Home Assistant

### Manual Installation

1. Download or clone this repository
2. Copy the `custom_components/hailo_ollama` folder to your Home Assistant `config/custom_components/` directory
3. Restart Home Assistant

## Configuration

1. Go to **Settings** → **Devices & Services**
2. Click **Add Integration**
3. Search for "Hailo Ollama"
4. Enter your Hailo-Ollama server host and port (default: `localhost:8000`)
5. Select a model from the available models on your server
6. Optionally customize the system prompt and streaming mode

## Usage

After configuration, the integration creates a conversation agent that can be used with Home Assistant Assist:

1. Go to **Settings** → **Voice assistants**
2. Create or edit an assistant
3. Select "Hailo Ollama" as the conversation agent

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| Host | `localhost` | Hailo-Ollama server hostname or IP |
| Port | `8000` | Hailo-Ollama server port |
| Model | *(first available)* | LLM model to use |
| System Prompt | `You are a helpful smart home assistant. Be concise.` | System prompt for the LLM |
| Streaming | `true` | Enable streaming responses |

## Troubleshooting

### Cannot connect to Hailo-Ollama

- Verify the Hailo-Ollama server is running: `curl http://HOST:PORT/api/version`
- Check firewall settings if running on a different machine
- Ensure the port is correct (default: 8000)

### No models found

- Download a model on your Hailo-Ollama server first
- Verify models are available: `curl http://HOST:PORT/api/tags`

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0).

You are free to use, modify, fork, and share this software. Any derivative works must also be open source under the same license.
