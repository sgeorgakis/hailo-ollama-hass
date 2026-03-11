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
- **Chat history**: The conversation entity maintains multi-turn context internally via a `_conversations` dict keyed by `conversation_id`.
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
- **Never manually bump `manifest.json` version** — versions are managed automatically by release-please

## Releases

Releases are fully automated via [release-please](https://github.com/googleapis/release-please) and GitHub Actions.

On every push to `main`, release-please inspects commits since the last release. When releasable commits exist, it opens (or updates) a **Release PR** that:
- Bumps the version in `manifest.json` according to SemVer
- Generates a changelog from commit messages

Merging the Release PR triggers a second workflow that creates the GitHub Release and attaches `hailo_ollama.zip` for HACS users.

### Commit message format (Conventional Commits)

All commits **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<optional scope>): <short description>

<optional body>
```

**Types and their effect on versioning:**

| Type | Description | Version bump |
|------|-------------|--------------|
| `fix` | Bug fix | patch (0.0.x) |
| `feat` | New feature | minor (0.x.0) |
| `feat!` or `fix!` or any `!` | Breaking change | major (x.0.0) |
| `chore` | Maintenance, dependency updates | none |
| `docs` | Documentation only | none |
| `test` | Test changes only | none |
| `refactor` | Code refactoring without behaviour change | none |
| `perf` | Performance improvement | none |

**Examples:**

```
fix: escape <think> tags in translation strings to prevent UNCLOSED_TAG error

feat: add temperature and top_p as configurable model parameters

feat!: drop support for Home Assistant versions below 2025.1

chore: update test dependencies

docs: document required Hailo-Ollama server setup
```

Only `fix` and `feat` commits (and breaking changes) appear in release notes. `chore`, `docs`, `test`, and `refactor` are silently included in the release but not shown in the changelog.
