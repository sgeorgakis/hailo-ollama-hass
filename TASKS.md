# Future Tasks

## Bug Fixes

## Features

- [x] **Image input support (vision models)**

  Allow users to send images alongside text to vision-capable models (e.g. `llava`).

  **How the Ollama API accepts images:**
  Images are passed as a list of base64-encoded strings in the message object:
  ```json
  {
    "model": "llava",
    "messages": [
      {
        "role": "user",
        "content": "What is in this image?",
        "images": ["<base64-encoded-image>"]
      }
    ]
  }
  ```

  **What needs to change:**
  - `conversation.py:91` — `_conversations` history is typed as `dict[str, list[dict[str, str]]]`; change value type to `dict[str, Any]` to allow the `images` key.
  - `conversation.py:121–125` — message construction only sets `role` and `content`. Add logic to attach an `images` list when image data is present.
  - `conversation.py:168–176` — `_build_payload` passes messages as-is; no change needed here, but verify images survive serialisation.
  - Investigate whether `homeassistant.components.conversation.ConversationInput` carries image/attachment data in the HA version targeted (2025.1+). If not, an alternative entry point (e.g. a custom service call) may be needed to pass image bytes or a file path.
  - Consider adding a config-flow option to mark the selected model as vision-capable, so the UI can surface image input controls appropriately.
- [x] Implement AI tasks functionality
- [ ] Expose model parameters (temperature, top_p, max_tokens) as configurable options
- [ ] Add sensor entities for response metrics (tokens/sec, response time, token count)
- [ ] Add support for additional languages beyond English in `supported_languages`
- [ ] Implement connection health check / auto-reconnect logic
- [ ] Add Home Assistant diagnostics support for debugging
- [ ] Add service calls for:
  - Listing available models
  - Switching models at runtime
  - Clearing conversation context

## Code Quality

- [ ] Add integration tests with mocked Hailo-Ollama server
- [ ] More granular error types (connection refused vs timeout vs invalid response)

## Documentation

- [ ] Document required Hailo-Ollama server setup
