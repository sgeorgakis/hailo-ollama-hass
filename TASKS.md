# Future Tasks

## Features

- [x] Add translations for Greek, German, Italian, Portuguese, Spanish, French, Brazilian Portuguese
- [ ] Update the project to support only Home Assistant version 2025 and on
- [ ] Allow user to choose in configuration if they want the thinking process to be returned or not
- [ ] Add options flow to allow reconfiguring model, system prompt, and streaming mode after initial setup
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
- [ ] Add HACS manifest for easy installation via HACS
