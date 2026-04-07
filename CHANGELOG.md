# Changelog

## [1.0.1](https://github.com/sgeorgakis/hailo-ollama-hass/compare/v1.0.0...v1.0.1) (2026-04-07)


### Bug Fixes

* decorate sensor/entity handlers with [@callback](https://github.com/callback) ([801bf4d](https://github.com/sgeorgakis/hailo-ollama-hass/commit/801bf4dd5c95a734fe79c54c17f1a5b41277ef4a))
* decorate sensor/entity handlers with [@callback](https://github.com/callback) and dispatch metrics from ai_task ([115c7b9](https://github.com/sgeorgakis/hailo-ollama-hass/commit/115c7b9ecb856ebeda6f3b6707a8ad1c031abbc6))

## [1.0.0](https://github.com/sgeorgakis/hailo-ollama-hass/compare/v0.3.0...v1.0.0) (2026-04-06)


### ⚠ BREAKING CHANGES

* add model management UI, services, response sensors, and auto-reconnect

### Features

* add model management UI, services, response sensors, and auto-reconnect ([9f99c2f](https://github.com/sgeorgakis/hailo-ollama-hass/commit/9f99c2fa4ea445a548fe7aaf4f883127c857a4ae))

## [0.3.0](https://github.com/sgeorgakis/hailo-ollama-hass/compare/v0.2.1...v0.3.0) (2026-04-06)


### Features

* add brand logo and dark mode icon assets ([b2a9915](https://github.com/sgeorgakis/hailo-ollama-hass/commit/b2a9915006e744b30c0e828832a1d0fecda3fe7f))
* remove vision_model config option; attachments always processed ([241f10b](https://github.com/sgeorgakis/hailo-ollama-hass/commit/241f10b0bbe1d559e8c4cf02d57d8f52f73330bb))
* support all languages in conversation agent ([1a78bfe](https://github.com/sgeorgakis/hailo-ollama-hass/commit/1a78bfea024a103b785277483818284bd554a308))


### Bug Fixes

* move icon to component root and require Python 3.13 ([a618426](https://github.com/sgeorgakis/hailo-ollama-hass/commit/a6184261ea01fd586ff2521b32855e72af36eed4))
* use brand/ folder for icon and require HA 2026.3 ([6d1555e](https://github.com/sgeorgakis/hailo-ollama-hass/commit/6d1555ec05aa697cefa1f916f18236a4bc683d11))

## [0.2.2](https://github.com/sgeorgakis/hailo-ollama-hass/compare/v0.2.1...v0.2.2) (2026-04-06)


### Features

* expose AI task platform so the integration appears in the HA AI task picker ([eb13aa2](https://github.com/sgeorgakis/hailo-ollama-hass/commit/eb13aa2281f06857cab62bcdc734fa4ce39fc787))


### Bug Fixes

* use `Platform.AI_TASK` enum instead of bare string so the platform registers correctly ([eb13aa2](https://github.com/sgeorgakis/hailo-ollama-hass/commit/eb13aa2281f06857cab62bcdc734fa4ce39fc787))
* correct `_async_generate_data` signature and pass `conversation_id` to `GenDataTaskResult` ([eb13aa2](https://github.com/sgeorgakis/hailo-ollama-hass/commit/eb13aa2281f06857cab62bcdc734fa4ce39fc787))
* use `_attr_supported_features` so entity is visible in the AI task agent picker ([eb13aa2](https://github.com/sgeorgakis/hailo-ollama-hass/commit/eb13aa2281f06857cab62bcdc734fa4ce39fc787))
* reload config entry on options change so updated settings take effect without HA restart ([eb13aa2](https://github.com/sgeorgakis/hailo-ollama-hass/commit/eb13aa2281f06857cab62bcdc734fa4ce39fc787))

## [0.2.1](https://github.com/sgeorgakis/hailo-ollama-hass/compare/v0.2.0...v0.2.1) (2026-04-03)


### Bug Fixes

* rename RunDataTaskResult to GenDataTaskResult to match HA api_task API ([9da5d54](https://github.com/sgeorgakis/hailo-ollama-hass/commit/9da5d540e2949b2605c60810f5fc2ae245e2ef55))

## [0.2.0](https://github.com/sgeorgakis/hailo-ollama-hass/compare/v0.1.4...v0.2.0) (2026-03-27)


### Features

* add AI task entity and vision model support ([07b78b3](https://github.com/sgeorgakis/hailo-ollama-hass/commit/07b78b34b9e7b3d8ecd71d5088f011cbf3d71977))

## [0.1.4](https://github.com/sgeorgakis/hailo-ollama-hass/compare/v0.1.3...v0.1.4) (2026-03-11)


### Bug Fixes

* resize icon to 256x256 ([46b1e34](https://github.com/sgeorgakis/hailo-ollama-hass/commit/46b1e342f3acad68aec982399ebc3519873fe9a2))

## [0.1.3](https://github.com/sgeorgakis/hailo-ollama-hass/compare/v0.1.2...v0.1.3) (2026-03-11)


### Bug Fixes

* handle orphaned &lt;/think&gt; tag and wrap thinking in &lt;i&gt; when shown ([634e66b](https://github.com/sgeorgakis/hailo-ollama-hass/commit/634e66bb8c9763f2fc2d711d8e37660fb740474a))
