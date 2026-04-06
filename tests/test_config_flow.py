"""Tests for Hailo Ollama config flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.hailo_ollama.const import (
    CONF_HOST,
    CONF_MODEL,
    CONF_PORT,
    CONF_STREAMING,
    CONF_SYSTEM_PROMPT,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_STREAMING,
    DEFAULT_SYSTEM_PROMPT,
    DOMAIN,
)
from custom_components.hailo_ollama.config_flow import HailoOllamaConfigFlow, HailoOllamaOptionsFlow


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    return hass


@pytest.fixture
def config_flow(mock_hass):
    """Create a config flow instance."""
    flow = HailoOllamaConfigFlow()
    flow.hass = mock_hass
    return flow


@pytest.mark.asyncio
async def test_test_connection_success(config_flow):
    """Test successful connection test."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"version": "1.0.0"})

    mock_session = MagicMock()
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._test_connection("localhost", 8000)

    assert result == "1.0.0"


@pytest.mark.asyncio
async def test_test_connection_failure(config_flow):
    """Test connection test failure."""
    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=Exception("Connection refused"))

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._test_connection("localhost", 8000)

    assert result is None


@pytest.mark.asyncio
async def test_fetch_models_from_api_tags(config_flow):
    """Test fetching models from /api/tags."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={"models": [{"name": "llama3.2:3b"}, {"name": "deepseek-r1:1.5b"}]}
    )

    mock_session = MagicMock()
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._fetch_models("localhost", 8000)

    assert result == ["llama3.2:3b", "deepseek-r1:1.5b"]


@pytest.mark.asyncio
async def test_fetch_models_string_format(config_flow):
    """Test fetching models when returned as strings."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={"models": ["llama3.2:3b", "deepseek-r1:1.5b"]}
    )

    mock_session = MagicMock()
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._fetch_models("localhost", 8000)

    assert result == ["llama3.2:3b", "deepseek-r1:1.5b"]


@pytest.mark.asyncio
async def test_fetch_models_empty(config_flow):
    """Test fetching models when none available."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"models": []})

    mock_session = MagicMock()
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._fetch_models("localhost", 8000)

    assert result == []


def test_config_flow_init():
    """Test config flow initialization."""
    flow = HailoOllamaConfigFlow()

    assert flow._host == DEFAULT_HOST
    assert flow._port == DEFAULT_PORT
    assert flow._models == []


# ---------------------------------------------------------------------------
# _fetch_models: fallback and both-fail paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_models_api_tags_fails_fallback_success(config_flow):
    """When /api/tags raises, fall back to /hailo/v1/list and return its models."""
    fallback_response = AsyncMock()
    fallback_response.status = 200
    fallback_response.json = AsyncMock(return_value={"models": ["hailo-model:1b"]})

    call_count = 0

    def get_side_effect(url, timeout):
        nonlocal call_count
        call_count += 1
        if "api/tags" in url:
            raise Exception("tags endpoint down")
        # fallback endpoint
        return AsyncMock(__aenter__=AsyncMock(return_value=fallback_response))

    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=get_side_effect)

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._fetch_models("localhost", 8000)

    assert result == ["hailo-model:1b"]


@pytest.mark.asyncio
async def test_fetch_models_both_fail(config_flow):
    """When both /api/tags and /hailo/v1/list fail, return an empty list."""
    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=Exception("network error"))

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        result = await config_flow._fetch_models("localhost", 8000)

    assert result == []


# ---------------------------------------------------------------------------
# _pull_model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pull_model_payload_error_treated_as_success():
    """ClientPayloadError (e.g. TransferEncodingError) during pull is treated as success."""
    import aiohttp
    from custom_components.hailo_ollama.config_flow import _pull_model

    async def fake_iter():
        yield b'{"status":"pulling"}\n'
        raise aiohttp.ClientPayloadError("Not enough data to satisfy transfer length header.")

    pull_resp = AsyncMock()
    pull_resp.status = 200
    pull_resp.content.iter_any = fake_iter

    session = MagicMock()
    session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=pull_resp)))

    success, status = await _pull_model(session, "localhost", 8000, "qwen2:1.5b")

    assert success is True


# ---------------------------------------------------------------------------
# async_step_user
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_step_user_no_input(config_flow):
    """Calling async_step_user(None) shows the host/port form."""
    form_result = {"type": "form", "step_id": "user"}
    config_flow.async_show_form = MagicMock(return_value=form_result)

    result = await config_flow.async_step_user(None)

    assert result["step_id"] == "user"
    config_flow.async_show_form.assert_called_once()


@pytest.mark.asyncio
async def test_async_step_user_cannot_connect(config_flow):
    """When _test_connection returns None, show form with cannot_connect error."""
    form_result = {"type": "form", "step_id": "user", "errors": {"base": "cannot_connect"}}
    config_flow.async_show_form = MagicMock(return_value=form_result)
    config_flow._test_connection = AsyncMock(return_value=None)

    user_input = {CONF_HOST: "badhost", CONF_PORT: 9999}
    result = await config_flow.async_step_user(user_input)

    assert result["errors"]["base"] == "cannot_connect"


@pytest.mark.asyncio
async def test_async_step_user_no_models(config_flow):
    """When connection succeeds but _fetch_models returns [], show no_models error."""
    form_result = {"type": "form", "step_id": "user", "errors": {"base": "no_models"}}
    config_flow.async_show_form = MagicMock(return_value=form_result)
    config_flow._test_connection = AsyncMock(return_value="1.0.0")
    config_flow._fetch_models = AsyncMock(return_value=[])

    user_input = {CONF_HOST: "localhost", CONF_PORT: 8000}
    result = await config_flow.async_step_user(user_input)

    assert result["errors"]["base"] == "no_models"


@pytest.mark.asyncio
async def test_async_step_user_success_proceeds_to_pick_model(config_flow):
    """When connection and models succeed, async_step_pick_model is called and form shown."""
    pick_model_form = {"type": "form", "step_id": "pick_model"}
    config_flow.async_show_form = MagicMock(return_value=pick_model_form)
    config_flow._test_connection = AsyncMock(return_value="1.0.0")
    config_flow._fetch_models = AsyncMock(return_value=["llama3.2:3b"])
    config_flow._fetch_available_models = AsyncMock(return_value=[])

    user_input = {CONF_HOST: "localhost", CONF_PORT: 8000}
    result = await config_flow.async_step_user(user_input)

    # async_step_pick_model(None) was called internally; it renders the pick_model form
    assert result["step_id"] == "pick_model"


# ---------------------------------------------------------------------------
# async_step_pick_model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_step_pick_model_no_input(config_flow):
    """Calling async_step_pick_model(None) shows the pick_model form."""
    form_result = {"type": "form", "step_id": "pick_model"}
    config_flow.async_show_form = MagicMock(return_value=form_result)
    config_flow._models = ["llama3.2:3b", "deepseek-r1:1.5b"]
    config_flow._available_models = []

    result = await config_flow.async_step_pick_model(None)

    assert result["step_id"] == "pick_model"
    config_flow.async_show_form.assert_called_once()


@pytest.mark.asyncio
async def test_async_step_pick_model_with_input_creates_entry(config_flow):
    """Providing model input creates a config entry with the correct data."""
    from custom_components.hailo_ollama.const import (
        CONF_SHOW_THINKING,
        DEFAULT_SHOW_THINKING,
    )

    created_entry = {
        "type": "create_entry",
        "title": "Hailo (llama3.2:3b)",
        "data": {
            CONF_HOST: "localhost",
            CONF_PORT: 8000,
            CONF_MODEL: "llama3.2:3b",
            CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
            CONF_STREAMING: True,
            CONF_SHOW_THINKING: DEFAULT_SHOW_THINKING,
        },
    }
    config_flow.async_create_entry = MagicMock(return_value=created_entry)
    config_flow._host = "localhost"
    config_flow._port = 8000
    config_flow._models = ["llama3.2:3b"]
    config_flow._available_models = []

    user_input = {
        CONF_MODEL: "llama3.2:3b",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
        CONF_SHOW_THINKING: DEFAULT_SHOW_THINKING,
    }
    result = await config_flow.async_step_pick_model(user_input)

    config_flow.async_create_entry.assert_called_once()
    call_kwargs = config_flow.async_create_entry.call_args
    assert call_kwargs.kwargs["title"] == "Hailo (llama3.2:3b)"
    assert call_kwargs.kwargs["data"][CONF_MODEL] == "llama3.2:3b"
    assert call_kwargs.kwargs["data"][CONF_HOST] == "localhost"
    assert call_kwargs.kwargs["data"][CONF_PORT] == 8000


# ---------------------------------------------------------------------------
# Options flow
# ---------------------------------------------------------------------------

def _make_options_flow(hass, data: dict, options: dict | None = None) -> HailoOllamaOptionsFlow:
    """Helper to create an options flow with a mock config entry."""
    config_entry = MagicMock()
    config_entry.data = data
    config_entry.options = options or {}
    flow = HailoOllamaOptionsFlow(config_entry)
    flow.hass = hass
    return flow


@pytest.mark.asyncio
async def test_options_flow_shows_form(mock_hass):
    """async_step_init with no input shows the options form."""
    flow = _make_options_flow(mock_hass, {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "model-a",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })
    flow._models = ["model-a", "model-b"]
    flow._available_models = []
    form_result = {"type": "form", "step_id": "init"}
    flow.async_show_form = MagicMock(return_value=form_result)

    result = await flow.async_step_init(None)

    assert result["step_id"] == "init"
    flow.async_show_form.assert_called_once()


@pytest.mark.asyncio
async def test_options_flow_saves_entry(mock_hass):
    """Submitting options form creates an entry with the new values."""
    from custom_components.hailo_ollama.const import CONF_SHOW_THINKING, DEFAULT_SHOW_THINKING

    flow = _make_options_flow(mock_hass, {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "model-a",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })
    flow._models = ["model-a", "model-b"]
    flow._available_models = []
    created = {"type": "create_entry", "data": {}}
    flow.async_create_entry = MagicMock(return_value=created)

    user_input = {
        CONF_MODEL: "model-b",
        CONF_SYSTEM_PROMPT: "Custom prompt",
        CONF_STREAMING: False,
        CONF_SHOW_THINKING: True,
    }
    result = await flow.async_step_init(user_input)

    flow.async_create_entry.assert_called_once_with(title="", data=user_input)


@pytest.mark.asyncio
async def test_pick_model_shows_downloadable_models_inline(mock_hass):
    """pick_model form includes model_to_pull SelectSelector when downloadable models exist."""
    from custom_components.hailo_ollama.config_flow import CONF_MODEL_TO_PULL

    flow = HailoOllamaConfigFlow()
    flow.hass = mock_hass
    flow._host = "localhost"
    flow._port = 8000
    flow._models = ["installed-model"]
    flow._available_models = ["installed-model", "new-model"]

    flow.async_show_form = MagicMock(return_value={"type": "form", "step_id": "pick_model"})

    await flow.async_step_pick_model(None)

    call_kwargs = flow.async_show_form.call_args.kwargs
    schema_keys = [getattr(k, "schema", k) for k in call_kwargs["data_schema"].schema]
    assert CONF_MODEL_TO_PULL in schema_keys


@pytest.mark.asyncio
async def test_pick_model_download_field_always_present(mock_hass):
    """pick_model form always includes model_to_pull even when no new models are available."""
    from custom_components.hailo_ollama.config_flow import CONF_MODEL_TO_PULL

    flow = HailoOllamaConfigFlow()
    flow.hass = mock_hass
    flow._host = "localhost"
    flow._port = 8000
    flow._models = ["model-a"]
    flow._available_models = ["model-a"]  # nothing new to download

    flow.async_show_form = MagicMock(return_value={"type": "form", "step_id": "pick_model"})

    await flow.async_step_pick_model(None)

    call_kwargs = flow.async_show_form.call_args.kwargs
    schema_keys = [getattr(k, "schema", k) for k in call_kwargs["data_schema"].schema]
    assert CONF_MODEL_TO_PULL in schema_keys


@pytest.mark.asyncio
async def test_pick_model_pull_success_refreshes_and_reshows(mock_hass):
    """Submitting model_to_pull pulls the model, refreshes the list, and re-shows the form."""
    from custom_components.hailo_ollama.config_flow import CONF_MODEL_TO_PULL

    flow = HailoOllamaConfigFlow()
    flow.hass = mock_hass
    flow._host = "localhost"
    flow._port = 8000
    flow._models = ["old-model"]
    flow._available_models = ["old-model", "new-model"]

    pull_chunks = [b'{"status":"pulling"}\n', b'{"status":"success"}\n']

    async def fake_pull_iter():
        for chunk in pull_chunks:
            yield chunk

    pull_resp = AsyncMock()
    pull_resp.status = 200
    pull_resp.content.iter_any = fake_pull_iter

    tags_resp = AsyncMock()
    tags_resp.status = 200
    tags_resp.json = AsyncMock(return_value={"models": [{"name": "old-model"}, {"name": "new-model"}]})

    session = MagicMock()
    session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=pull_resp)))
    session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=tags_resp)))

    flow.async_show_form = MagicMock(return_value={"type": "form", "step_id": "pick_model"})

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=session,
    ):
        await flow.async_step_pick_model({
            CONF_MODEL: "old-model",
            CONF_MODEL_TO_PULL: "new-model",
        })

    assert flow._models == ["old-model", "new-model"]
    flow.async_show_form.assert_called_once()
    assert flow.async_show_form.call_args.kwargs["step_id"] == "pick_model"


@pytest.mark.asyncio
async def test_pick_model_pull_failure_shows_error(mock_hass):
    """A failed pull shows pull_failed error and re-shows the pick_model form."""
    from custom_components.hailo_ollama.config_flow import CONF_MODEL_TO_PULL

    flow = HailoOllamaConfigFlow()
    flow.hass = mock_hass
    flow._host = "localhost"
    flow._port = 8000
    flow._models = ["old-model"]
    flow._available_models = ["old-model", "bad-model"]

    pull_resp = AsyncMock()
    pull_resp.status = 500
    pull_resp.text = AsyncMock(return_value="Internal Server Error")

    session = MagicMock()
    session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=pull_resp)))

    flow.async_show_form = MagicMock(return_value={"type": "form", "step_id": "pick_model"})

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=session,
    ):
        await flow.async_step_pick_model({
            CONF_MODEL: "old-model",
            CONF_MODEL_TO_PULL: "bad-model",
        })

    call_kwargs = flow.async_show_form.call_args.kwargs
    assert call_kwargs["errors"].get(CONF_MODEL_TO_PULL) == "pull_failed"


@pytest.mark.asyncio
async def test_options_pull_success_refreshes_and_reshows(mock_hass):
    """Submitting model_to_pull in options flow pulls, refreshes, and re-shows the form."""
    from custom_components.hailo_ollama.config_flow import CONF_MODEL_TO_PULL

    flow = _make_options_flow(mock_hass, {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "old-model",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })
    flow._models = ["old-model"]
    flow._available_models = ["old-model", "new-model"]

    pull_chunks = [b'{"status":"success"}\n']

    async def fake_pull_iter():
        for chunk in pull_chunks:
            yield chunk

    pull_resp = AsyncMock()
    pull_resp.status = 200
    pull_resp.content.iter_any = fake_pull_iter

    tags_resp = AsyncMock()
    tags_resp.status = 200
    tags_resp.json = AsyncMock(return_value={"models": [{"name": "old-model"}, {"name": "new-model"}]})

    session = MagicMock()
    session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=pull_resp)))
    session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=tags_resp)))

    flow.async_show_form = MagicMock(return_value={"type": "form", "step_id": "init"})

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=session,
    ):
        await flow.async_step_init({
            CONF_MODEL: "old-model",
            CONF_MODEL_TO_PULL: "new-model",
        })

    assert flow._models == ["old-model", "new-model"]
    flow.async_show_form.assert_called_once()
    assert flow.async_show_form.call_args.kwargs["step_id"] == "init"


@pytest.mark.asyncio
async def test_options_flow_fetches_models(mock_hass):
    """Options flow fetches models from /api/tags when _models is empty."""
    flow = _make_options_flow(mock_hass, {
        CONF_HOST: "localhost",
        CONF_PORT: 8000,
        CONF_MODEL: "model-a",
        CONF_SYSTEM_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_STREAMING: True,
    })

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"models": [{"name": "model-a"}, {"name": "model-b"}]})
    mock_session = MagicMock()
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    form_result = {"type": "form", "step_id": "init"}
    flow.async_show_form = MagicMock(return_value=form_result)
    flow._fetch_available_models = AsyncMock(return_value=[])

    with patch(
        "custom_components.hailo_ollama.config_flow.async_get_clientsession",
        return_value=mock_session,
    ):
        await flow.async_step_init(None)

    assert flow._models == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_options_flow_uses_options_as_defaults(mock_hass):
    """Options form pre-fills from entry.options when present."""
    from custom_components.hailo_ollama.const import CONF_SHOW_THINKING

    flow = _make_options_flow(
        mock_hass,
        data={CONF_HOST: "localhost", CONF_PORT: 8000, CONF_MODEL: "model-a"},
        options={
            CONF_MODEL: "model-b",
            CONF_SYSTEM_PROMPT: "Options prompt",
            CONF_STREAMING: False,
            CONF_SHOW_THINKING: True,
        },
    )
    flow._models = ["model-a", "model-b"]
    flow._available_models = []
    flow.async_show_form = MagicMock(return_value={"type": "form", "step_id": "init"})

    await flow.async_step_init(None)

    call_kwargs = flow.async_show_form.call_args.kwargs
    schema = call_kwargs["data_schema"].schema
    # The default for CONF_MODEL should come from options
    model_key = next(k for k in schema if getattr(k, "schema", None) == CONF_MODEL)
    assert model_key.default() == "model-b"
