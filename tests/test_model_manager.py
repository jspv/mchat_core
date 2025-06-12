# test_model_manager.py

import pytest
from dynaconf import Dynaconf
from unittest.mock import patch

@pytest.fixture
def dynaconf_test_settings(tmp_path, monkeypatch):
    """
    A pytest fixture to generate test configs on the fly and
    patch get_settings to return these configs.
    """
    settings_toml = tmp_path / "settings.toml"
    settings_content = """

# Azure Stuff
azure_client_id = "test-client-id"
azure_tenant_id = "test-tenant-id"
azure_client_id_secret = "test-client-secret"

[models.chat.gpt-4_1]
api_key = "dummy_key"
model = "gpt-4.1"
api_type = "open_ai"
base_url = "https://api.openai.com/v1"
_image_input_support = true
_cost_input = 2.00
_cost_output = 8.00
_max_context = 1047576
_max_output = 32768
_streaming_support = true
_tool_support = false
_system_prompt_support = true
_temperature_support = true
_structured_output_support = true

[models.image.dall-e-3]
api_key = "dummy_key"
model = "dall-e-3"
api_type = "open_ai"
size = "1024x1024"
quality = "standard"
num_images = 1

[models.embedding.text-embedding-ada-002]
api_key = "dummy_key"
model = "text-embedding-ada-002"
api_type = "open_ai"

# Example azure chat model that uses 'provider' to show Azure AD token usage
[models.chat.azure-model]
api_key = "provider"
azure_deployment = "mydeployment"
api_version = "2023-07-31"
azure_endpoint = "https://myazure.endpoint"
model = "gpt-4o"
api_type = "azure"
_streaming_support = true
_tool_support = false
_system_prompt_support = true
_temperature_support = true

[defaults]
chat_model = "gpt-4_1"
image_model = "dall-e-3"
embedding_model = "text-embedding-ada-002"
memory_model = "gpt-4_1"
chat_temperature = 0.7
memory_model_temperature = 0.1
memory_model_max_tokens = 2048
    """
    settings_toml.write_text(settings_content)
    test_settings = Dynaconf(settings_files=[str(settings_toml)])

    def dummy_get_settings(*args, **kwargs):
        # Always return our test_settings, ignoring passed arguments
        return test_settings

    # Patch the get_settings function in your module so
    # the test fixture above is always used
    monkeypatch.setattr("mchat_core.config.get_settings", dummy_get_settings)
    print("Test settings loaded:")
    print(test_settings.to_dict())
    return test_settings


def test_model_manager_loads_models(dynaconf_test_settings):
    """
    Test that our ModelManager loads the models listed in the test TOML file.
    """
    from mchat_core.model_manager import ModelManager
    mm = ModelManager()
    # Confirm each model is loaded
    # The print here is just for demonstration; not typically used in real tests
    print(dynaconf_test_settings.to_dict())
    assert "gpt-4_1" in mm.config
    assert "dall-e-3" in mm.config
    assert "text-embedding-ada-002" in mm.config
    # Confirm azure-model also got loaded
    assert "azure-model" in mm.config


def test_filter_models_and_props(dynaconf_test_settings):
    """
    Test filtering of models by various props and ensure
    property getters match the TOML for 'gpt-4_1'.
    """
    from mchat_core.model_manager import ModelManager
    mm = ModelManager()
    # Check explicit flags for gpt-4_1
    assert mm.get_streaming_support("gpt-4_1") is True
    assert mm.get_tool_support("gpt-4_1") is False
    assert mm.get_system_prompt_support("gpt-4_1") is True
    assert mm.get_temperature_support("gpt-4_1") is True
    assert mm.get_structured_output_support("gpt-4_1") is True

    # Filtering checks
    # Notice that azure-model is also a chat model with _streaming_support = true
    assert mm.filter_models({"model_type": ["chat"], "_streaming_support": [True]}) == [
        "gpt-4_1", "azure-model"
    ]
    # No chat model in this config has _tool_support = true
    assert mm.filter_models({"model_type": ["chat"], "_tool_support": [True]}) == []
    assert mm.filter_models({"api_type": ["open_ai"], "model_type": ["chat"]}) == ["gpt-4_1"]
    assert mm.filter_models({"model_type": ["image"]}) == ["dall-e-3"]


def test_property_lists(dynaconf_test_settings):
    """
    Testing the built-in properties for listing chat, image,
    and embedding models. Confirm we see the models from the TOML.
    """
    from mchat_core.model_manager import ModelManager
    mm = ModelManager()
    # For chat (our fixture includes gpt-4_1 and azure-model)
    assert set(mm.available_chat_models) == {"gpt-4_1", "azure-model"}
    # For image
    assert mm.available_image_models == ["dall-e-3"]
    # For embeddings
    assert mm.available_embedding_models == ["text-embedding-ada-002"]


def test_get_compatible_models(dynaconf_test_settings):
    """
    If an agent has 'tools' in their config, we only include models
    that also have '_tool_support' = True. Currently, none do,
    so the 'tooluser' agent returns [].
    """
    from mchat_core.model_manager import ModelManager
    mm = ModelManager()
    agents = {
        "default": {},
        "tooluser": {"tools": ["code_interpreter"]},
    }
    compat_default = mm.get_compatible_models("default", agents)
    compat_tool = mm.get_compatible_models("tooluser", agents)
    # 'gpt-4_1' and 'azure-model' have _tool_support = false in the test TOML
    # So they appear for 'default' but not for 'tooluser'
    assert "gpt-4_1" in compat_default
    assert "azure-model" in compat_default
    assert compat_tool == []


@patch("mchat_core.model_manager.OpenAIChatCompletionClient")
@patch("mchat_core.model_manager.DallEAPIWrapper")
def test_open_model_dispatches_clients(mock_dalle, mock_chat, dynaconf_test_settings):
    """
    Check that open_model() produces an OpenAIChatCompletionClient for 'gpt-4_1'
    and a DallEAPIWrapper for 'dall-e-3'.
    """
    from mchat_core.model_manager import ModelManager
    mm = ModelManager()
    mm.open_model("gpt-4_1")
    mock_chat.assert_called_once()
    mm.open_model("dall-e-3")
    mock_dalle.assert_called_once()


def test_open_model_azure_chat_with_token_provider(dynaconf_test_settings):
    """
    This test will confirm that if an Azure model has 'api_key' == 'provider',
    the ModelManager initializes the AzureADTokenProvider and opens an Azure
    chat client with that token provider.
    """
    from mchat_core.model_manager import ModelManager, AzureOpenAIChatCompletionClient
    mm = ModelManager()
    # "azure-model" is defined in test fixture to use 'api_key' = 'provider'
    client = mm.open_model("azure-model")
    # Confirm we got an Azure client
    assert isinstance(client, AzureOpenAIChatCompletionClient)
    # Confirm azure_token_provider is actually set
    assert mm.azure_token_provider is not None
    # config that azure_token_provider is not set when there is a key
    # Note - also tests ModelManager direct setting import
    dynaconf_test_settings.set("models__chat__azure-model__api_key","some-other-token")
    mm2 = ModelManager(settings_conf=dynaconf_test_settings)
    assert mm2.azure_token_provider is None


def test_open_model_invalid_model_type(dynaconf_test_settings):
    """
    Test that if config tries to open a model_type that isn't 'chat', 'image',
    or 'embedding', open_model raises a ValueError.
    """
    from mchat_core.model_manager import ModelManager, ModelConfig
    mm = ModelManager()
    # Manually insert an invalid model
    invalid_model_id = "unknown-model"
    mm.config[invalid_model_id] = ModelConfig(
        model_id=invalid_model_id,
        model="my-weird-model",
        model_type="some_weird_type",
        api_type="open_ai",
    )
    with pytest.raises(ValueError) as exc:
        mm.open_model(invalid_model_id)
    assert "Invalid model_type" in str(exc.value)


@patch("mchat_core.model_manager.OpenAIChatCompletionClient")
def test_open_model_embedding_should_fail_by_default(mock_openai, dynaconf_test_settings):
    """
    The code as written only supports opening chat or image models.
    Attempting to open an embedding model raises ValueError,
    unless you later extend the logic for embeddings.
    """
    from mchat_core.model_manager import ModelManager
    mm = ModelManager()
    with pytest.raises(ValueError) as exc:
        mm.open_model("text-embedding-ada-002")
    assert "Invalid model_type" in str(exc.value)


@patch("mchat_core.model_manager.ModelManager.open_model")
def test_static_ask_and_aask_sync(mock_open_model, dynaconf_test_settings):
    class MockResponse:
        def __init__(self, content):
            self.content = content
    class MockClient:
        async def create(self, msgs):
            return MockResponse("response")
    mock_open_model.return_value = MockClient()
    from mchat_core.model_manager import ModelManager
    output_sync = ModelManager.ask("hi", "gpt-4_1")
    assert "response" in output_sync