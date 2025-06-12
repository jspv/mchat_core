import pytest
from unittest.mock import patch, MagicMock
from mchat_core.azure_auth import AzureADTokenProvider

@patch("mchat_core.azure_auth.ClientSecretCredential")
def test_init_success(mock_cred):
    provider = AzureADTokenProvider(
        tenant_id="tenant",
        client_id="client",
        client_secret="secret",
        resource_scope="scope/.default"
    )
    assert provider.resource_scope == "scope/.default"
    mock_cred.assert_called_once_with(tenant_id="tenant", client_id="client", client_secret="secret")

@patch("mchat_core.azure_auth.ClientSecretCredential")
def test_resource_scope_default(mock_cred):
    provider = AzureADTokenProvider(
        tenant_id="tenant",
        client_id="client",
        client_secret="secret"
    )
    assert provider.resource_scope == "client/.default"

@patch("mchat_core.azure_auth.ClientSecretCredential")
def test_token_calls_get_token(mock_cred):
    mock_cred.return_value.get_token.return_value.token = "tok"
    provider = AzureADTokenProvider(
        tenant_id="tenant",
        client_id="client",
        client_secret="secret"
    )
    token = provider.token()
    assert token == "tok"
    mock_cred.return_value.get_token.assert_called_once_with("client/.default")

@patch("mchat_core.azure_auth.ClientSecretCredential")
def test_call_is_token(mock_cred):
    mock_cred.return_value.get_token.return_value.token = "tok"
    provider = AzureADTokenProvider(
        tenant_id="tenant",
        client_id="client",
        client_secret="secret"
    )
    assert provider() == "tok"

@pytest.mark.parametrize("tenant_id,client_id,client_secret", [
    (None, "client", "secret"),
    ("tenant", None, "secret"),
    ("tenant", "client", None),
    (None, None, None),
])
def test_missing_required_raises(tenant_id, client_id, client_secret):
    with pytest.raises(ValueError) as exc:
        AzureADTokenProvider(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
    assert "missing required settings" in str(exc.value).lower()
