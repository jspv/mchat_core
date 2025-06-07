from azure.identity import ClientSecretCredential

from .config import get_settings
from .logging_utils import get_logger, trace  # noqa: F401

logger = get_logger(__name__)
settings = get_settings()

tenant_id = settings.get("azure_tenant_id")
client_id = settings.get("azure_client_id")
client_secret = settings.get("azure_client_id_secret")
azure_endpoint = settings.get("azure_endpoint")

# Check if all required settings are present and create a credential if they're present
if all([tenant_id, client_id, client_secret]):
    credential = ClientSecretCredential(
        tenant_id=tenant_id, client_id=client_id, client_secret=client_secret
    )
else:
    credential = None

default_scope = f"{settings.get('azure_client_id')}/.default"


# Define a token provider class or function
class AzureADTokenProvider:
    """
    Example:
    token_provider = AzureADTokenProvider(
        credential, f"{settings.get('azure_client_id')}/.default"
    )
    """

    def __init__(
        self,
        resource_scope: str = default_scope,
    ):
        if not credential:
            error_msg = (
                "Azure models detected, missing required settings: "
                "AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.credential = credential
        self.resource_scope = resource_scope

    def __call__(self):
        token = self.credential.get_token(self.resource_scope)
        return token.token
