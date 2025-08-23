from azure.identity import ClientSecretCredential

from .logging_utils import get_logger, trace  # noqa: F401

logger = get_logger(__name__)


# Define a token provider class or function
class AzureADTokenProvider:
    """
    AzureADTokenProvider provides access tokens for Azure AD-protected resources using client credentials.

    Example:
        token_provider = AzureADTokenProvider(
            tenant_id="...",
            client_id="...",
            client_secret="...",
            resource_scope=".../.default"  # optional, defaults to "{client_id}/.default"
        )
        access_token = token_provider()  # or token_provider.token()
    """

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        resource_scope: str = None,  # Optional, defaults to "{client_id}/.default"
    ):
        """
        Initialize the AzureADTokenProvider with Azure AD credentials.

        Args:
            tenant_id (str): Azure Active Directory tenant ID.
            client_id (str): Azure AD application (client) ID.
            client_secret (str): Azure AD application client secret.
            resource_scope (str, optional): The resource scope for the token. Defaults
                to "{client_id}/.default" if not provided.

        Raises:
            ValueError: If any required credential is missing.
        """
        if not all([tenant_id, client_id, client_secret]):
            error_msg = (
                "Azure models detected, missing required settings: "
                "AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.credential = ClientSecretCredential(
            tenant_id=tenant_id, client_id=client_id, client_secret=client_secret
        )
        # If resource_scope is not provided, build it from client_id
        self.resource_scope = resource_scope or f"{client_id}/.default"

    def token(self):
        """
        Get an access token for the specified resource scope.  Caching is handled by
        the Azure SDK.

        Returns:
            str: The access token string.
        """
        token = self.credential.get_token(self.resource_scope)
        return token.token

    def __call__(self):
        """
        Callable interface to get an access token. Equivalent to calling .token().

        Returns:
            str: The access token string.
        """
        return self.token()
