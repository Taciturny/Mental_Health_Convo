import weaviate
import logging
from typing import Dict, Any

class WeaviateClient:
    """Class for managing Weaviate client operations."""

    def __init__(self, url: str = "http://localhost:8080"):
        """
        Initialize the WeaviateClient class.

        Args:
            url (str): The URL of the Weaviate instance. Defaults to localhost.
        """
        self.client = weaviate.Client(url)
        self.logger = logging.getLogger(__name__)

    def get_client(self) -> weaviate.Client:
        """
        Get the Weaviate client instance.

        Returns:
            weaviate.Client: The Weaviate client instance.
        """
        return self.client

    def check_connection(self) -> bool:
        """
        Check the connection to the Weaviate instance.

        Returns:
            bool: True if connected successfully, False otherwise.
        """
        try:
            self.client.is_ready()
            self.logger.info("Successfully connected to Weaviate.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Weaviate: {str(e)}")
            return False

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the current schema from Weaviate.

        Returns:
            Dict[str, Any]: The current Weaviate schema.
        """
        return self.client.schema.get()

    def close_connection(self) -> None:
        """Close the connection to the Weaviate instance."""
        if self.client:
            self.client = None
            self.logger.info("Connection to Weaviate closed.")
