import weaviate
import logging
from typing import Dict, Any

class WeaviateSchema:
    """Class for managing Weaviate schema operations."""

    def __init__(self, client: weaviate.Client):
        """
        Initialize the WeaviateSchema class.

        Args:
            client (weaviate.Client): An instance of the Weaviate client.
        """
        self.client = client
        self.logger = logging.getLogger(__name__)

    def schema_exists(self) -> bool:
        try:
            schema = self.client.schema.get()
            return any(class_obj['class'] == 'MentalHealth' for class_obj in schema['classes'])
        except Exception as e:
            self.logger.error(f"Error checking schema existence: {str(e)}")
            return False
        
    def create_schema(self) -> None:
        if self.schema_exists():
            self.logger.info("MentalHealth schema already exists.")
            return

        class_obj: Dict[str, Any] = {
            "class": "MentalHealth",
            "vectorizer": "none",
            "properties": [
                {
                    "name": "context",
                    "dataType": ["text"],
                    "description": "The context of the conversation",
                },
                {
                    "name": "response",
                    "dataType": ["text"],
                    "description": "The response to the context",
                },
                {
                    "name": "document",
                    "dataType": ["string"],
                    "description": "Unique document identifier",
                },
                {
                    "name": "context_vector",
                    "dataType": ["number[]"],
                    "vectorIndexType": "hnsw",
                    "vectorizer": "none",
                    "description": "Vector representation of the context",
                },
                {
                    "name": "response_vector",
                    "dataType": ["number[]"],
                    "vectorIndexType": "hnsw",
                    "vectorizer": "none",
                    "description": "Vector representation of the response",
                },
                {
                    "name": "context_response_vector",
                    "dataType": ["number[]"],
                    "vectorIndexType": "hnsw",
                    "vectorizer": "none",
                    "description": "Vector representation of the context and response",
                },
                {
                    "name": "context_sentiment_vector",
                    "dataType": ["number[]"],
                    "vectorIndexType": "hnsw",
                    "vectorizer": "none",
                    "description": "Sentiment vector representation of the context",
                },
                {
                    "name": "response_sentiment_vector",
                    "dataType": ["number[]"],
                    "vectorIndexType": "hnsw",
                    "vectorizer": "none",
                    "description": "Sentiment vector representation of the response",
                },
                {
                    "name": "context_response_sentiment_vector",
                    "dataType": ["number[]"],
                    "vectorIndexType": "hnsw",
                    "vectorizer": "none",
                    "description": "Sentiment vector representation of the combined context and response",
                }
            ]
        }

        try:
            self.client.schema.create_class(class_obj)
            self.logger.info("Schema created successfully.")
        except Exception as e:
            self.logger.error(f"Error creating schema: {str(e)}")

    def delete_schema(self) -> None:
        """Delete the MentalHealth class schema from Weaviate."""
        try:
            self.client.schema.delete_class("MentalHealth")
            self.logger.info("Schema deleted successfully.")
        except Exception as e:
            self.logger.error(f"Error deleting schema: {str(e)}")

    def delete_all_data(self) -> None:
        """Delete all data from the MentalHealth class."""
        try:
            self.client.data_object.delete(
                class_name="MentalHealth",
                where={
                    "operator": "NotNull",
                    "path": ["id"]
                }
            )
            self.logger.info("All data deleted successfully.")
        except Exception as e:
            self.logger.error(f"Error deleting data: {str(e)}")


    def perform_action(self, action: str) -> None:
        """Perform the user-specified action."""
        if action == 'delete_schema':
            self.delete_schema()
        elif action == 'delete_data':
            self.delete_all_data()
        elif action == 'create_schema':
            self.create_schema()
        else:
            print("Invalid action. Please choose 'delete_schema', 'delete_data', or 'create_schema'.")
