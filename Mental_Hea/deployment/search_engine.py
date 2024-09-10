import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models
from src.core.config import settings
from src.core.embeddings_model import EmbeddingsModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


QDRANT_URL = "https://e932e81a-113e-440f-96c0-c17b530bfe79.europe-west3-0.gcp.cloud.qdrant.io:6333/dashboard"


class SearchEngine:
    """Manages hybrid search operations using Qdrant vector database and text embeddings."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.embeddings_model = EmbeddingsModel.get_instance()
        self.client = QdrantClient(
            url=QDRANT_URL, api_key=settings.QDRANT_API_KEY
        )

    def search_dense(self, query_text: str):
        try:
            dense_embeddings, _ = self.embeddings_model.embeddings(
                [query_text]
            )
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_embeddings[0],
                using="text-dense",
                limit=5,
                with_payload=True,
            )
            return results
        except Exception as e:
            logger.error(f"Error performing dense search: {str(e)}")
            raise

    def search_late(self, query_text: str):
        try:
            _, late_embeddings = self.embeddings_model.embeddings([query_text])
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=late_embeddings[0],
                using="text-late",
                limit=5,
                with_payload=True,
            )
            return results
        except Exception as e:
            logger.error(f"Error performing late interaction search: {str(e)}")
            raise

    def search_hybrid(self, query_text: str):
        try:
            # Get embeddings from your model
            dense_embeddings, late_embeddings = (
                self.embeddings_model.embeddings([query_text])
            )

            # Prepare prefetch query for initial retrieval
            prefetch = [
                models.Prefetch(
                    query=dense_embeddings[0],
                    using="text-dense",
                    limit=30,  # Retrieve top 20 for reranking
                ),
            ]

            # Perform combined search with initial retrieval and reranking
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=late_embeddings[0],
                using="text-late",
                limit=5,  # Final limit after reranking
                with_payload=True,
            )

            return results

        except Exception as e:
            logger.error(
                f"Error performing hybrid search with reranking: {str(e)}"
            )
            raise
