import weaviate
import logging
from typing import Dict, Any, List
from app.services.embeddings import EmbeddingModel

class Search:
    """Class for performing various types of searches in Weaviate."""

    def __init__(self, client: weaviate.Client):
        """
        Initialize the Search class.

        Args:
            client (weaviate.Client): An instance of the Weaviate client.
        """
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.model = EmbeddingModel.get_instance()

    def vector_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a vector search in Weaviate.

        Args:
            query (str): The search query.
            limit (int): The maximum number of results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        query_vector = self.model.encode(query)[0]  # Get the first (and only) vector
        result = (
            self.client.query
            .get("MentalHealth", ["context", "response"])
            .with_near_vector({"vector": query_vector})
            .with_limit(limit)
            .do()
        )
        self.logger.info(f"Vector search performed with query: {query}")
        self.logger.debug(f"Vector search result: {result}")
        return result.get('data', {}).get('Get', {}).get('MentalHealth', [])

    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a semantic search in Weaviate using BM25.

        Args:
            query (str): The search query.
            limit (int): The maximum number of results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        result = (
            self.client.query
            .get("MentalHealth", ["context", "response"])
            .with_bm25(query)
            .with_limit(limit)
            .do()
        )
        self.logger.info(f"Semantic search performed with query: {query}")
        self.logger.debug(f"Semantic search result: {result}")
        return result.get('data', {}).get('Get', {}).get('MentalHealth', [])

    def hybrid_search(self, query: str, limit: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search in Weaviate, combining vector and BM25 search.

        Args:
            query (str): The search query.
            limit (int): The maximum number of results to return. Defaults to 5.
            alpha (float): The balance between vector and keyword search. Defaults to 0.5.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        query_vector = self.model.encode(query)
        result = (
            self.client.query
            .get("MentalHealth", ["context", "response"])
            .with_hybrid(query, alpha=alpha, vector=query_vector)
            .with_limit(limit)
            .do()
        )
        self.logger.info(f"Hybrid search performed with query: {query}")
        
        # Log the full response for debugging
        self.logger.debug(f"Hybrid search response: {result}")
        
        return result.get('data', {}).get('Get', {}).get('MentalHealth', [])

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Display the search results.

        Args:
            results (List[Dict[str, Any]]): The list of search results to display.
        """
        for i, item in enumerate(results, 1):
            self.logger.info(f"\nResult {i}:")
            self.logger.info(f"Context: {item['context']}")
            self.logger.info(f"Response: {item['response']}")
