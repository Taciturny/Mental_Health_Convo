import logging
from typing import List, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import settings

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self, host="localhost", port=6333, url=None, api_key=None):
        if url and api_key:
            # Cloud initialization
            self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        else:
            # Local initialization
            self.client = QdrantClient(host=host, port=port)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_collection(self, collection_name: str, vector_size: int):
        """
        Create a new collection in Qdrant.

        Args:
            collection_name (str): Name of the collection to create.
            vector_size (int): Size of the vector embeddings.
        """
        try:
            logger.info(f"Creating collection: {collection_name}")
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance="Cosine"),
                timeout=30
            )
            logger.info(f"Collection {collection_name} created successfully")
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise


    def prepare_points(self, data: List[tuple], embeddings: List):
        """
        Prepare points for insertion into Qdrant.

        Args:
            data (List[tuple]): List of data tuples (id, question, answer).
            embeddings (List): List of embeddings.

        Returns:
            List[PointStruct]: List of prepared points.
        """
        try:
            logger.info("Preparing points for insertion")
            points = [
                PointStruct(
                    id=str(item[0]),
                    vector=embedding.tolist(),
                    payload={
                        "question": item[1],
                        "answer": item[2]
                    }
                )
                for item, embedding in zip(data, embeddings)
            ]
            logger.info(f"Prepared {len(points)} points")
            return points
        except Exception as e:
            logger.error(f"Error preparing points: {str(e)}")
            raise

    def insert_points(self, collection_name: str, points: List[PointStruct], batch_size: int = 500):
        """
        Insert points into the Qdrant collection.

        Args:
            collection_name (str): Name of the collection to insert into.
            points (List[PointStruct]): List of points to insert.
            batch_size (int): Number of points to insert in each batch.
        """
        try:
            logger.info(f"Inserting {len(points)} points into {collection_name}")
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(collection_name=collection_name, points=batch)
            logger.info("Points inserted successfully")
        except Exception as e:
            logger.error(f"Error inserting points: {str(e)}")
            raise

    def init_collection(self, collection_name: str, vector_size: int):
        """
        Initialize a collection if it doesn't exist.

        Args:
            collection_name (str): Name of the collection.
            vector_size (int): Size of the vector embeddings.

        Returns:
            bool: True if a new collection was created, False if it already existed.
        """
        try:
            if not self.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' does not exist. Creating new collection.")
                self.create_collection(collection_name, vector_size)
                return True
            else:
                logger.info(f"Collection '{collection_name}' already exists.")
                return False
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Qdrant.

        Args:
            collection_name (str): Name of the collection to check.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        try:
            collections = self.client.get_collections().collections
            return any(collection.name == collection_name for collection in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False

    def index_and_insert_data(self, collection_name: str, data: List[Tuple[str, str, str]], embeddings: List[np.ndarray]):
        """
        Index and insert data into Qdrant.

        Args:
            collection_name (str): Name of the collection to insert into.
            data (List[Tuple[str, str, str]]): List of data tuples (id, question, answer).
            embeddings (List[np.ndarray]): List of embeddings.
        """
        try:
            logger.info(f"Indexing and inserting {len(data)} points into {collection_name}")
            points = self.prepare_points(data, embeddings)
            self.insert_points(collection_name, points)
            logger.info("Data indexed and inserted successfully")
        except Exception as e:
            logger.error(f"Error indexing and inserting data: {str(e)}")
            raise

    def collection_is_empty(self, collection_name: str) -> bool:
        """
        Check if the collection is empty.

        Args:
            collection_name (str): Name of the collection to check.

        Returns:
            bool: True if the collection is empty, False otherwise.
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            return collection_info.points_count == 0
        except Exception as e:
            logger.error(f"Error checking if collection is empty: {str(e)}")
            return True   # Assume empty if there's an error
