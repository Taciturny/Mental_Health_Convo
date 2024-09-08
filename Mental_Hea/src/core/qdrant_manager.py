import logging
from typing import List, Any, Dict

import time
import numpy as np
from qdrant_client import QdrantClient, models
from tenacity import retry, stop_after_attempt, wait_exponential
# from core.config import settings

logger = logging.getLogger(__name__)

import tqdm
from typing import List
from qdrant_client.http.models import PointStruct


logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self, url: str = None, api_key: str = None, host: str = "localhost", port: int = 6333):
        if url and api_key:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=600)
            logger.info("Initialized Qdrant client with cloud settings.")
        else:
            self.client = QdrantClient(host=host, port=port, timeout=600)
            logger.info("Initialized Qdrant client with local settings.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_collection(self, collection_name: str, retries: int = 3, delay: int = 5):
        """
        Create a Qdrant collection with the specified name and vector configurations.

        Args:
            collection_name (str): Name of the collection to be created.
            retries (int): Number of times to retry creating the collection.
            delay (int): Delay between retries in seconds.
        """
        vectors_config = {
            "text-dense": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            ),
            "text-late": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
            )
        }

        for attempt in range(retries):
            try:
                self.client.create_collection(
                    collection_name,
                    vectors_config=vectors_config,
                )
                logger.info(f"Collection '{collection_name}' created successfully.")
                return
            except TimeoutError as e:
                logger.warning(f"Timeout error creating collection '{collection_name}' (attempt {attempt + 1}/{retries}): {str(e)}")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Error creating collection '{collection_name}': {str(e)}")
                raise

        logger.error(f"Failed to create collection '{collection_name}' after {retries} attempts.")
        raise TimeoutError(f"Failed to create collection '{collection_name}'.")


    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            collections = self.client.get_collections().collections
            return any(collection.name == collection_name for collection in collections)
        except Exception as e:
            logger.error(f"Error checking if collection exists: {str(e)}")
            return False

    def create_collection_if_not_exists(self, collection_name: str):
        """Create a collection only if it doesn't already exist."""
        try:
            if not self.collection_exists(collection_name):
                self.create_collection(collection_name)
                logger.info(f"Collection '{collection_name}' created successfully.")
            else:
                logger.info(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise


    def prepare_and_upload_points(self, data: List[Dict[str, str]], dense_embeddings: List[np.ndarray], late_embeddings: List[np.ndarray], client, collection_name: str, batch_size: int = 50):
        """
        Prepare points and upload them to Qdrant in batches.
        
        Args:
            data (List[Dict[str, str]]): List of data dictionaries.
            dense_embeddings (List[np.ndarray]): List of dense embeddings.
            late_embeddings (List[np.ndarray]): List of late embeddings.
            client: Qdrant client instance to upload points.
            collection_name: The name of the collection in Qdrant.
            batch_size (int): The size of each batch. Default is 50.
        """
        
        # Iterate over data in batches and upload
        for batch_start in tqdm.tqdm(range(0, len(data), batch_size), total=len(data) // batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            
            batch_data = data[batch_start:batch_end]
            batch_dense_embeddings = dense_embeddings[batch_start:batch_end]
            batch_late_embeddings = late_embeddings[batch_start:batch_end]
            
            # Prepare points for the current batch
            batch_points = [
                PointStruct(
                    id=item['id'],  # Use the 'id' field from the data
                    vector={
                        "text-dense": batch_dense_embeddings[i].tolist(),
                        "text-late": batch_late_embeddings[i].tolist()
                    },
                    payload={
                        "id": item['id'],
                        "question": item['question'],
                        "answer": item['answer']
                    }
                )
                for i, item in enumerate(batch_data)
            ]
            
            # Upload the batch of points to Qdrant
            client.upload_points(
                collection_name=collection_name,
                points=batch_points,
                batch_size=batch_size
            )

    def _retry_operation(self, operation: Any, retries: int, delay: int, error_message: str):
            """
            Retry an operation with a specified number of retries and delay.
            Args:
                operation (Callable): The operation to perform.
                retries (int): Number of retries.
                delay (int): Delay between retries.
                error_message (str): Message to log on error.
            """
            for attempt in range(retries):
                try:
                    operation()
                    return
                except TimeoutError as e:
                    logger.warning(f"{error_message} (attempt {attempt + 1}/{retries}): {str(e)}")
                    time.sleep(delay)
                except Exception as e:
                    logger.error(f"{error_message}: {str(e)}")
                    if attempt == retries - 1:
                        raise
                    time.sleep(delay)

    def _process_in_batches(self, data: List[Any], batch_size: int, operation: Any, log_message: str):
        """
        Process data in batches and perform an operation on each batch.
        Args:
            data (List[Any]): List of data to process.
            batch_size (int): Number of items per batch.
            operation (Callable): Operation to perform on each batch.
            log_message (str): Log message to use after processing each batch.
        """
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch = data[start:end]
            operation(batch)
            logger.info(log_message)

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
            return True  
        

    def get_collection_points_count(self, collection_name: str) -> int:
        """
        Get the number of points in the specified collection.

        Args:
            collection_name (str): Name of the collection.

        Returns:
            int: The number of points in the collection.
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting collection points count: {str(e)}")
            return 0



