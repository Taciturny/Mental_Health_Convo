import sys
from pathlib import Path
project_root = Path(__file__).parent  # Point to the 'src' directory
sys.path.append(str(project_root))

import re
import numpy as np
from typing import List, Tuple, Dict
import tqdm
import logging
from src.core.embeddings_model import EmbeddingsModel
from src.core.data_loader import DataLoader
from src.core.qdrant_manager import QdrantManager



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def initialize_qdrant(collection_name: str, host: str = "localhost", port: int = 6333) -> QdrantManager:
    qdrant_manager = QdrantManager(host=host, port=port)
    qdrant_manager.create_collection_if_not_exists(collection_name)
    return qdrant_manager


def load_and_embed_data(file_path: str, batch_size: int = 100) -> Tuple[List[Dict[str, str]], List[np.ndarray], List[np.ndarray]]: # you can increase  the batch no
    data_loader = DataLoader(file_path)
    embeddings_model = EmbeddingsModel.get_instance()
    
    all_data = []
    all_dense_embeddings = []
    all_late_embeddings = []

    logger.info(f"Starting to load and embed data from {file_path}")
    for i, batch in enumerate(tqdm.tqdm(data_loader.load_data_in_batches(batch_size), desc="Loading and embedding data")):
        logger.info(f"Processing batch {i+1}, batch size: {len(batch)}")
        all_data.extend(batch)
        texts = [item['question'] for item in batch]  
        dense_embeddings, late_embeddings = embeddings_model.embeddings(texts)
        all_dense_embeddings.extend(dense_embeddings)
        all_late_embeddings.extend(late_embeddings)
    
    logger.info(f"Finished loading and embedding. Total data points: {len(all_data)}")
    return all_data, all_dense_embeddings, all_late_embeddings


def upload_data_to_qdrant(qdrant_manager: QdrantManager, collection_name: str, data: List[Dict[str, str]], dense_embeddings: List[np.ndarray], late_embeddings: List[np.ndarray]):
    qdrant_manager.prepare_and_upload_points(data, dense_embeddings, late_embeddings, qdrant_manager.client, collection_name)



def is_relevant_query(query: str, relevant_keywords: List[str]) -> bool:
    """
    Check if the query is relevant based on a list of keywords.
    
    Args:
    query (str): The user's input query
    relevant_keywords (List[str]): List of relevant keywords
    
    Returns:
    bool: True if the query is relevant, False otherwise
    """
    query_words = re.findall(r'\w+', query.lower())
    return any(keyword in query_words for keyword in relevant_keywords)
