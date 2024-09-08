import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import re
import numpy as np
from typing import List, Tuple, Dict
import tqdm
import logging
from transformers import pipeline
from src.core.embeddings_model import EmbeddingsModel
from src.core.data_loader import DataLoader
from src.core.qdrant_manager import QdrantManager
from .config import settings



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def initialize_qdrant(collection_name: str) -> QdrantManager:
    # Check if deployment mode is cloud or local
    if settings.DEPLOYMENT_MODE == 'cloud':
        # Use cloud configuration
        qdrant_manager = QdrantManager(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    else:
        # Use local configuration
        qdrant_manager = QdrantManager(host="localhost", port=6333)
    
    # Create collection if it does not exist
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


# Global sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis")

def is_relevant_query(query: str, relevant_keywords: List[str]) -> bool:
    """
    Check if the query is relevant based on keywords and sentiment analysis.
    
    Args:
    query (str): The user's input query
    relevant_keywords (List[str]): List of relevant keywords
    
    Returns:
    bool: True if the query is relevant, False otherwise
    """
    # Check for exact keyword matches
    query_words = set(query.lower().split())
    if any(keyword in query_words for keyword in relevant_keywords):
        return True
    
    # Use sentiment analysis
    result = sentiment_analyzer(query)[0]
    
    # Consider negative sentiment as potentially relevant to mental health
    if result['label'] == 'NEGATIVE' and result['score'] > 0.7:
        return True
    
    return False


def merge_search_results(dense_results, late_results, hybrid_results, weights=(0.3, 0.3, 0.4)):
    all_results = {}
    for results, weight in zip([dense_results, late_results, hybrid_results], weights):
        for result in results:
            if result.id not in all_results:
                all_results[result.id] = {'score': 0, 'payload': result.payload}
            all_results[result.id]['score'] += result.score * weight
    
    merged_results = sorted(all_results.items(), key=lambda x: x[1]['score'], reverse=True)
    return [{'id': id, 'score': data['score'], 'payload': data['payload']} for id, data in merged_results]
