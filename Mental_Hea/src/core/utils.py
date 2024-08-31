import sys
from pathlib import Path
project_root = Path(__file__).parent  # Point to the 'src' directory
sys.path.append(str(project_root))

from typing import Tuple
import logging
from src.core.search_engine import SearchEngine
from src.core.embeddings_model import EmbeddingsModel
from src.core.data_loader import DataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_components(collection_name: str, vector_size: int, host: str, port: int, data_path: str) -> Tuple[SearchEngine, DataLoader, EmbeddingsModel]:
    """
    Initialize all necessary components for the mental health Q&A system.

    Args:
        collection_name (str): Name of the Qdrant collection.
        vector_size (int): Size of the embedding vectors.
        host (str): Qdrant host address.
        port (int): Qdrant port number.
        data_path (str): Path to the parquet file containing the data.

    Returns:
        Tuple[SearchEngine, DataLoader, EmbeddingsModel]: Initialized components.
    """
    try:
        search_engine = SearchEngine(collection_name, vector_size, host, port)
        data_loader = DataLoader(data_path)
        embeddings_model = EmbeddingsModel.get_instance()
        return search_engine, data_loader, embeddings_model
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def index_data(search_engine: SearchEngine, data_loader: DataLoader, embeddings_model: EmbeddingsModel):
    """
    Load data, generate embeddings, and index it in Qdrant.

    Args:
        search_engine (SearchEngine): Initialized SearchEngine instance.
        data_loader (DataLoader): Initialized DataLoader instance.
        embeddings_model (EmbeddingsModel): Initialized EmbeddingsModel instance.
    """
    try:
        data = data_loader.load_data()
        questions = [item[1] for item in data]  # Assuming the question is the second item in each tuple
        embeddings = embeddings_model.get_embeddings(questions)
        search_engine.qdrant_manager.index_and_insert_data(search_engine.collection_name, data, embeddings)
    except Exception as e:
        logger.error(f"Error indexing data: {str(e)}")
        raise
