import logging
from typing import List, Tuple

import numpy as np
from fastembed import TextEmbedding

# from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding

logger = logging.getLogger(__name__)


class EmbeddingsModel:
    """Manages the embeddings models for dense and sparse text embeddings."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingsModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _initialize(self):
        try:
            self.dense_embedding_model = TextEmbedding(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            # self.sparse_embedding_model = Bm25("Qdrant/bm25")
            self.late_interaction_embedding_model = (
                LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
            )
            logger.info("All embedding models initialized successfully.")
        except Exception as e:
            logger.error(
                f"Error initializing embeddings models: {str(e)}",
                exc_info=True,
            )
            raise

    def embeddings(
        self, inputs: List[str], batch_size: int = 32
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate dense and sparse embeddings for a list of input texts in batches.

        Args:
            inputs (List[str]): List of text inputs for which to generate embeddings.
            batch_size (int): Size of each batch. Default is 32.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Dense embeddings and late interaction embeddings.
        """
        dense_embeddings = []
        late_embeddings = []

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]

            # Generate embeddings for the current batch
            dense_batch = self.dense_embedding_model.embed(batch_inputs)
            late_batch = self.late_interaction_embedding_model.embed(
                batch_inputs
            )

            dense_embeddings.extend(dense_batch)
            late_embeddings.extend(late_batch)

        return dense_embeddings, late_embeddings
