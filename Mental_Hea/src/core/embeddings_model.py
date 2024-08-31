from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingsModel:
    """Manages the embeddings model for text encoding."""

    _instance = None

    def __new__(cls):
        """Create or return the singleton instance of EmbeddingsModel."""
        if cls._instance is None:
            cls._instance = super(EmbeddingsModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of EmbeddingsModel."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _initialize(self):
        """Initialize the embeddings model."""
        try:
            logger.info("Initializing embeddings model")
            self.model = SentenceTransformer('all-MiniLM-L12-v2') # all-MiniLM-L6-v2
        except Exception as e:
            logger.error(f"Error initializing embeddings model: {str(e)}")
            raise  # Re-raise the exception instead of setting self.model to None

    def get_embeddings(self, texts):
        """
        Generate embeddings for the given texts.

        Args:
            texts (List[str]): List of texts to encode.

        Returns:
            List[numpy.ndarray]: List of embeddings.
        """
        if self.model is None:
            raise ValueError("Embeddings model is not initialized")
        try:
            return self.model.encode(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise  # Re-raise the exception instead of returning an empty list
