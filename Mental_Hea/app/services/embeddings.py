from sentence_transformers import SentenceTransformer, util
import os

class EmbeddingModel:
    """Singleton class for managing the sentence embedding model."""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        model_name = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        if isinstance(text, str):
            text = [text]  # Convert single string to list
        return self.model.encode(text).tolist()
    
    def classify(self, query, response):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        response_embedding = self.model.encode(response, convert_to_tensor=True)
    
        similarity = util.pytorch_cos_sim(query_embedding, response_embedding)
        
        if similarity > 0.7:
            return "relevant"
        elif similarity > 0.5:
            return "somewhat relevant"
        else:
            return "not relevant"
