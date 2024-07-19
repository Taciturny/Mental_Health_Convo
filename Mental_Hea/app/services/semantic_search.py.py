# import weaviate

# def semantic_search(query):
#     client = weaviate.Client("http://localhost:8080")
#     response = client.query.get("MentalHealth", ["context", "response"]) \
#         .with_near_text({"concepts": [query]}).do()
#     return response['data']['Get']['MentalHealth']

# if __name__ == "__main__":
#     results = semantic_search("I feel anxious")
#     print(results)
from app.models.llm_model import LLMModel

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L12-v2"):
        self.llm_model = LLMModel(model_name)

    def get_semantic_vector(self, text):
        return self.llm_model.get_embeddings(text)
