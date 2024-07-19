from transformers import AutoTokenizer, AutoModel
import torch

class LLMModel:
    def __init__(self, model_name="all-MiniLM-L12-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings


