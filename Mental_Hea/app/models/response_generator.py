from transformers import AutoTokenizer, AutoModel
from app.models.llm_model import LLMModel

class ResponseGenerator:
    def __init__(self, query_model_name="all-MiniLM-L12-v2", gen_model_name="gpt-2"):
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self.query_model = AutoModel.from_pretrained(query_model_name)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.gen_model = AutoModel.from_pretrained(gen_model_name)
        self.llm_model = LLMModel(query_model_name)

    def get_response(self, user_query, client):
        # Construct and send the GraphQL query to Weaviate
        query = """
        {
          Get {
            MentalHealthConversation(
              limit: 1,
              nearText: {
                concepts: ["multiple issues in counseling"],
              }
            ) {
              context
              response
            }
          }
        }
        """
        
        response = client.query.raw(query)
        retrieved_context = response['data']['Get']['MentalHealthConversation'][0]['context']

        # Generate the prompt and response
        prompt = f"Context: {retrieved_context}\n\nUser Query: {user_query}\n\nResponse:"
        inputs = self.gen_tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True)
        outputs = self.gen_model.generate(**inputs, max_length=100)
        response_text = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response_text
