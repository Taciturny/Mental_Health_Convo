import cohere
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class CohereModel:
    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)

    def generate_response(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        try:
            response = self.client.generate(
                model='command',
                prompt=prompt,
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.7),
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )
            return response.generations[0].text
        except Exception as e:
            logger.error(f"Error generating response with Cohere: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request. Could you please try again?"
