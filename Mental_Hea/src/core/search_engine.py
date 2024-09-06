import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import re
import logging
import numpy as np
from typing import Dict, Any
from .qdrant_manager import QdrantManager
from .embeddings_model import EmbeddingsModel
from qdrant_client import QdrantClient, models
from .llm_model import EnsembleModel  
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
    """Manages hybrid search operations using Qdrant vector database and text embeddings."""

    def __init__(self, collection_name: str, host: str = "localhost", port: int = 6333):
        self.collection_name = collection_name
        self.qdrant_client = QdrantManager(host=host, port=port)
        self.embeddings_model = EmbeddingsModel.get_instance()
        self.client = QdrantClient("http://localhost:6333")
        self.llm_model = EnsembleModel.get_instance()

    def _init_collection(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            if not self.qdrant_client.collection_exists(self.collection_name):
                self.qdrant_client.create_collection(self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise


    def search_dense(self, query_text: str):
        try:
            dense_embeddings, _ = self.embeddings_model.embeddings([query_text])
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_embeddings[0],
                using="text-dense",
                limit=3,
                with_payload=True
            )
            return results
        except Exception as e:
            logger.error(f"Error performing dense search: {str(e)}")
            raise

    def search_late(self, query_text: str):
        try:
            _, late_embeddings = self.embeddings_model.embeddings([query_text])
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=late_embeddings[0],
                using="text-late",
                limit=2,
                with_payload=True
            )
            return results
        except Exception as e:
            logger.error(f"Error performing late interaction search: {str(e)}")
            raise


    def search_hybrid(self, query_text: str):
        try:
            # Get embeddings from your model
            dense_embeddings, late_embeddings = self.embeddings_model.embeddings([query_text])
            
            # Prepare prefetch query for initial retrieval
            prefetch = [
                models.Prefetch(
                    query=dense_embeddings[0],
                    using="text-dense",
                    limit=30,  # Retrieve top 20 for reranking
                ),
            ]
            
            # Perform combined search with initial retrieval and reranking
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=late_embeddings[0],  
                using="text-late",  
                limit=5,  # Final limit after reranking
                with_payload=True
            )

            return results

        except Exception as e:
            logger.error(f"Error performing hybrid search with reranking: {str(e)}")
            raise

    def compute_relevance(self, query: str, response: str) -> float:
        # Get dense embeddings for query and response
        query_dense, _ = self.embeddings_model.embeddings([query])
        response_dense, _ = self.embeddings_model.embeddings([response])
        
        # Ensure embeddings are 2D arrays
        query_dense = np.array(query_dense).reshape(1, -1)
        response_dense = np.array(response_dense).reshape(1, -1)
        
        # Compute similarity using dense embeddings
        similarity = cosine_similarity(query_dense, response_dense)[0][0]
        
        return similarity

    def score_response(self, response: str, query: str) -> float:
        """
        Scores a response based on relevance, fluency, and empathy.
        Args:
        - response: The generated response.
        - query: The user's query.
        Returns:
        - A score combining relevance, fluency, and empathy metrics.
        """
        relevance_score = self.compute_relevance(query, response)    # - Relevance (0.3): Ensures the response is directly relevant to the user's query.
        fluency_score = self.llm_model.compute_fluency(response)     # - Fluency (0.2): Assesses how naturally and coherently the response is phrased.
        empathy_score = self.llm_model.compute_empathy(response)      # - Empathy (0.5): Prioritizes understanding and compassion, which is crucial for mental health contexts.
        
        # Combine the scores with weights:       
        combined_score = 0.3 * relevance_score + 0.2 * fluency_score + 0.5 * empathy_score   #   Higher weight on empathy reflects the importance of providing supportive and sensitive responses.
        return combined_score

    def compute_confidence_score(self, response, query, context):
        # Implement a simple confidence score based on response length and context relevance
        response_length_score = min(len(response) / 100, 1)  # Normalize to 0-1
        context_relevance_score = self.compute_relevance(query, context)
        return (response_length_score + context_relevance_score) / 2



    def format_response(self, response: str) -> str:
        # Remove any prompt artifacts
        response = response.split("Response:")[-1].strip()
        
        # Split into sentences and rejoin with proper spacing
        sentences = response.split('.')
        formatted_response = '. '.join(sentence.strip().capitalize() for sentence in sentences if sentence.strip())
        
        # Split into paragraphs and rejoin with proper spacing
        paragraphs = formatted_response.split('\n')
        formatted_response = '\n\n'.join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())
        
        return formatted_response

    def fallback_to_search(self, query: str, search_type: str) -> Dict[str, Any]:
        if search_type == 'dense':
            results = self.search_dense(query)
        elif search_type == 'late':
            results = self.search_late(query)
        else:  # default to hybrid
            results = self.search_hybrid(query)

        if results.points:
            best_result = results.points[0].payload
            fallback_response = f"Based on the information I have, {best_result.get('answer', 'No specific answer found.')}"
            return {
                "query": query,
                "response": self.format_response(fallback_response),
                "search_results": results,
                "search_type": search_type,
                "model_type": "fallback",
                "confidence_score": 0.5  # Default confidence score for fallback
            }
        else:
            return {
                "query": query,
                "response": "I'm sorry, but I couldn't find any relevant information to answer your query. Could you please rephrase your question or provide more details?",
                "search_results": results,
                "search_type": search_type,
                "model_type": "fallback",
                "confidence_score": 0.1  # Low confidence score when no results found
            }

    def construct_prompt(self, query, context):
        return f"Context:\n{context}\n\nUser Query: {query}\n\nAs an AI assistant specializing in mental health, please provide a helpful and empathetic response:"


    def rag(self, query: str, search_type: str = 'hybrid', model_type: str = 'ensemble') -> Dict[str, Any]:
        try:
            # Step 1: Perform search based on the specified type
            if search_type == 'dense':
                search_results = self.search_dense(query)
            elif search_type == 'late':
                search_results = self.search_late(query)
            else:  # default to hybrid
                search_results = self.search_hybrid(query)

            # Step 2: Generate context from search results
            context = self.generate_context(search_results)

            # Step 3: Construct a more focused prompt
            prompt = f"""
            Given the following context and question, provide a concise and relevant answer:

            Context: {context}

            Question: {query}

            Answer:
            """

            # Step 4: Generate response using specified model or ensemble
            generated_responses = self.llm_model.generate_text(prompt, model_name=model_type)

            # Step 5: Post-process and format the response
            best_response = self.post_process_response(generated_responses[0], query)

            # Step 6: Prepare the final result
            result = {
                "query": query,
                "response": best_response,
                "search_results": search_results,
                "search_type": search_type,
                "model_type": model_type,
                "confidence_score": self.compute_confidence_score(best_response, query, context)
            }

            return result

        except Exception as e:
            logger.error(f"Error in RAG search: {str(e)}", exc_info=True)
            return self.fallback_to_search(query, search_type)

    def generate_context(self, search_results):
        context = ""
        for point in search_results.points:
            context += f"Q: {point.payload['question']}\nA: {point.payload['answer']}\n\n"
        return context.strip()

    def post_process_response(self, response: str, query: str) -> str:
        # Remove any generated question-like phrases
        response = re.sub(r'^(Question:|Q:).*?\n', '', response, flags=re.IGNORECASE|re.MULTILINE)
        
        # Remove the query from the beginning of the response if it's there
        response = re.sub(f'^{re.escape(query)}:\s*', '', response, flags=re.IGNORECASE)
        
        # Capitalize the first letter and ensure the response ends with a period
        response = response.strip().capitalize()
        if not response.endswith('.'):
            response += '.'
        
        return response
