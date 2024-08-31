import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import re
import logging
# import traceback
from typing import List, Dict, Any
from qdrant_client.http import models
from .qdrant_manager import QdrantManager
from .embeddings_model import EmbeddingsModel
from .llm_model import EnsembleModel  
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
    """Manages search operations using Qdrant vector database and text embeddings."""

    def __init__(self, collection_name: str, vector_size: int = 384, host: str = "localhost", port: int = 6333):
        self.collection_name = collection_name
        self.vector_size = vector_size
        # Ensure host is a string and port is an integer
        if not isinstance(host, str):
            raise ValueError("Host must be a string.")
        if not isinstance(port, int):
            raise ValueError("Port must be an integer.")
        
        self.qdrant_manager = QdrantManager(host=host, port=port)  # Updated: removed collection_name here
        self.embeddings_model = EmbeddingsModel.get_instance()
        self.llm_model = EnsembleModel.get_instance()
        self._init_collection()

    def _init_collection(self, create_new: bool = False):
        try:
            if create_new:
                self.qdrant_manager.create_collection(self.collection_name, self.vector_size)
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                # Check if collection exists, create only if it doesn't
                if not self.qdrant_manager.collection_exists(self.collection_name):
                    self.qdrant_manager.create_collection(self.collection_name, self.vector_size)
                    logger.info(f"Created new collection: {self.collection_name}")
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise


    def vector_search(self, query: str, limit: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embeddings_model.get_embeddings([query])[0]
            search_results = self.qdrant_manager.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=threshold
            )
            results = [
                {
                    "id": hit.id,
                    "question": hit.payload["question"],
                    "answer": hit.payload["answer"],
                    "score": hit.score
                }
                for hit in search_results
            ]
            logger.info(f"Vector search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
            raise
    def keyword_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            # Split the query into individual words
            keywords = query.lower().split()

            # Create a keyword filter
            keyword_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="question",
                        match=models.MatchText(text=keyword)
                    ) for keyword in keywords
                ] + [
                    models.FieldCondition(
                        key="answer",
                        match=models.MatchText(text=keyword)
                    ) for keyword in keywords
                ]
            )

            search_results = self.qdrant_manager.client.search(
                collection_name=self.collection_name,
                query_filter=keyword_filter,
                limit=limit,
                query_vector=[0.0] * 384 
            )

            results = []
            for hit in search_results:
                # Calculate a simple score based on keyword matches
                question_matches = sum(keyword in hit.payload["question"].lower() for keyword in keywords)
                answer_matches = sum(keyword in hit.payload["answer"].lower() for keyword in keywords)
                total_matches = question_matches + answer_matches
                score = total_matches / len(keywords)  # Normalize score

                results.append({
                    "id": hit.id,
                    "question": hit.payload["question"],
                    "answer": hit.payload["answer"],
                    "score": score
                })

            # Sort results by score in descending order
            results.sort(key=lambda x: x["score"], reverse=True)

            logger.info(f"Keyword search completed. Found {len(results)} results.")
            return results

        except Exception as e:
            logger.error(f"Error performing keyword search: {str(e)}")
            raise

    def hybrid_search(self, query: str, limit: int = 5, threshold: float = 0.5, boost: float = 0.5) -> List[Dict[str, Any]]:
        try:
            vector_results = self.vector_search(query, limit=limit, threshold=threshold)
            keyword_results = self.keyword_search(query, limit=limit)

            # Combine results
            results = vector_results + keyword_results
            
            # Re-rank results
            for result in results:
                vector_score = result['score']
                keyword_match = any(
                    keyword in result['question'].lower() or keyword in result['answer'].lower()
                    for keyword in query.lower().split()
                )
                if keyword_match:
                    result['score'] = vector_score * (1 - boost) + boost
                else:
                    result['score'] = vector_score * (1 - boost)

            # Remove duplicates and sort
            unique_results = {result['id']: result for result in results}.values()
            sorted_results = sorted(unique_results, key=lambda x: x['score'], reverse=True)[:limit]

            logger.info(f"Hybrid search completed. Found {len(sorted_results)} results.")
            return sorted_results
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            raise

    def compute_relevance(self, query: str, response: str) -> float:
        query_embedding = self.embeddings_model.get_embeddings([query])[0]
        response_embedding = self.embeddings_model.get_embeddings([response])[0]
        similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
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
        relevance_score = self.compute_relevance(query, response)
        fluency_score = self.llm_model.compute_fluency(response)
        empathy_score = self.llm_model.compute_empathy(response)
        
        # Combine the scores with weights:
        # - Relevance (0.3): Ensures the response is directly relevant to the user's query.
        # - Fluency (0.2): Assesses how naturally and coherently the response is phrased.
        # - Empathy (0.5): Prioritizes understanding and compassion, which is crucial for mental health contexts.
        #   Higher weight on empathy reflects the importance of providing supportive and sensitive responses.
        combined_score = 0.3 * relevance_score + 0.2 * fluency_score + 0.5 * empathy_score
        return combined_score

    def rag(self, query: str, max_results: int = 5, threshold: float = 0.5) -> Dict[str, Any]:
        try:
            # Step 1: Perform hybrid search
            search_results = self.hybrid_search(query, limit=max_results, threshold=threshold)

            # Step 2: Generate context from search results
            context = self.generate_context(search_results)

            # Step 3: Generate response using ensemble model
            prompt = self.construct_prompt(query, context)
            generated_responses = self.llm_model.generate_text(prompt, max_new_tokens=200, num_return_sequences=1)

            # Step 4: Post-process and format the response
            best_response = self.post_process_response(generated_responses[0], query)
            formatted_response = self.format_response(best_response)

            # Step 5: Prepare the final result
            result = {
                "query": query,
                "response": formatted_response,
                "search_results": search_results,
                "confidence_score": self.compute_confidence_score(formatted_response, query, context)
            }

            return result

        except Exception as e:
            logger.error(f"Error in RAG search: {str(e)}")
            return self.fallback_to_hybrid(query)

    def generate_context(self, search_results):
        return "\n".join([f"Q: {result['question']}\nA: {result['answer']}" for result in search_results])

    def construct_prompt(self, query, context):
        return f"Context:\n{context}\n\nUser Query: {query}\n\nAs an AI assistant specializing in mental health, please provide a helpful and empathetic response:"

    def compute_confidence_score(self, response, query, context):
        # Implement a simple confidence score based on response length and context relevance
        response_length_score = min(len(response) / 100, 1)  # Normalize to 0-1
        context_relevance_score = self.compute_relevance(query, context)
        return (response_length_score + context_relevance_score) / 2

    def fallback_to_hybrid(self, query: str) -> Dict[str, Any]:
        results = self.hybrid_search(query, limit=1, threshold=0.5)
        if results:
            best_result = results[0]
            fallback_response = f"Based on the information I have, {best_result['answer']}"
            return {
                "response": self.format_response(fallback_response), 
                "search_results": results,
                "confidence_score": 0.5  # Default confidence score for fallback
            }
        else:
            return {
                "response": "I'm sorry, but I couldn't find any relevant information to answer your query. Could you please rephrase your question or provide more details?",
                "search_results": [],
                "confidence_score": 0.1  # Low confidence score when no results found
            }
        
    def post_process_response(self, response: str, query: str) -> str:
        # Remove common prompt artifacts and clean up the response
        artifacts = ["Response:", "Q:", "q:", "User Query:", "A:", "a:", "Context:", "Question:", "Answer:"]
        for artifact in artifacts:
            response = response.replace(artifact, "")
        
        # Split the response into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        
        # Capitalize the first letter of each sentence and join them
        cleaned_sentences = [sentence.strip().capitalize() for sentence in sentences if sentence.strip()]
        final_response = ' '.join(cleaned_sentences)
        
        if not final_response:
            return self.fallback_to_hybrid(query)["response"]
        
        return final_response


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

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.

        Returns:
            Dict[str, Any]: Collection statistics.
        """
        try:
            # Force a refresh of the collection info
            self.qdrant_manager.client.refresh_collection(collection_name=self.collection_name)
            
            info = self.qdrant_manager.get_collection_info(self.collection_name)
            stats = {
                "name": self.collection_name,
                "vector_size": self.vector_size,
                "total_vectors": info.vectors_count if info.vectors_count is not None else 0,
                "points_count": info.points_count if info.points_count is not None else 0
            }
            logger.info(f"Retrieved collection stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            # Return default stats if there's an error
            return {
                "name": self.collection_name,
                "vector_size": self.vector_size,
                "total_vectors": 0,
                "points_count": 0
            }
        
    def clear_collection(self):
        """Clear all data from the current collection."""
        try:
            self.qdrant_manager.client.delete_collection(self.collection_name)
            self._init_collection()
            logger.info(f"Collection {self.collection_name} cleared and reinitialized")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
