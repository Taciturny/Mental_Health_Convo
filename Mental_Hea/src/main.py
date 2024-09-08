import sys
from pathlib import Path
project_root = Path(__file__).parent  # Point to the 'src' directory
sys.path.append(str(project_root))

import logging
from core.utils import initialize_qdrant, load_and_embed_data, upload_data_to_qdrant, is_relevant_query
from core.search_engine import SearchEngine
from core.llm_model import EnsembleModel
from core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='chatbot.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def main():
    relevant_keywords = [
        'depression', 'anxiety', 'stress', 'therapy', 'mental', 'health',
        'counseling', 'psychiatry', 'psychology', 'disorder', 'treatment',
        'medication', 'symptoms', 'diagnosis', 'support', 'wellbeing', 'burn-out'
    ]
    
    try:
        # Initialize Qdrant
        qdrant_manager = initialize_qdrant(settings.COLLECTION_NAME_LOCAL)
        logger.info("Qdrant initialized successfully.")

        # Check if collection is empty
        if qdrant_manager.collection_is_empty(settings.COLLECTION_NAME_LOCAL):
            # Load and embed data
            data, dense_embeddings, late_embeddings = load_and_embed_data(settings.DATA_FILE_PATH)
            logger.info(f"Loaded and embedded {len(data)} data points.")

            # Upload data to Qdrant
            upload_data_to_qdrant(qdrant_manager, settings.COLLECTION_NAME_LOCAL, data, dense_embeddings, late_embeddings)
            logger.info("Data uploaded to Qdrant successfully.")
        else:
            logger.info("Collection already contains data. Skipping embedding and uploading.")

        # Initialize search engine and EnsembleModel
        search_engine = SearchEngine(settings.COLLECTION_NAME_LOCAL)
        ensemble_model = EnsembleModel.get_instance()
        logger.info("Search engine and EnsembleModel initialized.")

        # Query input loop
        while True:
            query = input("Enter your search query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                print("Thank you for using the Mental Health QA system. Goodbye!")
                break

            # Check if the query is relevant using the improved function
            if not is_relevant_query(query, relevant_keywords):
                print("Your question might not be directly related to mental health. However, I'll try to provide a helpful response.")
            
            try:
                # Perform RAG search
                rag_result = search_engine.rag(query=query, search_type='hybrid', model_type='ensemble')
                print("\nRAG Result:")
                print(f"Query: {rag_result['query']}")
                print(f"Response: {rag_result['response']}")
                print(f"Confidence Score: {rag_result['confidence_score']}")
                print(f"Search Type: {rag_result['search_type']}")
                print(f"Model Type: {rag_result['model_type']}")
                print("\nTop search results:")
                print_results(rag_result['search_results'])
                
                # Ask for feedback
                feedback = input("Was this response helpful? (yes/no): ").lower()
                if feedback == 'no':
                    print("I apologize that the response wasn't helpful. Please try rephrasing your question or ask something more specific about mental health.")
            
            except Exception as e:
                logger.error(f"An error occurred during the search: {str(e)}", exc_info=True)
                print("I apologize, but an error occurred while processing your query. Please try again.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

def print_results(results):
    if results and hasattr(results, 'points') and results.points:
        for point in results.points:
            payload = point.payload
            print(f"Question: {payload.get('question', 'N/A')}")
            print(f"Answer: {payload.get('answer', 'N/A')}")
            print(f"Score: {point.score}\n")
    else:
        print("No results found")

if __name__ == "__main__":
    main()
