import sys
from pathlib import Path
project_root = Path(__file__).parent  # Point to the 'src' directory
sys.path.append(str(project_root))

import logging
from core.utils import initialize_components, index_data

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



def main():
    # Configuration
    COLLECTION_NAME = "mental_health_qa"
    VECTOR_SIZE = 384  # Size of all-MiniLM-L12-v2 embeddings
    HOST = "localhost"
    PORT = 6333
    DATA_PATH = "./data/preprocessed_data.parquet"

    try:
        # Initialize components
        search_engine, data_loader, embeddings_model = initialize_components(COLLECTION_NAME, VECTOR_SIZE, HOST, PORT, DATA_PATH)


    # Check if collection exists and is empty
        collection_exists = search_engine.qdrant_manager.collection_exists(COLLECTION_NAME)
        if not collection_exists:
            logger.info("Collection doesn't exist. Creating and indexing data...")
            search_engine.qdrant_manager.create_collection(COLLECTION_NAME, VECTOR_SIZE)
            index_data(search_engine, data_loader, embeddings_model)
        elif search_engine.qdrant_manager.collection_is_empty(COLLECTION_NAME):
            logger.info("Collection is empty. Indexing data...")
            index_data(search_engine, data_loader, embeddings_model)
        else:
            logger.info("Collection already exists and contains data. Skipping indexing.")

        # User interaction loop
        while True:
            query = input("Enter your mental health question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                print("Thank you for using the Mental Health QA system. Goodbye!")
                break

            # Perform RAG search
            logger.info(f"Performing RAG search for query: '{query}'")
            try:
                rag_response = search_engine.rag(query)
                print(f"\nRAG response for query '{query}':")
                if rag_response:
                    print(rag_response)
                else:
                    print("No response generated from RAG search.")
            except Exception as e:
                logger.error(f"RAG search failed: {str(e)}")
                print("RAG search failed. Falling back to hybrid search results.")
                fallback_response = search_engine.fallback_to_hybrid(query)
                print(fallback_response)

    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    main()
