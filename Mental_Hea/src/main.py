import sys
from pathlib import Path
project_root = Path(__file__).parent  # Point to the 'src' directory
sys.path.append(str(project_root))

import logging
from core.utils import initialize_qdrant, load_and_embed_data, upload_data_to_qdrant, is_relevant_query
from core.search_engine import SearchEngine

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
    collection_name = "mental_health_qa"
    file_path = "./data/preprocessed_data.parquet"

    # Define a list of relevant keywords for mental health queries
    relevant_keywords = [
        'depression', 'anxiety', 'stress', 'therapy', 'mental', 'health',
        'counseling', 'psychiatry', 'psychology', 'disorder', 'treatment',
        'medication', 'symptoms', 'diagnosis', 'support', 'wellbeing'
    ]
    
    
    try:
        # Initialize Qdrant
        qdrant_manager = initialize_qdrant(collection_name)
        logger.info("Qdrant initialized successfully.")

        # Check if collection is empty
        if qdrant_manager.collection_is_empty(collection_name):
            # Load and embed data
            data, dense_embeddings, late_embeddings = load_and_embed_data(file_path)
            logger.info(f"Loaded and embedded {len(data)} data points.")

            # Upload data to Qdrant
            upload_data_to_qdrant(qdrant_manager, collection_name, data, dense_embeddings, late_embeddings)
            logger.info("Data uploaded to Qdrant successfully.")
        else:
            logger.info("Collection already contains data. Skipping embedding and uploading.")

        # Initialize search engine
        search_engine = SearchEngine(collection_name)
        logger.info("Search engine initialized.")

        # Query input loop
        while True:
            query = input("Enter your search query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                print("Thank you for using the Mental Health QA system. Goodbye!")
                break

            # Check if the query is relevant
            if not is_relevant_query(query, relevant_keywords):
                print("I'm sorry, but your question doesn't seem to be related to mental health. Please try asking a question about mental health topics.")
                continue


            # uncomment if you want to print the individual searches
            try:
                # # Perform dense search  
                # dense_results = search_engine.search_dense(query_text=query)
                # print("\nDense Search Results:")
                # print_results(dense_results, query)

                # # Perform late interaction search
                # late_results = search_engine.search_late(query_text=query)
                # print("\nLate Interaction Search Results:")
                # print_results(late_results, query)

                # Perform hybrid search
                # hybrid_results = search_engine.search_hybrid(query_text=query)
                # print("\nHybrid Search Results:")
                # print_results(hybrid_results, query)


                rag_result = search_engine.rag(query=query)
                print("\nRAG Result:")
                print(f"Query: {rag_result['query']}")
                print(f"Response: {rag_result['response']}")
                print(f"Confidence Score: {rag_result['confidence_score']}")
                print("\nTop search results:")
                print_results(rag_result['search_results'])
            
            except Exception as e:
                logger.error(f"An error occurred during the search: {str(e)}", exc_info=True)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


def print_results(results):
    if results and hasattr(results, 'points') and results.points:
        print(f"Top {len(results.points)} results:")
        for point in results.points:
            payload = point.payload
            print(f"Question: {payload.get('question', 'N/A')}")
            print(f"Answer: {payload.get('answer', 'N/A')}")
            print(f"Score: {point.score}\n")
    else:
        print("No results found")


if __name__ == "__main__":
    main()
